"""demars_core.api -- the public entry point.

    from demars_core import deaverage
    rec = deaverage("mystructure.cif", calculator="sevennet", out_dir="./mar")
    print(rec.summary())

`deaverage(source, calculator=...)` runs the DeMARS engine on an arbitrary
disordered CIF/Structure, with no ICSD id anywhere -- via the engine's real,
formal decoupled signature (NO monkeypatching):

    structure = engine.structure_from_text(cif_text)
    evidence  = evidence.evidence_from_text(cif_text)     # iid=None -> no ICSD sibling search
    relax     = <native SevenNet | generic-ASE relaxer>   # pluggable MLIP backend
    engine.run_self_driving(structure=structure, evidence=evidence, relax=relax, ...)

The ICSD id is now just one of several ways the engine can obtain
(structure, evidence, relax); the local ICSD workflow -- run_self_driving(iid) --
still works unchanged because those parameters fall back to the ICSD loaders when
absent.

INTERNAL CONTRACT -- what `tools/` depends on (this is NOT the public API)
-------------------------------------------------------------------------
`deaverage()` is the published interface, but the DeMARS stage pipeline in `tools/` does
NOT run through it: each stage needs a piece separately (evidence alone, a relax of one
given structure, record assembly), so those CLIs import private names. The layering is
deliberate -- engine layer vs research layer -- but the consequence is easy to miss:

    the surface a 1,000-entry campaign actually rests on is not the one `__all__`
    and `README.md` describe, and nothing tests the difference.

Changing any of the following breaks `tools/` silently:

  api._make_relax(calculator, d3, rigid_units) -> (relax, mode, prov)   tools/demars_engine.py
      the `mode` TAG is a value dependency: the caller decides from it whether to run the
      final tier. Single-sourced as `api.FINAL_TIER_MODES` so the two copies cannot drift.
  api._final_omni(built, d3) -> (final_MAR dict, relaxed atoms)         tools/demars_engine.py
      `built` is a SHAPE dependency: --from mode has no enumeration, so it hand-assembles
      the 5-tuple (confs, rel, val, samples, info) this function unpacks.
  _engine.mar_engine.structure_from_text, .run_self_driving             tools/demars_engine.py
  _engine.mar_evidence.evidence_from_text                               tools/demars_evidence.py
  _engine.mar_record.build_record                                       tools/demars_record.py

Prefer widening a public signature to adding a sixth entry here. The connectivity gate
used to be on this list; it is `demars_core.connectivity` now, which is why it isn't.
"""
import os

from ._engine import mar_engine as engine
from ._engine import mar_evidence as evidence
from . import _torchsim as backend
from . import _checkpoint as _ckpt
from .io import to_cif_text
from .record import MARRecord, write_custom_ensemble
from . import calculators as _calc

# `write_custom_ensemble` is re-exported here because this is where a bespoke build script looks:
# it already imports `_make_relax` / `_final_omni` from this module. A custom build that cannot
# find the file contract will not write it.
__all__ = ['deaverage', 'write_custom_ensemble']

# The `mode` tags `_make_relax` can return for which the omni-mpa final tier is available
# (SevenNet-native weights). Named rather than inlined because `tools/demars_engine.py` --from
# mode makes the SAME decision on the SAME tags: two copies of the literal would drift, and the
# failure is silent -- a renamed tag matches neither copy and the final tier just stops running.
FINAL_TIER_MODES = ('sevennet', 'sevennet-ase-rigid')


def _make_relax(calculator, d3, rigid_units=False, max_steps=None):
    """Build the engine's `relax=` callable (confs -> (E, rel, val)) for the chosen backend, plus a
    mode tag and the compute provenance. SevenNet specs drive the batched GPU relaxer; an ASE
    Calculator instance drives the serial ASE relaxer -- the engine stays agnostic to which
    potential it is.

    `rigid_units=True` forces the SERIAL ASE path even for a SevenNet spec, because that is the only
    one that can hold a discrete unit's bonds rigid through a fixed-cell pre-relaxation (torch-sim
    has no holonomic constraint). Much slower -- no batching -- so it is opt-in, for builds that
    actually contain an oxo-anion / molecular ion / octahedral unit.
    """
    if _calc.is_sevennet_spec(calculator):
        model, modal = _calc.parse_sevennet_spec(calculator)

        if rigid_units:
            calc, prov = backend.ase_calculator(model=model, modal=modal, d3=d3)

            def _relax(confs):
                return _calc.ase_relax_batch(confs, calc, constrain='auto', steps=max_steps)
            return _relax, 'sevennet-ase-rigid', prov

        def _relax(confs):
            backend.use_model(model=model, modal=modal, d3=d3)
            return backend.batched_fire_relax(confs, steps=max_steps)
        return _relax, 'sevennet', None                 # provenance filled by the backend on build

    calc = calculator                                   # an ase.calculators.Calculator
    constrain = 'auto' if rigid_units else None
    return (lambda confs: _calc.ase_relax_batch(confs, calc, constrain=constrain,
                                                steps=max_steps)), 'ase-generic', None


def _final_unavailable(reason, n_atoms):
    """The final tier was REQUESTED and could not run. -> the marker that says so.

    Three outcomes have to stay distinguishable, and two of them used to collapse into one:

        final_MAR is None            the caller never asked for a final tier (no --final)
        final_MAR['available'] False asked for, could not run -- `reason` says why
        final_MAR['E_final_...']     it ran

    Returning None on failure made "could not" indistinguishable from "did not ask", which is the
    defect D10 is about, and an empty final_MAR has already been read as the latter. Same vocabulary
    the engine uses for `disorder_descriptor` ({'available': False, 'reason': ...}) rather than a
    null that a reader has to guess at.
    """
    return {'available': False, 'requested': True, 'reason': reason,
            'n_atoms': n_atoms, 'modal': 'mpa'}


def _final_omni(built, d3=False):
    """Recompute the winning decoration at the omni-mpa modal (the DeMARS final
    tier). SevenNet-native only; returns (final_MAR_dict, relaxed_atoms)."""
    confs, rel, val, samples, info = built
    # the SAME choice `distribution.lowest` and the written `representative.*` make -- a bracket probe
    # is not a candidate, so the final tier relaxes the structure that actually ships.
    ship = engine.ship_candidates(samples)
    lo = min(ship, key=lambda x: x['E_per_atom'])
    widx = samples.index(lo)
    # Sampling is over: give the nano tier's GPU memory back before omni asks for its own. Without
    # this both models sit on the card at once, on top of the allocator pool the sampling run grew,
    # and the final tier OOMs on a structure that fits perfectly well by itself (measured: 21.2 of
    # 23.6 GiB already held when omni asked for 3.7). Provenance survives -- see release_models.
    backend.release_models()
    backend.use_model(model='7net-omni', modal='mpa', d3=d3)
    n_at = len(rel[widx])
    try:
        Ef, relf, vf = backend.batched_fire_relax([rel[widx]])
    except Exception as exc:
        # The ensemble is already relaxed and on disk. A final tier that will not fit on this
        # device is a fact about the device, not a reason to throw the run away -- one entry lost a
        # completed 102-frame ensemble to exactly this. Report it and let the nano tier ship.
        # The exception text goes into the record verbatim so the cause stays diagnosable.
        return _final_unavailable(f'{type(exc).__name__}: {exc}', n_at), None
    if not vf[0]:
        return _final_unavailable('the omni-mpa relaxation did not converge', n_at), None
    return ({'E_final_per_atom': round(float(Ef[0]) / len(relf[0]), 4),
             'n_atoms': len(relf[0]), 'modal': 'mpa' + ('+D3' if d3 else '')},
            relf[0])


def _compute_provenance(calculator, mode, d3, prov=None):
    """Which weights actually produced these energies -- stamped into the record.

    A record without this cannot be checked against the run that claims to have produced it: the
    same command on two machines can resolve to different SevenNet checkpoints, and the numbers
    move with them. For the native path the backend reports what it resolved (and the hash when the
    models config pins one); for an injected ASE calculator we can only name the object, which is
    still better than recording nothing."""
    if prov is not None:                      # the ASE-rigid path already resolved and recorded it
        return prov
    if mode == 'sevennet':
        return backend.active_provenance()
    from . import models as _models
    return _models.provenance(spec=repr(calculator), tag='ase-generic', d3=bool(d3))


def deaverage(source, calculator='sevennet', *, n_samples=30, min_cell=15.0,
              excl=1.1, d3=False, max_rounds=3, final=True, out_dir=None,
              same_excl=None, couple_cut=None, rigid_units=False, checkpoint=True,
              max_steps=None, max_atoms=None, couple_formers=None, couple_anions=None):
    """De-average a disordered crystal into a Minimal Atomistic Representation.

    Parameters
    ----------
    source : CIF path | structure-file path | CIF text | pymatgen Structure | ase.Atoms
        The disordered (partial-occupancy / split-site) crystal.
    calculator : str | ase.calculators.Calculator
        'sevennet'/'7net-nano'/'7net-omni'/<checkpoint.pth> for the native batched
        path, or ANY ASE Calculator instance (MACE, CHGNet, ORB, ...) for the
        generic serial path.
    n_samples : int      decorations to sample+relax per round (engine NR).
    min_cell : float     minimum supercell edge in Angstrom (>=15 for production).
    d3 : bool            dispersion -- OFF by default (DeMARS uses no D3).
    final : bool         SevenNet only: recompute the winner at the omni-mpa modal.
    rigid_units : bool   hold discrete former-ligand units (SO4, BO3, MoO6, NH4, ...) rigid through
                         a fixed-cell pre-relaxation, then release. Use for structures whose
                         evidence shows a rigid unit: relaxing such a build freely lets a
                         neighbouring cation tear a ligand off its centre, which the
                         composition-only gates cannot see. Forces the SERIAL ASE path (torch-sim
                         has no bond constraint), so it is much slower -- opt-in per structure.
    out_dir : str|None   if given, write ensemble.xyz + representative.cif + deaverage_output.json.
    checkpoint : bool|str  keep finished configs on disk during sampling (needs out_dir), so a
                         timeout / OOM / eviction costs one config instead of the whole round, and
                         a re-run resumes. A str is an explicit path. Removed once ensemble.xyz
                         is written.

    Returns
    -------
    MARRecord
    """
    cif_text = to_cif_text(source)
    src_desc = source if isinstance(source, str) and len(source) < 200 else f'<{type(source).__name__}>'

    struct = engine.structure_from_text(cif_text)
    ev = evidence.evidence_from_text(cif_text)          # ICSD-free: no sibling search
    relax, mode, prov = _make_relax(calculator, d3, rigid_units=rigid_units, max_steps=max_steps)

    ckpt = None
    if checkpoint and out_dir:
        ckpt = checkpoint if isinstance(checkpoint, str) else os.path.join(out_dir, '_ckpt.xyz')
        # loss unit: 1 on the serial paths (free -- they relax one at a time anyway), a third of a
        # standard round on the batched one, where handing over fewer configs at once has a cost
        relax = _ckpt.checkpointed(relax, ckpt, tag=_ckpt.relaxer_tag(mode, calculator, d3),
                                   chunk=1 if mode != 'sevennet' else 10)

    e_ev, built, info, trace = engine.run_self_driving(
        structure=struct, evidence=ev, relax=relax,
        NR=n_samples, MIN_A=min_cell, EXCL=excl, d3=d3,
        max_rounds=max_rounds, same_excl=same_excl, couple_cut=couple_cut, max_atoms=max_atoms,
        couple_formers=couple_formers, couple_anions=couple_anions)

    rec = MARRecord.from_engine(e_ev, built, info, trace,
                                source=src_desc, calculator=str(calculator))
    rec.mlip = _compute_provenance(calculator, mode, d3, prov)
    final_struct = None
    # the omni-mpa final tier is a FREE relax of an already-clean representative, so it runs on the
    # batched backend even when sampling went through the constrained ASE path
    if final and built is not None and mode in FINAL_TIER_MODES:
        fm, final_struct = _final_omni(built, d3=d3)
        rec.final_MAR = fm

    if out_dir and built is not None:
        rec.write_ensemble(out_dir, final_struct=final_struct)
        if ckpt:
            _ckpt.discard(ckpt)         # ensemble.xyz is the answer now; don't leave a rival copy
    return rec
