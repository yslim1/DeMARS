"""Convex-hull stability for a MAR -- `E_above_hull` and formation energy.

The question this answers is the one no other check can: the charge, fidelity, connectivity and SQS
gates all ask "is this cell a faithful, self-consistent realisation of the CIF", and an ordered
sibling gives a DeltaE against ONE competitor. None of them says whether the MAR is
thermodynamically reachable at all, or what it would decompose into. Where a phase has no ordered
sibling, this is the only stability reference there is.

TWO MODES, `mode=`. The hull vertices and the target MUST live on one energy scale -- mixing an MLIP
target into a DFT hull is meaningless -- so `mp-direct` is legal only at the one tier that shares
MP's scale, while `self-consistent` is legal anywhere and is the DEFAULT:

  `self-consistent` (the default):
    MP is used only to CHOOSE the competing phases and give their structures; every reference is
    re-relaxed with the SAME calculator as the MAR. Both sides of the subtraction then carry the same
    MLIP error and it largely cancels, instead of sitting on the target alone: measured, three MARs
    that all land ON the hull here can read tens of meV/atom above it in `mp-direct`. Costs
    ~20-90 reference relaxations instead of one -- about 2x the stage, a few % of the pipeline.
    Also the only option when the tier is not omni-mpa (a MACE or CHGNet energy is not on MP's
    scale) or when MP's GGA thermo has a gap, which it does for some f-elements.

  `mp-direct` (cheap; only when the tier IS omni-mpa without dispersion):
    MP's GGA/GGA+U entries are taken as they are, on whichever scale `corrections` asks for. The
    tier gate is about the RAW scale underneath: omni-mpa is trained on MP's own trajectories, so its
    energies sit on MP's raw PBE(+U) scale and can share a hull with MP's; a MACE or CHGNet energy
    cannot, whatever correction is layered on top. The ONLY computation is the MAR's own relaxation.
    Requires MP to cover the chemsys; falls back to self-consistent (stamped in `mode_note`) when
    it cannot.

**Which ENERGY SCALE.** `corrections` picks it, and it is stamped in the artifact because the same
MAR can pass on one scale and fail on the other:

  `none`    raw PBE(+U). The scale the MLIP actually predicts: `mpa` is a training-corpus label
            (MPtrj + Alexandria, next to `omat24` / `matpes_pbe` / `mp_r2scan` in the model's own
            modal map), NOT a correction scheme, so the network's output is uncorrected DFT.
  `mp2020`  MP's experiment-calibrated scale (MaterialsProject2020Compatibility): per-anion and
            GGA/GGA+U corrections, applied to the REFERENCES and the TARGET alike. The DEFAULT, and
            the scale MP's own hull is on -- measured across Fe-Ti-O, 75 of 79 materials' published
            `energy_above_hull` reproduce here and none on the raw scale. MP's docs describe these
            corrections as the mechanism that lets calculations at different levels of theory share
            one phase diagram, which is exactly what a hull requires.

Measured 2026-08-24, not assumed. The corrections are large (-1.2 to -1.9 eV/atom for iron oxides)
but they are per-species offsets, and a hull decomposition conserves composition -- so they CANCEL
almost exactly -- measured, 0.0 meV/atom on a sulfide and a cobaltate, and at or under a couple of
meV/atom for most members of a ferrite chemsystem. They stop
cancelling when the decomposition crosses a ROLE boundary, because MP applies a correction "only
when [the material] contains a corrected element as an anion" and uses GGA+U for "oxide and fluoride
compounds containing any of the transition metals V, Cr, Mn, Fe, Co, Ni, W, and Mo". A ternary oxide
of one of those metals is corrected; the same metal in its ELEMENTAL form is not -- so a decomposition
crossing into the metal does not cancel, and the raw scale can then put a METALLIC phase among the
products where the corrected one gives the ternary oxide. That is the misplacement those U terms were
fitted to remove. Measured on MP's own entries, one target moved 149 meV/atom between the scales.
Mixed-valence oxides are the DeMARS corpus, so this switch is not cosmetic.

Applying `mp2020` to OUR energy needs the VASP settings MP would have used -- a bare entry is
silently DROPPED by the compatibility scheme. They are derived from `MPRelaxSet`, MP's own input
set, not invented: it reproduces MP's per-atom correction exactly (FeO -1.4715 eV/atom, equal to
what MP's own FeO entries carry). An entry the scheme still refuses aborts the hull rather than
letting one entry sit on a different scale from the rest.

**Honest states.** This is the one DeMARS check that depends on an external service, so it reports
`state` the way the gates do -- `derived` / `ambiguous` / `not_run` -- and never a bare number
without one. No API key, no `mp-api` installed, a fetch that failed, a chemsys MP cannot cover: all
`not_run`, which is UNCHECKED, not "on the hull". `E_above_hull` is present only when
`state == 'derived'`.

This feeds `gates.hull` -- the fifth gate (`_engine.mar_record.hull_gate`), which REPORTS the number
and leaves `pass` null unless a threshold is configured; see HULL_TOL_EV_PER_ATOM for why there is no
default line. It is the one gate that cannot re-run itself inside stage 6, because it
needs a network call and a relaxation, so an absent artifact leaves that gate `not_run`, which is
UNCHECKED and never a pass. The `mp-api` client is a hard dependency for exactly that reason -- a
gate whose machinery may or may not be installed makes "all gates clean" mean different things on
different machines. The KEY is deployment config, and its absence is a reported state, not a crash.

Usage:  python -m demars_core.hull <representative_final.vasp> [--calculator omni] [--out <dir>]
"""
import json
import os
import sys

__all__ = ['compute_hull', 'mp_entries', 'mp_reference_structures', 'mp_available']

CORRECTION_SCHEMES = ('none', 'mp2020')

MODES = ('self-consistent', 'mp-direct')

# Guard against a pathological reference cell, not a cost cap: across the chemsystems measured here
# it never fires (no candidate exceeded 300 atoms). Anything it does skip is
# named in `meta['skipped']`, because a dropped candidate can be a hull vertex.
MAX_REF_ATOMS = 300

# PHYSICS, not cost: only phases on MP's hull can be its vertices, but the references are re-relaxed
# here, so a phase MP puts slightly above its hull can come out BELOW on our scale and become one.
# 0.12 eV/atom is the margin for that reordering -- comfortably above the largest MLIP-vs-MP
# discrepancy measured on a target (67 meV/atom).
REF_EHULL_TOL = 0.12

# No cap on the number of reference compositions. A cap of 90 as a cost guard measures out to be the
# wrong trade: a five-element chemsystem can carry more than 90 candidates
# within REF_EHULL_TOL, so the cap silently dropped possible hull vertices -- while the atoms it saved
# were ~15%, a few seconds. A cap that changes the answer to save seconds is
# not a guard. Set an int to reinstate one; whatever it drops is then named in `meta['truncated']`.
MAX_REFS = None


def _result(state, reason, **extra):
    """Every return goes through here, so no path can hand back a number with no state, and no path
    can hand back a bare `null` -- which in the record's schema means an optional step never ran."""
    out = {'state': state, 'reason': reason}
    out.update(extra)
    return out


def mp_available():
    """-> (bool, reason). Whether the Materials Project can be reached at all: the client package
    installed AND a key resolvable. Both are optional in this package, so both are reportable
    conditions rather than crashes."""
    try:
        import mp_api.client                            # noqa: F401
    except Exception:
        return False, ('mp-api is not importable -- it is a hard dependency of demars-core '
                       '(the hull is a gate): pip install -e demars-core')
    from . import models as _models
    if not _models.mp_api_key():
        return False, ('no Materials Project API key -- set $MP_API_KEY or '
                       "materials_project.api_key in demars.yaml (get one at "
                       'https://next-gen.materialsproject.org/api)')
    return True, None


def _mp_entry_parameters(structure):
    """The VASP settings MP itself would have used for this structure, from `MPRelaxSet` -- MP's own
    input set, so the U assignment is theirs, not ours (Fe-O -> GGA+U with U=5.3 on Fe; Fe-S -> plain
    GGA, because MP applies no U to sulfides). MaterialsProject2020Compatibility reads `run_type`,
    `is_hubbard`, `hubbards` and `potcar_spec` and DROPS an entry that lacks them, so an MLIP energy
    cannot be corrected without them.

    `potcar_spec` carries MP's POTCAR symbols in the 'PAW_PBE <symbol> <date>' shape the scheme
    parses; the date is a placeholder because no POTCAR was read -- what the check needs is the
    functional and the symbol.
    """
    from pymatgen.io.vasp.sets import MPRelaxSet
    vs = MPRelaxSet(structure)
    incar = vs.incar
    hub = ({sym: float(u) for sym, u in zip(vs.poscar.site_symbols, incar.get('LDAUU') or [])}
           if incar.get('LDAU') else {})
    return {'run_type': 'GGA+U' if hub else 'GGA', 'is_hubbard': bool(hub), 'hubbards': hub,
            'potcar_spec': [{'titel': f'PAW_PBE {sym} 01Jan1000', 'hash': None}
                            for sym in vs.potcar_symbols]}


def mp2020_correct(structure, energy, entry_id=None):
    """One (structure, total energy) -> a MaterialsProject2020Compatibility-corrected ComputedEntry,
    or None if the scheme refuses it. The caller MUST treat None as a refusal to hull: an entry that
    silently drops out leaves the rest of the set on a different scale, which is the one thing a
    hull cannot survive."""
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from pymatgen.entries.computed_entries import ComputedStructureEntry
    ent = ComputedStructureEntry(structure, float(energy), entry_id=entry_id,
                                 parameters=_mp_entry_parameters(structure))
    try:
        return MaterialsProject2020Compatibility().process_entry(ent, clean=True)
    except Exception:
        return None


def _tier_is_mp_scale(calculator, d3):
    """Is this tier on MP's own raw PBE(+U) energy scale? Only omni-mpa without dispersion is --
    that equivalence is a validated property of those weights, not a general MLIP property, so
    every other calculator has to re-relax the references instead of borrowing MP's numbers."""
    from . import calculators as _calc
    if d3 or not _calc.is_sevennet_spec(calculator):
        return False
    model, modal = _calc.parse_sevennet_spec(calculator)
    return str(model).lower() == '7net-omni' and modal == 'mpa'


# ---- the Materials Project side ---------------------------------------------

def mp_entries(elements, api_key=None, corrections='none'):
    """MP GGA/GGA+U ComputedEntries for the chemsys, on the requested scale. The `mp-direct`
    reference set: energies taken straight from MP, nothing recomputed.

    `corrections='none'` strips the MP2020 post-hoc corrections back off (`uncorrected_energy`) --
    raw PBE(+U), the scale the MLIP predicts. `'mp2020'` keeps MP's corrected energies, which is what
    materialsproject.org publishes. Returns (entries, meta); `meta['covered']` names the elements MP
    actually had, so the caller can refuse rather than build a hull with a missing vertex."""
    from mp_api.client import MPRester
    from pymatgen.entries.computed_entries import ComputedEntry
    from . import models as _models
    with MPRester(api_key or _models.mp_api_key()) as mpr:
        ents = mpr.get_entries_in_chemsys(sorted(elements),
                                          additional_criteria={'thermo_types': ['GGA_GGA+U']})
    out, covered = [], set()
    for e in ents:
        E = e.energy if corrections == 'mp2020' else e.uncorrected_energy
        out.append(ComputedEntry(e.composition, E, entry_id=str(e.entry_id)))
        covered |= {el.symbol for el in e.composition.elements}
    # `n_candidates` and `covered`, the SAME key names mp_reference_structures uses -- the result
    # assembly reads one meta shape, and a fetcher that spells a key differently silently nulls the
    # field (this one said `n_entries`, and every mp-direct hull reported n_candidates: null).
    return out, {'source': f'MP-GGA/GGA+U (corrections={corrections})',
                 'n_candidates': len(out), 'covered': sorted(covered)}


def mp_reference_structures(elements, api_key=None, tol=REF_EHULL_TOL, max_refs=MAX_REFS):
    """Candidate hull vertices as STRUCTURES, from MP's summary endpoint (which carries structures
    and, unlike the thermo endpoint, covers f-elements). MP picks the competing phases and supplies
    their geometry; the ENERGIES are recomputed by the caller, so no MP-DFT energy enters the hull.
    Two-step to keep the download small: cheap metadata over every sub-chemsystem -> choose the
    near-stable formulas plus one elemental endpoint per element -> fetch structures for just those.
    Returns ([(reduced_formula, ase.Atoms, mp_id, mp_ehull)], meta)."""
    from itertools import combinations

    from mp_api.client import MPRester
    from pymatgen.core import Composition
    from pymatgen.io.ase import AseAtomsAdaptor

    from . import models as _models
    subs = ['-'.join(sorted(c)) for r in range(1, len(elements) + 1)
            for c in combinations(sorted(elements), r)]
    with MPRester(api_key or _models.mp_api_key()) as mpr:
        meta_docs = mpr.materials.summary.search(
            chemsys=subs, fields=['material_id', 'formula_pretty', 'energy_above_hull'])
        best = {}                                        # reduced formula -> (ehull, material_id)
        for dd in meta_docs:
            rf = dd.formula_pretty
            eh = dd.energy_above_hull if dd.energy_above_hull is not None else 9.0
            if rf not in best or eh < best[rf][0]:
                best[rf] = (eh, str(dd.material_id))
        items = sorted(best.items(), key=lambda kv: kv[1][0])
        keep = {rf: mid for rf, (eh, mid) in items if eh <= tol}
        for el in elements:                              # guarantee an elemental endpoint per element
            for rf, (eh, mid) in items:
                c = Composition(rf)
                if len(c.elements) == 1 and c.elements[0].symbol == el:
                    keep.setdefault(rf, mid)
                    break
        truncated = []
        if max_refs is not None and len(keep) > max_refs:
            ordered = [rf for rf, _ in items if rf in keep]          # near-stable first
            truncated = ordered[max_refs:]
            keep = {rf: keep[rf] for rf in ordered[:max_refs]}
        keep_ids = list(keep.values())
        sdocs = mpr.materials.summary.search(
            material_ids=keep_ids,
            fields=['material_id', 'formula_pretty', 'structure', 'energy_above_hull'])
    out, skipped = [], []
    for dd in sdocs:
        st = dd.structure
        if st is None:
            continue
        if len(st) > MAX_REF_ATOMS:
            skipped.append({'formula': dd.formula_pretty, 'n_atoms': len(st)})
            continue
        eh = dd.energy_above_hull if dd.energy_above_hull is not None else 9.0
        out.append((dd.formula_pretty, AseAtomsAdaptor.get_atoms(st), str(dd.material_id),
                    round(float(eh), 4)))
    covered = sorted(set().union(*[set(a.get_chemical_symbols()) for _, a, _, _ in out])) if out else []
    return out, {'source': 'materials-project (structures; energies recomputed here)',
                 'skipped': skipped, 'truncated': truncated, 'n_candidates': len(out),
                 'n_chemsys_materials': len(best), 'covered': covered}


# ---- the hull ---------------------------------------------------------------

def _pd_and_target(entries, target):
    """PhaseDiagram over `entries`, then read `target` off it. Split out so both modes share one
    definition of what E_above_hull / formation energy / decomposition mean. `target` arrives as a
    finished entry -- already on the same scale as `entries`, which is the caller's job."""
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    pd = PhaseDiagram(entries)
    tgt = target
    decomp, ehull = pd.get_decomp_and_e_above_hull(tgt, allow_negative=True)
    return {'E_above_hull_eV_per_atom': round(float(ehull), 4),
            'formation_E_eV_per_atom': round(float(pd.get_form_energy_per_atom(tgt)), 4),
            'on_hull': bool(abs(ehull) < 1e-3),
            # float(), not just round(): pymatgen hands the amounts back as numpy scalars on some
            # paths and plain floats on others, and a numpy scalar reaches json.dump's `default=str`
            # -- which writes the amount as a STRING. A field whose type depends on which reference
            # set was used is not a schema.
            'decomposition': {e.composition.reduced_formula: round(float(amt), 3)
                              for e, amt in decomp.items()}}


def compute_hull(structure, calculator='omni', *, mode='self-consistent', d3=False, api_key=None,
                 references=None, corrections='mp2020', max_steps=None, label=None, out_dir=None):
    """E_above_hull for one MAR. `structure` is anything ASE can read (a path) or an `ase.Atoms` --
    normally the run's `representative_final`.

    `calculator` is the tier, and it must be **the tier the representative was finalized with**: an
    E_above_hull computed at a different level than the structure it describes is not a stability
    number, it is two numbers subtracted by accident. Default 'omni' = the DeMARS final tier.

    `references` overrides the Materials Project entirely -- a list of `(formula, ase.Atoms)` to
    re-relax as the competing phases. That is how the hull math is exercised offline; it is also the
    escape hatch for a chemsys MP does not cover.

    `mode` picks where the reference ENERGIES come from. Default `'self-consistent'`: MP supplies the
    competing phases' structures and every one is re-relaxed with the same calculator as the MAR, so
    the MLIP-vs-DFT offset cancels out of the subtraction instead of landing on the target alone --
    measured, that offset is most of what `mp-direct` reports on MARs that sit on the hull once the
    references are recomputed. `'mp-direct'` takes MP's own
    energies as the vertices and relaxes only the MAR: far cheaper (one cell instead of ~20-90, which
    on the corpus is roughly 2x end to end), legal only at omni-mpa, and it falls back to
    self-consistent when MP cannot cover the chemsys. Injected `references` force self-consistent.

    `corrections` is the ENERGY SCALE, applied to references and target alike, never one side only,
    and stamped in the result. Default `'mp2020'`: MP's hull IS the corrected hull -- across Fe-Ti-O,
    75 of 79 materials' published `energy_above_hull` reproduce on the corrected scale and none on
    the raw one (raw is off by up to 0.16 eV/atom) -- so a hull built from MP references belongs on
    that scale. `'none'` = raw PBE(+U) from `uncorrected_energy`, which is internally consistent but
    produces a hull that is NOT the one MP publishes.

    `out_dir` gets `hull.json` AND the tier-relaxed MAR as `representative_final.{vasp,cif}`, so
    one hull run also upgrades the deliverable structure to the final tier. **Point `out_dir` at the run's `_work/<tag>/`, not the run dir**, or
    those files land beside the engine's own and `tools/README.md`'s rival-`representative_final`
    hazard applies (the record resolves the MAR through `engine.json`, never by globbing, so a stray
    copy misleads a reader rather than the code).
    """
    from ase.io import read as ase_read
    from pymatgen.core import Composition

    from .api import _compute_provenance, _make_relax

    if corrections not in CORRECTION_SCHEMES:
        raise ValueError(f'corrections must be one of {CORRECTION_SCHEMES}, got {corrections!r}')
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}, got {mode!r}')
    atoms = ase_read(structure) if isinstance(structure, str) else structure

    els = sorted(set(atoms.get_chemical_symbols()))
    chemsys = '-'.join(els)
    # An injected ASE calculator's repr carries a memory address, which would put a value in the
    # record that is different on every run and means nothing to a reader. Name the class instead.
    tier_name = calculator if isinstance(calculator, str) else type(calculator).__name__
    label = label or (structure if isinstance(structure, str) else atoms.get_chemical_formula())
    base = {'chemsys': chemsys, 'requested_tier': tier_name, 'label': label,
            'corrections': corrections}

    def _write(res):
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'hull.json'), 'w', encoding='utf-8') as fh:
                json.dump(res, fh, indent=1, ensure_ascii=False, default=str)
        return res

    # The hull describes the ORDERED deliverable. ASE cannot hold partial occupancies, so handing it
    # the input CIF instead of the shipped frame does not fail -- it silently drops the minority
    # species and hulls a composition nobody built. Refuse; the composition is the one thing a hull
    # cannot be wrong about.
    occ = (getattr(atoms, 'info', None) or {}).get('occupancy') or {}
    disordered = {site: sp for site, sp in occ.items()
                  if len(sp) > 1 or any(float(o) < 1 - 1e-6 for o in sp.values())}
    if disordered:
        return _write(_result(
            'not_run',
            f'this structure is DISORDERED (partial/mixed occupancies on {len(disordered)} '
            f'site(s): {sorted(disordered)[:4]}) -- ASE drops the minority species, so the hull '
            'would describe a composition that was never built. Run this on the shipped '
            'representative_final, not on the input CIF.', **base))

    # ---- assemble the references. Injected ones win; otherwise MP has to be reachable at all.
    requested_mode, refs, ref_meta, mp_ents, mode_note = mode, None, None, None, None
    if references is not None:
        refs = [(rf, at, None, None) for rf, at in references]
        mode = 'self-consistent'
        mode_note = ('caller supplied the references, so mp-direct does not apply'
                     if requested_mode == 'mp-direct' else None)
        ref_meta = {'source': 'caller-supplied references (energies recomputed here)',
                    'n_candidates': len(refs), 'skipped': [], 'truncated': [], 'covered': sorted(
                        set().union(*[set(a.get_chemical_symbols()) for _, a, _, _ in refs]))
                    if refs else []}
    else:
        ok, why = mp_available()
        if not ok:
            return _write(_result('not_run', why, **base))
        if requested_mode == 'mp-direct':
            # Asking for mp-direct at a tier that is not on MP's scale is a caller error, not a data
            # gap: quietly relaxing into self-consistent would answer a question nobody asked.
            if not _tier_is_mp_scale(calculator, d3):
                return _write(_result(
                    'not_run', f"mode 'mp-direct' needs the omni-mpa tier without dispersion -- "
                    f'{tier_name!r} is not on MP\'s energy scale, so MP\'s energies cannot be its '
                    "vertices. Use mode 'self-consistent'.", **base))
            try:
                mp_ents, ref_meta = mp_entries(els, api_key, corrections=corrections)
            except Exception as exc:
                return _write(_result('not_run', f'MP entry fetch failed: {type(exc).__name__}: {exc}',
                                      **base))
            missing = sorted(set(els) - set(ref_meta['covered']))
            if mp_ents and not missing:
                mode = 'mp-direct'
            else:
                # MP's GGA thermo has a gap here (the f-element case). NOT an error -- fall through
                # to self-consistent, which uses the structure endpoint instead. `mode` in the
                # result says which one actually ran; `mode_note` says why it changed.
                mp_ents, mode = None, 'self-consistent'
                mode_note = (f'mp-direct requested, but MP GGA/GGA+U has no entry for '
                             f'{missing or els} -- fell back to self-consistent')
        if mp_ents is None:
            try:
                refs, ref_meta = mp_reference_structures(els, api_key)
            except Exception as exc:
                return _write(_result('not_run',
                                      f'MP reference fetch failed: {type(exc).__name__}: {exc}',
                                      **base))
            mode = 'self-consistent'

    if mode == 'self-consistent':
        missing = sorted(set(els) - set(ref_meta.get('covered') or []))
        if not refs or missing:
            return _write(_result(
                'ambiguous',
                f'no hull reference for element(s) {missing or els}; an incomplete simplex cannot '
                'be read as a hull' if refs else 'no reference phases found for this chemsys',
                references_found=[rf for rf, _a, _i, _e in (refs or [])], **base))

    # ---- the computation. One batch, one tier: the MAR first, then every reference that has to be
    # relaxed, so nothing can drift onto a different setting between them.
    relax, mode_tag, prov = _make_relax(calculator, d3, max_steps=max_steps)
    batch = [atoms] + [a for (_rf, a, _i, _e) in (refs or [])]
    try:
        E, rel, val = relax(batch)
    except Exception as exc:
        return _write(_result('not_run', f'relaxation failed: {type(exc).__name__}: {exc}',
                              mode=mode, **base))
    if not val[0]:
        return _write(_result('not_run', 'the MAR did not converge at this tier', mode=mode, **base))

    mlip = _compute_provenance(calculator, mode_tag, d3, prov)
    tier = f'{mode_tag}:{tier_name}' + ('+D3' if d3 else '')
    mar_at = rel[0]

    # A hull run doubles as the final-tier upgrade of the deliverable, so the relaxed MAR is
    # written here -- and written BEFORE the phase diagram: the relaxation succeeded, and that
    # structure is worth keeping even if the hull itself cannot be read.
    if out_dir:
        from ase.io import write as _ase_write
        os.makedirs(out_dir, exist_ok=True)
        _ase_write(os.path.join(out_dir, 'representative_final.vasp'), mar_at, format='vasp',
                   sort=True)
        _ase_write(os.path.join(out_dir, 'representative_final.cif'), mar_at, format='cif')

    # Every entry that is not already on the requested scale is put on it HERE, target included.
    # `_entry` returns (entry, None) or (None, reason) because MaterialsProject2020Compatibility
    # returns None rather than raising: swallowing that leaves one entry on the raw scale inside a
    # corrected hull, and the resulting number looks perfectly reasonable.
    from pymatgen.io.ase import AseAtomsAdaptor

    def _entry(at, energy, entry_id=None):
        """-> (entry, None) or (None, reason). Correction applied when the scheme asks for it."""
        if corrections == 'none':
            from pymatgen.entries.computed_entries import ComputedEntry
            return ComputedEntry(Composition(at.get_chemical_formula()), float(energy),
                                 entry_id=entry_id), None
        ent = mp2020_correct(AseAtomsAdaptor.get_structure(at), energy, entry_id=entry_id)
        if ent is None:
            return None, (f'MaterialsProject2020Compatibility refused '
                          f'{at.get_chemical_formula()}{f" ({entry_id})" if entry_id else ""}')
        return ent, None

    if mode == 'mp-direct':
        # MP's own entries came back on the requested scale already (mp_entries picked the field).
        entries, ref_rows = mp_ents, ref_meta['source'] + ' (taken directly, not re-relaxed)'
        n_used = len(mp_ents)
    else:
        entries, ref_rows = [], []
        for k, (rf, _a, mp_id, ref_ehull) in enumerate(refs, start=1):
            if not val[k]:
                ref_rows.append({'formula': rf, 'mp_id': mp_id, 'relaxed': False})
                continue
            at = rel[k]
            ent, why = _entry(at, E[k], entry_id=mp_id)
            if ent is None:
                return _write(_result('ambiguous', why + ' -- refusing to hull a mixed scale',
                                      mode=mode, tier=tier, mlip=mlip, **base))
            entries.append(ent)
            ref_rows.append({'formula': rf, 'mp_id': mp_id, 'relaxed': True,
                             'n_atoms': len(at), 'E_per_atom': round(float(E[k]) / len(at), 4),
                             'correction_eV': round(float(getattr(ent, 'correction', 0.0)), 4),
                             'mp_ehull_prior': ref_ehull})
        n_used = len(entries)
        if n_used < len(els):
            return _write(_result('ambiguous',
                                  f'only {n_used} reference(s) relaxed for a {len(els)}-element '
                                  'hull -- too few for a valid simplex',
                                  mode=mode, tier=tier, references=ref_rows, mlip=mlip, **base))

    target, why = _entry(mar_at, E[0])
    if target is None:
        return _write(_result('ambiguous', why + ' -- the MAR cannot be put on the reference scale',
                              mode=mode, tier=tier, mlip=mlip, **base))
    try:
        read = _pd_and_target(entries, target)
    except Exception as exc:
        return _write(_result('ambiguous', f'phase diagram failed: {type(exc).__name__}: {exc}',
                              mode=mode, tier=tier, mlip=mlip, **base))

    return _write(_result(
        'derived', None,
        tier=tier, mode=mode, self_consistent=(mode == 'self-consistent'),
        ref_source=ref_meta.get('source'), n_candidates=ref_meta.get('n_candidates'),
        n_refs_used=n_used, references=ref_rows, mode_note=mode_note,
        skipped_large_refs=ref_meta.get('skipped') or [],
        truncated_refs=ref_meta.get('truncated') or [], mlip=mlip,
        mar={'label': label,
             'composition': mar_at.get_chemical_formula(), 'n_atoms': len(mar_at),
             'E_per_atom': round(float(E[0]) / len(mar_at), 4),
             'correction_eV': round(float(getattr(target, 'correction', 0.0)), 4),
             # `below_mp_hull` only means something against MP's own hull; in self-consistent mode
             # the hull is ours, so the comparison does not exist rather than being False.
             'below_mp_hull': (bool(read['E_above_hull_eV_per_atom'] < -1e-3)
                               if mode == 'mp-direct' else None),
             **read},
        **base))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='python -m demars_core.hull')
    ap.add_argument('structure', help='the MAR to place on the hull (normally representative_final)')
    ap.add_argument('--calculator', default='omni',
                    help='the tier the representative was FINALIZED with (default omni = omni-mpa)')
    ap.add_argument('--d3', action='store_true', help='dispersion (DeMARS uses none by default)')
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--out', default=None, help='write <out>/hull.json')
    args = ap.parse_args(argv)

    real = sys.stdout
    sys.stdout = sys.stderr                      # keep MLIP / MP chatter off the JSON on stdout
    try:
        res = compute_hull(args.structure, calculator=args.calculator, d3=args.d3,
                           max_steps=args.max_steps, out_dir=args.out)
    finally:
        sys.stdout = real
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    if res['state'] != 'derived':
        print(f'WARN: hull state is {res["state"]!r} — {res["reason"]}', file=sys.stderr)
    return 0 if res['state'] == 'derived' else 1


if __name__ == '__main__':
    sys.exit(main())
