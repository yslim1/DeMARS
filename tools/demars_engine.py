#!/usr/bin/env python3
"""Stage (3) -- THE CORE ENGINE, from a FILE (no ICSD).

Thin wrapper over demars_core.deaverage(). Writes the deliverables AND an
`engine.json` in the exact shape tools/demars_record.py consumes -- the research
pipeline's mar_engine.py contract:

    {distribution:{n_relaxed, ground_E, spread_meV, std_meV, energies_sorted, lowest},
     engine:{supercell, group_orbits, sampling, exclusion_A, n_groups_merged, ...},
     ensemble_files:{ensemble, n_frames, representative, representative_final},
     final_MAR:{E_final_per_atom, n_atoms, modal}}

Two modes, mirroring the research tool:
  <cif>                  enumerate + relax the disordered CIF  (the normal path)
  <cif> --from <struct>  relax ONE provided structure through the same nano ->
                         omni-mpa tiers. Use for a custom/repaired build, OR to put
                         an ordered sibling on the same tier for the dE comparison
                         (`icsd-query extract <id>` fetches it). Never adopted as the MAR.

Usage:
  python tools/demars_engine.py <structure.cif> --out <rundir> [--nr 30] [--min-nm 1.5]
         [--excl 1.1] [--same-excl X] [--couple-cut X] [--rounds 3] [--final] [--d3]
         [--couple-formers Ga] [--couple-anions As]
         [--calculator sevennet] [--summary]
"""
import argparse
import contextlib
import json
import os
import sys

import _icsd_env                                    # noqa: F401  MUST precede demars_core

from demars_core import deaverage
from demars_core.io import to_cif_text


@contextlib.contextmanager
def _quiet_stdout():
    """Keep stdout pure JSON. The SevenNet backend prints progress chatter
    ('Converting model backend...') on stdout, which corrupts the contract for any
    caller that pipes us. Route library stdout to stderr for the compute."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _engine_json(rec, max_steps=None):
    """MARRecord -> the mar_engine.py-shaped engine.json that build_record() reads.

    Two deliberate fixups: the distribution key is `lowest` (MARRecord calls it
    `representative`), and the engine block must be the FULL einfo -- MARRecord
    filters it down to a display subset and drops `group_orbits`, which the
    fidelity gate needs.
    """
    d = rec.to_dict()
    dist = dict(d.get('distribution') or {})
    dist['lowest'] = dist.pop('representative', None)
    dist['ground_E'] = dist.get('ground_E_per_atom')
    dist['highest_E'] = dist.get('highest_E_per_atom')

    einfo = dict(d.get('engine') or {})
    if rec._built is not None:                # the unfiltered engine info
        einfo.update(rec._built[4])
    # The step ceiling is applied inside the relax callable, which build() never sees, so it cannot
    # reach `sampling` from the engine side -- record it here, where the flag lives. Without this the
    # record is not reproducible from its own machine-readable fields: a reader following
    # generation_recipe would re-run the default budget, which is the `--same-excl` gap over again.
    if isinstance(einfo.get('sampling'), dict):
        einfo['sampling']['max_steps'] = {
            'value': max_steps, 'source': 'override --max-steps'} if max_steps else {
            'value': None, 'source': 'auto: clip(20 * n_atoms, 400, 1000)'}
    return {'source': d.get('source'), 'status': d.get('status'),
            'calculator': d.get('calculator'), 'formula': d.get('formula'),
            'oxidation_states': d.get('oxidation_states'),
            'distribution': dist, 'engine': einfo,
            'self_driving': d.get('self_driving'),
            'ensemble_files': d.get('ensemble_files') or {},
            'mlip': d.get('mlip'),                    # compute provenance -> record generation_recipe
            'final_MAR': d.get('final_MAR') or {}}


def _stamp_sibling(ej, sib_ev, icsd_id=None):
    """Move the sibling RESULT and its PROVENANCE into engine.json together (defect D11).

    `deaverage()` derives its evidence with no iid, so the inner run skips the lookup and stamps
    `not_run`. We re-run it out here and overwrite the value -- and used to leave the inner
    provenance behind, so one file said the search both happened (a searched-and-absent string)
    and never ran (`states.ordered_sibling == 'not_run'`, still listed in `unconfident`). Those are
    exactly the two states AGENTS.md forbids collapsing, and a reviewer who read the honest field
    spent a whole round refuting a run that had in fact been given `--icsd-id`.

    `icsd_id` is stamped for the same reason: without it, a run WITH self-exclusion and a run
    WITHOUT are byte-identical in their provenance, so the artifact cannot answer the question.
    """
    if icsd_id is not None:
        ej['engine']['icsd_id'] = icsd_id
    if sib_ev is None:
        return ej
    ej['engine']['ordered_sibling'] = sib_ev.get('ordered_sibling')
    ej['engine']['ordered_sibling_ids'] = sib_ev.get('ordered_sibling_ids', [])
    prov = sib_ev.get('ordered_sibling_provenance') or {}
    summ = ej['engine'].get('provenance_summary')
    if not prov.get('state') or not isinstance(summ, dict):
        return ej
    summ.setdefault('states', {})['ordered_sibling'] = prov['state']
    unc = [n for n in (summ.get('unconfident') or []) if n != 'ordered_sibling']
    if not prov.get('confident'):
        unc.append('ordered_sibling')
    summ['unconfident'] = unc
    summ['n_unconfident'] = len(unc)
    return ej


def _rename_deterministic_output(out_dir, rec):
    """deaverage()'s own record.json -> deaverage_output.json, and fix the path it left behind.

    Stage 6 writes its mar-1.0 `record.json` into the same tree, so the two must not share a name --
    one would silently overwrite the other. The new name deliberately carries neither "record" (that
    word belongs to stage 6) nor "engine" (`engine.json` is a DIFFERENT file in this same dir): it
    names what produced it instead.

    `write_ensemble()` recorded the PRE-rename name, so `ensemble_files['record']` pointed at a file
    that no longer existed -- dangling in every run through this CLI. Fixed here, where the rename
    happens, so every consumer of engine.json sees the real path.

    -> the new path, or None if there was nothing to rename.
    """
    det = os.path.join(out_dir, 'record.json')
    if not os.path.exists(det):
        return None
    dea = os.path.join(out_dir, 'deaverage_output.json')
    os.replace(det, dea)
    if (rec.ensemble_files or {}).get('record') == det:
        rec.ensemble_files['record'] = dea
    return dea


def _elements(spec):
    """'Ga,As' -> {'Ga', 'As'}; None/'' -> None (keep the module table).

    D37: `--couple-cut` retargets a cutoff and cannot ADD a role, so a chemistry whose cross-orbit
    contacts match neither COORD_FORMERS nor COORD_ANIONS was unreachable by any value (a Ga
    chalcogenide, a Co intermetallic -- both named in the engine's own `unmatched` provenance).
    """
    if not spec:
        return None
    els = {e.strip() for e in spec.split(',') if e.strip()}
    if not els:
        return None
    bad = sorted(e for e in els if not (e[:1].isupper() and e.isalpha() and len(e) <= 2))
    if bad:
        raise SystemExit(f'--couple-formers/--couple-anions take element symbols, got {bad}')
    return els


def _relax_one(path, calculator, d3, out_dir, rigid_units=False, max_steps=None):
    """--from mode: single structure through nano -> omni-mpa, no enumeration.

    This is how a custom/repaired build is realised, so it is exactly where `rigid_units` matters:
    a hand-placed polyanion relaxed freely can lose a ligand to a neighbouring cation.
    """
    from ase.io import write as asewrite
    from pymatgen.io.ase import AseAtomsAdaptor
    from demars_core.api import _make_relax, _final_omni, FINAL_TIER_MODES
    from demars_core._engine import mar_engine as engine

    struct = engine.structure_from_text(to_cif_text(path))
    # hand an ase.Atoms to the relaxer: the batched backend takes either, but the serial ASE path
    # sets `atoms.calc` and a pymatgen Structure has no such attribute
    atoms_in = AseAtomsAdaptor.get_atoms(struct)
    relax, mode, _prov = _make_relax(calculator, d3, rigid_units=rigid_units, max_steps=max_steps)
    E, rel, val = relax([atoms_in])               # the engine's relax callable
    if not val or not val[0]:
        raise SystemExit(f'relaxation failed for {path}')
    atoms = rel[0]
    e_nano = round(float(E[0]) / len(atoms), 5)

    out = {'mode': 'from', 'input': os.path.abspath(path),
           'single_MAR': {'E_nano_per_atom': e_nano, 'n_atoms': len(atoms)}}
    os.makedirs(out_dir, exist_ok=True)
    if mode in FINAL_TIER_MODES:
        # no enumeration here, so hand-assemble the 5-tuple `_final_omni` unpacks:
        # (confs, relaxed, valid, samples, info) -- see api.py's INTERNAL CONTRACT note
        built = ([atoms_in], rel, val, [{'E_per_atom': e_nano, 'valid': True}], {})
        fm, final_struct = _final_omni(built, d3=d3)
        # `_final_omni` NEVER returns None. Both of its failure exits return the MARKER dict
        # `api._final_unavailable()` builds -- {'available': False, 'requested': True, 'reason': ...}
        # -- paired with `final_struct = None`. So `if fm is None` never fired, and the else-branch
        # read `fm['E_final_per_atom']`, a key the marker does not carry: KeyError, mid-run, on
        # a real run. D23 gave `deaverage()` the "asked for / could not run" distinction and missed
        # the CLI's copy of the same tier -- this is that half.
        # `final_struct` is the unambiguous discriminator: it exists only when the tier actually ran.
        if final_struct is None:
            # Asked for and could not run. Keep the marker verbatim -- the reason is the whole point,
            # and dropping it back to prose is what made "could not" look like "did not ask" (D10).
            out['single_MAR']['final_MAR'] = fm
        else:
            out['single_MAR'].update(E_final_per_atom=fm['E_final_per_atom'], modal=fm['modal'])
            asewrite(f'{out_dir}/representative_final.vasp', final_struct, format='vasp', sort=True)
            asewrite(f'{out_dir}/representative_final.cif', final_struct, format='cif')
            out['files'] = {'representative_final': f'{out_dir}/representative_final.vasp'}
    asewrite(f'{out_dir}/representative.xyz', atoms, format='extxyz')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('structure', help='CIF / POSCAR / xyz path (disorder preserved from CIF)')
    ap.add_argument('--out', required=True, help='run dir for the deliverables + engine.json')
    ap.add_argument('--from', dest='from_struct', default=None,
                    help='relax THIS structure instead of enumerating (custom/repaired build)')
    ap.add_argument('--nr', type=int, default=30, help='decorations to sample+relax per round')
    ap.add_argument('--min-nm', type=float, default=1.5, help='min supercell edge in NANOMETRES')
    ap.add_argument('--excl', type=float, default=1.1, help='cross-element exclusion-merge cutoff (A)')
    ap.add_argument('--same-excl', type=float, default=None, help='same-element merge cutoff (A)')
    ap.add_argument('--couple-cut', type=float, default=None, help='former-anion co-placement cutoff (A)')
    # These REPLACE COORD_FORMERS/COORD_ANIONS for THIS structure's coupling path. Read
    # `provenance_summary` first: an `unmatched` couple_cut names the elements to consider.
    ap.add_argument('--couple-formers', default=None, metavar='EL[,EL...]',
                    help='elements playing the FORMER role in the coupling path (e.g. Ga)')
    ap.add_argument('--couple-anions', default=None, metavar='EL[,EL...]',
                    help='elements playing the ANION role in the coupling path (e.g. As)')
    ap.add_argument('--max-steps', type=int, default=None,
                    help='optimizer-step ceiling per config. Default: auto (20 x n_atoms, clipped '
                         'to [400, 1000]). Raise it when configs come back "did not converge in N '
                         'steps"; the value is recorded in engine.json sampling.max_steps.')
    ap.add_argument('--max-atoms', type=int, default=None,
                    help='supercell atom budget. Default: 800 (MAXAT*1.6). The cell rule loses to '
                         'this budget silently in effect -- a large primitive cell (256 atoms and '
                         'up) cannot reach 1.5 nm on every axis within 800, and the run ships a '
                         'short axis. Raise it when the cell rule matters more than the cost; the '
                         'value is recorded in engine.json supercell_provenance.')
    ap.add_argument('--rounds', type=int, default=3, help='self-driving diagnose->re-enumerate passes')
    ap.add_argument('--final', action='store_true', help='recompute the winner at the omni-mpa modal')
    ap.add_argument('--d3', action='store_true', help='dispersion (OFF by default -- DeMARS uses no D3)')
    ap.add_argument('--calculator', default='sevennet')
    ap.add_argument('--summary', action='store_true', help='human digest to stderr')
    ap.add_argument('--icsd-id', type=int, default=None,
                    help='if this CIF IS an ICSD entry: self-exclude it from the sibling search')
    ap.add_argument('--no-siblings', action='store_true', help='skip the ordered-sibling lookup')
    ap.add_argument('--rigid-units', action='store_true',
                    help='hold discrete former-ligand units (SO4/BO3/MoO6/NH4/...) rigid through a '
                         'fixed-cell pre-relax, then release. Use when the evidence shows a rigid '
                         'unit: a free relax lets a neighbouring cation tear a ligand off its '
                         'centre, and charge/fidelity are composition-only and cannot see it. '
                         'Forces the SERIAL ASE path (torch-sim has no bond constraint) -- much '
                         'slower, so opt in per structure.')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.from_struct:
        with _quiet_stdout():
            out = _relax_one(args.from_struct, args.calculator, args.d3, args.out,
                             rigid_units=args.rigid_units, max_steps=args.max_steps)
        with open(f'{args.out}/engine.json', 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1, default=str)
        print(json.dumps(out, indent=1, default=str))
        return 0

    with _quiet_stdout():
        rec = deaverage(args.structure, calculator=args.calculator,
                        n_samples=args.nr, min_cell=args.min_nm * 10.0, excl=args.excl,
                        d3=args.d3, max_rounds=args.rounds, final=args.final,
                        out_dir=args.out, same_excl=args.same_excl, couple_cut=args.couple_cut,
                        couple_formers=_elements(args.couple_formers),
                        couple_anions=_elements(args.couple_anions),
                        rigid_units=args.rigid_units, max_steps=args.max_steps,
                        max_atoms=args.max_atoms)

    # the skills + mar_record expect a .vasp final; write_ensemble only emits .cif
    fin_cif = (rec.ensemble_files or {}).get('representative_final')
    if fin_cif and os.path.exists(fin_cif):
        from ase.io import read as aseread, write as asewrite
        vasp = fin_cif.replace('.cif', '.vasp')
        asewrite(vasp, aseread(fin_cif), format='vasp', sort=True)
        rec.ensemble_files['representative_final'] = vasp

    _rename_deterministic_output(args.out, rec)

    ej = _engine_json(rec, max_steps=args.max_steps)

    # deaverage() derives its own evidence with no iid, so mar_evidence skips the sibling
    # search and stamps 'not searched'. build_record() reads ordered_sibling from HERE (not
    # from the evidence), so fill it in ourselves or the record loses the sibling entirely.
    _stamp_sibling(ej, None, args.icsd_id)          # record the flag even with --no-siblings
    if not args.no_siblings:
        try:
            from demars_evidence import evidence_for
            _stamp_sibling(ej, evidence_for(args.structure, iid=args.icsd_id), args.icsd_id)
        except Exception as e:                      # never fail the run over the lookup
            print(f'WARN: ordered-sibling lookup failed: {e}', file=sys.stderr)

    with open(f'{args.out}/engine.json', 'w', encoding='utf-8') as fh:
        json.dump(ej, fh, indent=1, default=str)
    if args.summary:
        print(rec.summary(), file=sys.stderr)
    print(json.dumps(ej, indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
