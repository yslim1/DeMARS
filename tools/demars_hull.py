#!/usr/bin/env python3
"""Stage ④b -- the CONVEX HULL stability read, from a FILE (no ICSD).

Thin wrapper over `demars_core.hull.compute_hull`. The question: is this MAR thermodynamically
reachable, and what would it decompose into? Every other check asks whether the cell is a faithful
realisation of the CIF; this is the only one that asks whether the phase is stable at all -- and
where a composition has no ordered ICSD sibling, it is the only stability reference there is.

Usage:
  python tools/demars_hull.py <rundir>/_work/<tag>/representative_final.vasp \
         [--calculator omni] [--out <rundir>/_work/<tag>]
  -> stdout = JSON; --out writes <out>/hull.json (stage ⑥ takes it with --hull) AND the
     tier-relaxed MAR as <out>/representative_final.{vasp,cif}, so one hull run also
     upgrades the deliverable to the final tier.

**Point --out at the run's `_work/<tag>/`, not the run dir.** That is where `representative*`
belong (see tools/README.md): in the run dir these land beside the engine's own copies and a reader
cannot tell which is the MAR. The record resolves that through `engine.json`, never by globbing, so
a stray copy misleads a person rather than the code -- but it still misleads.

**Run it on the structure that SHIPPED**, at **the tier that structure was finalized with**
(`--calculator omni` is the DeMARS final tier and the default). An E_above_hull computed at a
different level than the structure it describes is not a stability number.

**This needs a Materials Project key**: `$MP_API_KEY` or `materials_project.api_key` in
`demars.yaml`. Without one it returns `state: "not_run"` with the reason and exits 1 -- which is
UNCHECKED, never "on the hull". Exit 0 only for `state: "derived"`.

This artifact IS `gates.hull`, the fifth gate, and the ONLY gate that cannot re-run itself in
stage 6 -- so pass it with `--hull` every time. Skipping this stage leaves the gate `not_run`, and
an unchecked gate is not a pass.
"""
import argparse
import json
import os
import sys

import _icsd_env                                    # noqa: F401  MUST precede demars_core

from demars_core.hull import compute_hull           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('structure', help='the MAR that shipped (normally representative_final.vasp)')
    ap.add_argument('--calculator', default='omni',
                    help='the tier the representative was FINALIZED with (default omni = omni-mpa)')
    ap.add_argument('--mode', default='self-consistent', choices=('self-consistent', 'mp-direct'),
                    help='where the reference ENERGIES come from. Default self-consistent: MP gives '
                         'the competing phases\' structures and every one is re-relaxed at your '
                         'tier, so the MLIP-vs-DFT offset cancels instead of landing on the MAR '
                         'alone. mp-direct takes MP\'s own energies and relaxes only the MAR -- '
                         'much cheaper, omni-mpa only, and it falls back when MP has a gap')
    ap.add_argument('--d3', action='store_true', help='dispersion; DeMARS uses none')
    ap.add_argument('--corrections', default='mp2020', choices=('mp2020', 'none'),
                    help="energy scale (default mp2020 = MP's own hull scale, the one its published "
                         "energy_above_hull is on). 'none' = raw PBE(+U) from uncorrected_energy: "
                         'internally consistent, but not the hull MP publishes. Applied to '
                         'references AND target alike, and stamped in the artifact')
    ap.add_argument('--max-steps', type=int, default=None, help='relaxation step ceiling')
    ap.add_argument('--label', default=None, help='what to call this MAR in the artifact')
    ap.add_argument('--out', default=None, help='write <out>/hull.json (stage ⑥ reads it via --hull)')
    args = ap.parse_args()

    real = sys.stdout
    sys.stdout = sys.stderr                # keep MLIP / MP chatter off the JSON on stdout
    try:
        res = compute_hull(args.structure, calculator=args.calculator, mode=args.mode, d3=args.d3,
                           corrections=args.corrections, max_steps=args.max_steps,
                           label=args.label, out_dir=args.out)
    finally:
        sys.stdout = real

    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    if res['state'] != 'derived':
        print(f'WARN: hull state is {res["state"]!r} — {res["reason"]}\n'
              '      That is UNCHECKED, not "on the hull". Do NOT report a stability claim from '
              'this artifact, and do not quote a hull number you did not compute.', file=sys.stderr)
        return 1
    m = res['mar']
    print(f'hull: {res["chemsys"]}  E_above_hull = {m["E_above_hull_eV_per_atom"]} eV/atom  '
          f'(tier {res["tier"]}, mode {res["mode"]}, corrections {res["corrections"]}, '
          f'{res["n_refs_used"]} refs)', file=sys.stderr)
    if res.get('mode_note'):
        print(f'      mode: {res["mode_note"]}', file=sys.stderr)
    if res.get('truncated_refs'):
        print(f'      NOTE: {len(res["truncated_refs"])} candidate phase(s) were dropped by a '
              f'reference cap: {res["truncated_refs"][:6]}', file=sys.stderr)
    if res.get('skipped_large_refs'):
        print(f'      NOTE: skipped {len(res["skipped_large_refs"])} oversized reference cell(s): '
              f'{res["skipped_large_refs"][:4]}', file=sys.stderr)
    if m['E_above_hull_eV_per_atom'] < -1e-3:
        print('      NOTE: BELOW the reference hull — either a genuinely new stable ordering or a '
              'reference set missing a competing phase. Say which you think it is.', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
