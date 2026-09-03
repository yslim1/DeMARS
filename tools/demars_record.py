#!/usr/bin/env python3
"""Stage (6) -- the mar-1.0 RECORD assembler, from a FILE (no ICSD).

Thin wrapper over demars_core._engine.mar_record.build_record. YOU write
judgment.json (your stage-(5) verdict); this merges it with the engine output +
the CIF evidence into the canonical mar-1.0 record and RECOMPUTES the charge &
fidelity gates itself -- it does not trust your transcription. "Code proves."

Usage:
  python tools/demars_record.py <structure.cif> --engine <engine.json> \
         --judgment <judgment.json> --hull <hull.json> [--review <review.json>] \
         [--connectivity <connectivity.json>] --out <dir>

**`gates.connectivity` and `gates.sqs` run themselves.** Stage ⑥ audits the shipped frame and
checks whether the enumeration's clustered draw is what shipped -- charge and fidelity were always
recomputed here, and leaving the two STRUCTURAL checks to a manual call had them absent from 15 of
15 frozen-run records. `--connectivity` is therefore OPTIONAL, and worth passing only when the
stage-⑤ artifact carries what the auto-run cannot: `--expect` (an enforced coordination) or a
multi-frame audit of `ensemble.xyz`. A supplied artifact wins over the auto-run.
If the shipped frame cannot be read -- a moved or archived tree, which is 9 of 15 frozen-run
records -- the gate degrades to `not_run` (UNCHECKED, never clean) and this tool warns.

`--review` is the mar-reviewer output -- one round object, or a JSON ARRAY of every
analyst->reviewer round oldest-first. Re-run this stage with it once the review loop
has settled: it is cheap (no relaxation) and it is the only way the verdict becomes
part of the record instead of dying with the conversation. WITHOUT it the record says
`"review": null` = UNREVIEWED, which is not the same as reviewed-and-clean.

stdout = the record. Writes <dir>/record.json. Warns on stderr for missing
judgment fields, failing gates and an unreviewed/still-blocking record -- read those
warnings.
"""
import argparse
import json
import os
import sys

import _icsd_env                                    # noqa: F401  MUST precede demars_core

from demars_core._engine.mar_record import build_record
from demars_evidence import evidence_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('structure', help='the SAME structure file the engine ran on')
    ap.add_argument('--engine', required=True, help='engine.json from tools/demars_engine.py')
    ap.add_argument('--judgment', default=None, help='YOUR judgment.json (stage 5)')
    ap.add_argument('--hull', default=None,
                    help='hull.json from tools/demars_hull.py. `gates.hull` is the one gate '
                         'that cannot run itself (MP call + relaxation), so WITHOUT this it is '
                         '"not_run" = UNCHECKED, which is not a pass')
    ap.add_argument('--review', default=None,
                    help='mar-reviewer output: one round object, or a JSON array of all rounds')
    ap.add_argument('--connectivity', default=None,
                    help='stage-⑤ audit JSON (tools/demars_connectivity.py --json). OPTIONAL -- '
                         'the gate audits the shipped frame itself; supply this only for --expect '
                         'or a multi-frame ensemble audit, which wins over the auto-run')
    ap.add_argument('--hull-tol', type=float, default=None,
                    help='stability threshold in eV/atom for gates.hull. UNSET (default) = the gate '
                         'reports E_above_hull and leaves `pass` null, which is what the research '
                         'pipeline did: there is no inherited threshold, and the terms that set one '
                         '(configurational entropy, the mode\'s energy-scale error) are '
                         'system-dependent. Falls back to hull.tol_eV_per_atom in demars.yaml')
    ap.add_argument('--out', default=None, help='write <out>/record.json')
    ap.add_argument('--icsd-id', type=int, default=None,
                    help='if this CIF IS an ICSD entry: stamp it and self-exclude from the sibling search')
    args = ap.parse_args()

    engine = json.load(open(args.engine))
    judg = json.load(open(args.judgment)) if args.judgment else {}
    hull = json.load(open(args.hull)) if args.hull else None
    rev = json.load(open(args.review)) if args.review else None
    conn = json.load(open(args.connectivity)) if args.connectivity else None
    ev = evidence_for(args.structure, iid=args.icsd_id)

    tol = args.hull_tol
    if tol is None:
        from demars_core import models as _models
        tol = ((_models.load_config().get('hull') or {}) or {}).get('tol_eV_per_atom')
    rec = build_record(args.icsd_id, engine, judg, hull=hull, evidence=ev, review=rev,
                       connectivity=conn, hull_tol=tol, out_dir=args.out)
    # the file IS the key here, not an id. Stored relative to the record when the CIF sits beside it
    # (so the entry directory can be moved), absolute when it comes from a shared corpus elsewhere.
    from demars_core._engine.mar_record import portable_path
    rec['source'] = (portable_path(args.structure, args.out)[0] if args.out
                     else os.path.abspath(args.structure))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(f'{args.out}/record.json', 'w', encoding='utf-8') as fh:
            json.dump(rec, fh, indent=1, ensure_ascii=False, default=str)

    miss = [k for k in ('class', 'interpretation', 'disorder_pattern', 'confidence')
            if not judg.get(k)]
    if miss:
        print(f'WARN: judgment missing {miss}', file=sys.stderr)
    if rec.get('escalate') is True:
        print('WARN: judgment.escalate is a bare `true` — the record and the gallery now flag an '
              'ESCALATION with nothing in it. That boolean is the ENGINE\'s dummy-species flag, not '
              'this field: write a sentence saying what needs a human (or null), and re-run this '
              'stage.', file=sys.stderr)
    for g, v in (rec.get('gates') or {}).items():
        if isinstance(v, dict) and v.get('pass') is False:
            print(f'WARN: gate {g} FAILS: {v}', file=sys.stderr)
    _rp = rec.get('representative') or {}
    if _rp.get('source') == 'engine-repick' and _rp.get('E_final_eV_per_atom') is None:
        print(f"WARN: the shipped frame is a REPICK with no final tier — "
              f"{_rp.get('E_final_unavailable_reason')}\n"
              '      Every other path gives the shipped structure a final-tier energy, and '
              '`gates.hull` is only a stability number at the tier the structure was finalized '
              'with. Relax this frame with `demars_engine.py --from <frame> --final` into its own '
              '_work tag and name that run in judgment.representative_pick.final_from.',
              file=sys.stderr)
    if _rp.get('pick_unresolved'):
        print(f"WARN: representative_pick could NOT be resolved — {_rp['pick_unresolved']}. The "
              'record therefore ships the enumeration LOWEST, which is the frame you asked to '
              'replace. Fix the config_label and re-run this stage.', file=sys.stderr)
    _cg = (rec.get('gates') or {}).get('connectivity') or {}
    if _cg.get('state') == 'not_run':
        print(f'WARN: gates.connectivity is "not_run" — {_cg.get("basis")}. That is UNCHECKED, '
              'not clean. Fix the representative path, or run tools/demars_connectivity.py --json '
              'on the shipped frame and pass it with --connectivity.', file=sys.stderr)
    elif _cg.get('state') in ('vacuous', 'ambiguous'):
        print(f'WARN: gates.connectivity is {_cg["state"]!r} (pass=None, not True): '
              f'{_cg.get("basis")}', file=sys.stderr)
    _hg = (rec.get('gates') or {}).get('hull') or {}
    if _hg.get('state') == 'not_run':
        print(f'WARN: gates.hull is "not_run" — {_hg.get("basis")}. This is the one gate that does '
              'NOT run itself: run tools/demars_hull.py on the shipped representative and pass its '
              'hull.json with --hull. UNCHECKED is not clean.', file=sys.stderr)
    elif _hg.get('state') == 'derived' and _hg.get('pass') is None:
        print(f'NOTE: gates.hull reports E_above_hull = '
              f'{_hg.get("E_above_hull_eV_per_atom")} eV/atom and does NOT judge it — no threshold '
              'is configured (--hull-tol, or hull.tol_eV_per_atom in demars.yaml). Read the number; '
              'do not report this gate as passed.', file=sys.stderr)
    elif _hg.get('state') in ('vacuous', 'ambiguous'):
        print(f'WARN: gates.hull is {_hg["state"]!r} (pass=None, not True): {_hg.get("basis")}',
              file=sys.stderr)
    _rv = rec.get('review')
    if _rv is None:
        print('WARN: no --review — this record is UNREVIEWED. That is not the same as '
              'reviewed-and-clean; do not report it as confirmed.', file=sys.stderr)
    elif _rv.get('unresolved_blocking'):
        print(f"WARN: after {_rv['n_rounds']} review round(s) the verdict is "
              f"{_rv.get('final_verdict')!r} with UNRESOLVED blocking objection(s): "
              f"{_rv['unresolved_blocking']}", file=sys.stderr)

    print(json.dumps(rec, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
