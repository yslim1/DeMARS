#!/usr/bin/env python3
"""Stage 5 -- the polyanion / rigid-unit CONNECTIVITY gate, as a CLI.

The gate itself is `demars_core.connectivity.audit_frame`; this file only reads frames,
parses `--expect`, prints, and sets the exit code. It used to hold the audit logic, which
put the one gate that catches what charge and fidelity cannot OUTSIDE the installable
package -- so a standalone install did not have it and the package's own test had to
sys.path into tools/ to reach it.

Usage:
  python demars_connectivity.py <struct.{vasp,cif,xyz}> [--json] [--tol 1.25] [--expect S=4,P=4]
  python demars_connectivity.py <ensemble.xyz>            # audits ALL frames
Exit code 0 = clean (all centres consistent), 1 = defect found.
"""
import sys, json, argparse

from ase.io import read

import _icsd_env                                    # noqa: F401  MUST precede demars_core

from demars_core.connectivity import audit_frame    # noqa: E402


def _parse_expect(spec):
    """'S=4,P=4' -> {'S': 4, 'P': 4}"""
    if not spec:
        return None
    out = {}
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        el, _, cn = part.partition('=')
        if not cn:
            raise SystemExit(f'--expect needs EL=CN pairs, got {part!r}')
        out[el.strip()] = int(cn)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('struct')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--tol', type=float, default=1.25,
                    help='bond cutoff as a multiple of the covalent-radius sum (default 1.25)')
    ap.add_argument('--expect', default=None,
                    help='enforce a coordination instead of inferring it, e.g. "S=4,P=4"')
    args = ap.parse_args()
    expect = _parse_expect(args.expect)

    frames = read(args.struct, index=':')
    if not isinstance(frames, list):
        frames = [frames]

    all_defects = []
    per_frame = []
    for fi, at in enumerate(frames):
        summ, defs = audit_frame(at, tol=args.tol, expect=expect)
        per_frame.append({'frame': fi, 'summary': summ, 'defects': defs})
        for d in defs:
            d2 = dict(d); d2['frame'] = fi; all_defects.append(d2)

    clean = len(all_defects) == 0
    out = {'file': args.struct, 'n_frames': len(frames), 'clean': clean,
           'tol': args.tol, 'expect': expect,
           'n_defects': len(all_defects), 'defects': all_defects,
           'per_frame_summary': per_frame[0]['summary'] if len(frames) == 1 else None}

    if args.json:
        print(json.dumps(out, indent=1))
    else:
        tag = 'CLEAN' if clean else f'DEFECT x{len(all_defects)}'
        print(f'[connectivity] {args.struct}  ({len(frames)} frame(s))  -> {tag}')
        # '_coverage' is a report of what the gate did NOT examine, not a centre -- printing it as
        # one killed this branch with KeyError('ligands') after the verdict was already correct.
        centres = {k: v for k, v in per_frame[0]['summary'].items() if not k.startswith('_')}
        if not centres:
            print('    (no former-ligand unit detected -- nothing for this gate to check)')
        for centre, s in centres.items():
            print(f'    {centre}-{"/".join(s["ligands"])}: expected {s["expected"]}'
                  f' ({s["expected_from"]})  hist {s["coord_hist"]}  ({s["n_ok"]}/{s["n_centres"]} ok)')
        cov = per_frame[0]['summary'].get('_coverage') or {}
        if cov.get('ligand_role'):
            # excluded from the CENTRE population because they are a ligand of another unit
            # (96 cyanide N on one entry read as stripped centres). Never drop it silently.
            print('    ligand role (not centres): '
                  + ', '.join(f'{el} x{n}' for el, n in sorted(cov['ligand_role'].items())))
        if cov.get('not_examined'):
            print(f'    NOT examined: {", ".join(cov["not_examined"])}'
                  '  (absent from COORD_FORMERS -- "clean" does not cover these)')
        for d in all_defects[:20]:
            print(f'    !! frame {d["frame"]} {d["centre"]} atom {d["atom_index"]}: '
                  f'{d["coordination"]}-coord (expected {d["expected"]})')
        if len(all_defects) > 20:
            print(f'    ... +{len(all_defects) - 20} more')
    sys.exit(0 if clean else 1)


if __name__ == '__main__':
    main()
