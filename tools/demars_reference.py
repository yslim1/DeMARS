#!/usr/bin/env python3
"""Self-check: run a bundled COD structure and compare against a stored reference record.

Third parties could always RUN the pipeline; nothing told them whether the answer was right. This
gives the package a ground truth it owns -- the fixtures are COD (CC0), so they redistribute
freely, unlike anything ICSD.

    python tools/demars_reference.py --list
    python tools/demars_reference.py                      # check every reference
    python tools/demars_reference.py cod_9003141          # check one
    python tools/demars_reference.py cod_9003141 --generate   # (maintainers) re-stamp it

WHAT IS COMPARED, AND WHY THE SPLIT MATTERS
  exact      enumeration facts -- supercell, template size, formula, how many decorations exist,
             the composition and charge of the adopted one. These come from the seeded enumerator
             and are independent of the potential and the hardware, so any difference is a real
             behaviour change.
  tolerant   energies. An MLIP's numbers move with weights, torch build, and device; comparing them
             bitwise would make the check fail for the wrong reason and get switched off. The band
             is wide on purpose: it catches "wrong checkpoint / wrong modal", not last-bit noise.
  reported   n_relaxed and the sampling shortfall are printed, never failed on: a smaller machine
             legitimately relaxes fewer configs.

Running with a DIFFERENT potential than the reference was stamped with is supported and normal --
the exact block still applies (it does not depend on the potential), and the energy block is
skipped with a note rather than silently passed.
"""
import argparse
import json
import os
import sys

import _icsd_env  # noqa: F401  -- must precede any demars_core import

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REFDIR = os.path.join(ROOT, 'reference')
FIXDIR = os.path.join(ROOT, 'demars-core', 'tests', 'fixtures')

# Deliberately small: a self-check a newcomer will actually wait for (minutes on CPU), not a
# production run. Production settings are --nr 30 --min-nm 1.5 --final; see AGENTS.md.
SETTINGS = dict(n_samples=6, min_cell=8.0, max_rounds=1, final=False)

E_TOL_MEV = 20.0        # per atom. Wide: it separates "wrong weights" from hardware noise.


def _facts(rec, name):
    """-> the comparable facts, flattened out of a MARRecord dict."""
    d = rec.get('distribution') or {}
    lo = d.get('representative') or {}
    return {
        'entry': name,
        'settings': SETTINGS,
        'exact': {
            'status': rec.get('status'),
            'formula': rec.get('formula'),
            'supercell': rec.get('supercell'),
            'n_atoms_template': rec.get('n_atoms_template'),
            'n_total': d.get('n_total'),
            'representative_composition': lo.get('composition'),
            'representative_charge': lo.get('charge'),
            'representative_n_atoms': lo.get('n_atoms'),
        },
        'tolerant': {
            'ground_E_per_atom': d.get('ground_E_per_atom'),
            'E_tol_meV_per_atom': E_TOL_MEV,
        },
        'reported': {
            'n_relaxed': d.get('n_relaxed'),
            'spread_meV': d.get('spread_meV'),
        },
        'stamped_with': (rec.get('mlip') or {}).get('model_tag'),
        'stamped_sha256': (rec.get('mlip') or {}).get('checkpoint_sha256'),
    }


def _run(cif, calculator):
    from demars_core.api import deaverage
    rec = deaverage(cif, calculator=calculator, **SETTINGS)
    return rec.to_dict() if hasattr(rec, 'to_dict') else rec


def _compare(ref, got):
    """-> (ok, lines). Prints every field, so a failure says which one and by how much."""
    ok, out = True, []
    for k, want in ref['exact'].items():
        have = got['exact'].get(k)
        same = want == have
        ok &= same
        out.append(f'  {"ok " if same else "FAIL"}  {k}: {have!r}'
                   + ('' if same else f'   expected {want!r}'))

    same_pot = (ref.get('stamped_sha256') and
                ref['stamped_sha256'] == got.get('stamped_sha256'))
    we, ge = ref['tolerant']['ground_E_per_atom'], got['tolerant']['ground_E_per_atom']
    if not same_pot:
        out.append(f'  --    ground_E_per_atom: {ge!r} (not compared: stamped with '
                   f'{ref.get("stamped_with")!r}, ran with {got.get("stamped_with")!r})')
    elif we is None or ge is None:
        out.append(f'  --    ground_E_per_atom: {ge!r} (nothing to compare)')
    else:
        d_meV = abs(ge - we) * 1000.0
        same = d_meV <= ref['tolerant']['E_tol_meV_per_atom']
        ok &= same
        out.append(f'  {"ok " if same else "FAIL"}  ground_E_per_atom: {ge:.4f} '
                   f'(reference {we:.4f}, off by {d_meV:.1f} meV/atom)')

    for k, v in got['reported'].items():
        out.append(f'  --    {k}: {v!r}   (reference {ref["reported"].get(k)!r}, not a criterion)')
    return ok, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('entries', nargs='*', help='reference name(s); default = all')
    ap.add_argument('--calculator', default='7net-nano')
    ap.add_argument('--generate', action='store_true', help='(re)write the reference file')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    os.makedirs(REFDIR, exist_ok=True)
    names = args.entries or sorted(
        f[:-5] for f in os.listdir(REFDIR) if f.endswith('.json'))
    if args.list:
        for n in names:
            print(n)
        return 0
    if not names:
        print('no references stored yet -- run with --generate', file=sys.stderr)
        return 2

    failed = []
    for name in names:
        cif = os.path.join(FIXDIR, f'{name}.cif')
        if not os.path.isfile(cif):
            print(f'{name}: no such fixture ({cif})', file=sys.stderr)
            failed.append(name)
            continue
        print(f'== {name}  ({os.path.basename(cif)}, {SETTINGS})')
        got = _facts(_run(cif, args.calculator), name)
        path = os.path.join(REFDIR, f'{name}.json')
        if args.generate:
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(got, fh, indent=1, sort_keys=True)
            print(f'   wrote {path}')
            continue
        with open(path, encoding='utf-8') as fh:
            ref = json.load(fh)
        ok, lines = _compare(ref, got)
        print('\n'.join(lines))
        print(f'   -> {"PASS" if ok else "FAIL"}')
        if not ok:
            failed.append(name)

    if failed:
        print(f'\n{len(failed)} reference(s) did not match: {", ".join(failed)}', file=sys.stderr)
        return 1
    print('\nall references matched' if not args.generate else '\ndone')
    return 0


if __name__ == '__main__':
    sys.exit(main())
