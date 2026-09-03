#!/usr/bin/env python3
"""Stage (1a) -- the CIF evidence bundle, from a FILE (no ICSD).

Thin wrapper over demars_core._engine.mar_evidence.evidence_from_text. `icsd_id`
is null (the key here is a file path) unless you pass --icsd-id.

The ordered-sibling search IS wired up: tools/ordered_lookup.py backs
mar_evidence's `_sib` bootstrap against the local ICSD DB, located via ICSD_DB_DIR
(set by `_icsd_env` from `paths.icsd_db` in demars.yaml). Disable with
--no-siblings, or automatically if the DB is absent -- then `ordered_sibling`
reads "none (no sibling DB configured)", which means UNCHECKED.

Usage:
  python tools/demars_evidence.py <structure.cif> [--summary] [--out <rundir>]
                                  [--icsd-id N] [--no-siblings]

stdout = JSON (the bundle). --summary also prints a human digest to stderr.
--out <rundir> additionally writes <rundir>/evidence.json.
"""
import argparse
import json
import os
import sys

import _icsd_env                                         # noqa: F401  MUST precede demars_core

from demars_core._engine import mar_evidence as MEV      # noqa: E402
from demars_core.io import to_cif_text                   # noqa: E402

SIBLING_DB = MEV._sib is not None


def evidence_for(path, iid=None, siblings=True):
    """The (1a) bundle for a structure file.

    `iid` only stamps the record and self-excludes in the sibling search -- pass it
    when the CIF you are analysing IS an ICSD entry, or it may list itself as its
    own ordered sibling.
    """
    search = bool(siblings and SIBLING_DB)
    ev = MEV.evidence_from_text(to_cif_text(path), iid=iid, search_siblings=search)
    ev['source'] = os.path.abspath(path)
    if not search:
        ev['ordered_sibling'] = ('none (sibling search disabled)' if siblings is False
                                 else 'none (no sibling DB configured)')
        ev['ordered_sibling_ids'] = []
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('structure', help='CIF / POSCAR / xyz path (disorder preserved from CIF)')
    ap.add_argument('--summary', action='store_true', help='human digest to stderr')
    ap.add_argument('--out', default=None, help='also write <out>/evidence.json')
    ap.add_argument('--icsd-id', type=int, default=None,
                    help='if this CIF IS an ICSD entry: stamp it and self-exclude from the sibling search')
    ap.add_argument('--no-siblings', action='store_true', help='skip the ordered-sibling lookup')
    args = ap.parse_args()

    ev = evidence_for(args.structure, iid=args.icsd_id, siblings=not args.no_siblings)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(f'{args.out}/evidence.json', 'w', encoding='utf-8') as fh:
            json.dump(ev, fh, indent=1, ensure_ascii=False, default=str)
    if args.summary:
        try:
            print(MEV._digest(ev), file=sys.stderr)
        except Exception as e:                       # digest is cosmetic; never fail the bundle
            print(f'(digest unavailable: {e})', file=sys.stderr)
    print(json.dumps(ev, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
