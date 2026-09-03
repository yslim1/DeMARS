"""Ordered-sibling lookup backend for mar_evidence's `_sib` bootstrap.

Search ICSD for entries with the SAME nominal reduced composition whose CIF site
table is fully ordered (no mixed, no partial occupancy). Same-SG ordered sibling =
a re-refinement that resolves the disorder; different-SG = ordered polymorph /
superstructure.

It is NOT run directly: `mar_evidence`
exec's it via the `DEMARS_ORDERED_LOOKUP` env var, truncated at the
`ledger = json.load` line below, and keeps the namespace as `_sib`
(needs: Composition, db, z, is_ordered). Keep that sentinel line last.

The database directory comes from ICSD_DB_DIR, which `_icsd_env` sets from
demars.yaml (`paths.icsd_db`). Nothing here is hard-coded to one machine: with
the variable unset this module raises, mar_evidence catches it, and the bundle
reports "none (no sibling DB configured)" -- which means UNCHECKED, not "no
ordered form exists".
"""
import json, os, sqlite3, zipfile
from collections import defaultdict

from pymatgen.core import Composition

_DIR = os.environ.get('ICSD_DB_DIR')
if not _DIR:
    raise RuntimeError('ICSD_DB_DIR is not set (see paths.icsd_db in demars.yaml)')
db = sqlite3.connect(os.path.join(_DIR, 'icsd.sqlite'), check_same_thread=False)
z = zipfile.ZipFile(os.path.join(_DIR, 'icsd_cif.zip'))


def parse_occs(text):
    """[(xyz, occ)] per site row; minimal parser (same as picker)"""
    lines = text.split('\n'); i = 0; n = len(lines); out = []
    while i < n:
        if lines[i].lstrip().startswith('_atom_site_'):
            hdr = []
            while i < n and lines[i].lstrip().startswith('_atom_site_'):
                hdr.append(lines[i].strip()); i += 1
            if '_atom_site_occupancy' not in hdr: return None
            oc = hdr.index('_atom_site_occupancy')
            xc = hdr.index('_atom_site_fract_x') if '_atom_site_fract_x' in hdr else None
            while i < n:
                s = lines[i].strip()
                if not s or s[0] in '_#' or s.startswith(('loop_', 'data_')): break
                t = s.split()
                if len(t) >= len(hdr):
                    try: occ = float(t[oc].split('(')[0])
                    except Exception: occ = 1.0
                    xyz = tuple(round(float(t[xc + k].split('(')[0]), 4) for k in range(3)) if xc is not None else None
                    out.append((xyz, occ))
                i += 1
            return out
        i += 1
    return out


def is_ordered(fn):
    """True iff the zipped CIF `fn` has no mixed and no partial site. None if unreadable."""
    try: txt = z.read(fn).decode('utf-8', 'ignore')
    except KeyError: return None
    sites = parse_occs(txt)
    if not sites: return None
    grp = defaultdict(float)
    for xyz, occ in sites: grp[xyz] += occ
    pos_count = defaultdict(int)
    for xyz, occ in sites: pos_count[xyz] += 1
    mixed = any(c > 1 for c in pos_count.values())
    partial = any(abs(v - 1.0) > 0.02 for v in grp.values())
    return not (mixed or partial)


# --- SENTINEL: mar_evidence truncates the exec'd source here. Nothing below runs. ---
ledger = json.load(open(os.path.join(_DIR, 'nonexistent-ledger.json')))
