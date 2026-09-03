"""Lightweight occupancy parser for ICSD CIFs — shared by build_db and backfill.

Disorder detection without a full structure parse: scan the `_atom_site_*` loop,
find the `_atom_site_occupancy` column, and return the minimum per-site occupancy.
A site occupancy below 1 means a vacancy, a split site, or a mixed (shared) site —
all of which pymatgen would report as ``Structure.is_ordered == False``.
"""
from __future__ import annotations

from typing import Optional

# Occupancy at or above this is treated as fully occupied (absorbs rounding like
# 0.999 and reported uncertainties). Below it, the site is partially occupied.
DISORDER_TOL = 0.99


def min_site_occupancy(text: str) -> Optional[float]:
    """Return the minimum ``_atom_site_occupancy`` in the atom-site loop.

    Returns ``None`` when the CIF has no atom-site loop with an occupancy column
    (i.e. occupancy is unknown — not the same as "fully ordered").
    """
    lines = text.splitlines()
    n = len(lines)
    min_occ: Optional[float] = None
    i = 0
    while i < n:
        if lines[i].strip().lower() == "loop_":
            # Collect the loop header tags (consecutive lines beginning with '_').
            tags = []
            j = i + 1
            while j < n and lines[j].strip().startswith("_"):
                tags.append(lines[j].strip().lower())
                j += 1
            occ_idx = None
            for idx, t in enumerate(tags):
                if t == "_atom_site_occupancy":
                    occ_idx = idx
            has_fract = any(t.startswith("_atom_site_fract") for t in tags)
            if occ_idx is not None and has_fract:
                k = j
                while k < n:
                    row = lines[k].strip()
                    if (
                        not row
                        or row.lower() == "loop_"
                        or row.startswith("_")
                        or row.startswith(";")
                        or row.startswith("#")
                    ):
                        break
                    toks = row.split()
                    if len(toks) >= len(tags):
                        val = toks[occ_idx].split("(")[0]  # strip uncertainty
                        try:
                            o = float(val)
                        except ValueError:
                            pass
                        else:
                            min_occ = o if min_occ is None else min(min_occ, o)
                    k += 1
                i = k
                continue
        i += 1
    return min_occ


def is_disordered(min_occ: Optional[float], tol: float = DISORDER_TOL) -> Optional[bool]:
    """Classify from a minimum occupancy. ``None`` occ → ``None`` (unknown)."""
    if min_occ is None:
        return None
    return min_occ < tol
