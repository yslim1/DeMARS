"""Programmatic access to the ICSD CIF SQLite database.

Example:
    from icsd_db import search, get_cif_text, get_structure, get_atoms

    rows = search(chemsys="Li-La-Zr-O", sg=230)
    for r in rows:
        s = get_structure(r["icsd_id"])      # pymatgen.Structure
        a = get_atoms(r["icsd_id"])          # ase.Atoms
"""
from __future__ import annotations

import os
import sqlite3
import zipfile
from typing import List, Optional, Sequence

ICSD_DIR = os.environ.get("ICSD_DB_DIR") or os.path.expanduser("~/icsd_db")
DB_PATH = os.path.join(ICSD_DIR, "icsd.sqlite")

_conn = None
_zip_path = None


def _get_conn() -> sqlite3.Connection:
    global _conn, _zip_path
    if _conn is None:
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(DB_PATH)
        _conn = sqlite3.connect(DB_PATH)
        _conn.row_factory = sqlite3.Row
        row = _conn.execute("SELECT value FROM meta WHERE key='zip_path'").fetchone()
        _zip_path = row[0] if row else None
    return _conn


def _sort_chemsys(spec: str) -> str:
    return "-".join(sorted(s for s in spec.split("-") if s))


def search(
    chemsys: Optional[str] = None,
    formula: Optional[str] = None,
    subsystem: Optional[Sequence[str]] = None,
    exact_subsystem: bool = False,
    sg: Optional[int] = None,
    n_elements: Optional[int] = None,
    disordered: Optional[bool] = None,
    limit: Optional[int] = None,
) -> List[sqlite3.Row]:
    """Run a flexible search and return rows (sqlite3.Row, dict-like).

    - chemsys: exact match, e.g. "Li-La-Zr-O" (auto-sorted)
    - formula: exact reduced formula, e.g. "LiFePO4"
    - subsystem: list of elements that must all be present (or exact set if exact_subsystem)
    - sg: spacegroup number filter
    - n_elements: filter by number of distinct elements
    - disordered: True → only partial-occupancy structures; False → only fully
      ordered; None → no filter. (Entries with unknown occupancy are excluded
      from both True and False.)
    """
    conn = _get_conn()
    where = []
    args: list = []

    if chemsys:
        where.append("c.chemsys = ?")
        args.append(_sort_chemsys(chemsys))
    if formula:
        where.append("c.reduced_formula = ?")
        args.append(formula)
    if sg is not None:
        where.append("c.spacegroup_number = ?")
        args.append(sg)
    if n_elements is not None:
        where.append("c.n_elements = ?")
        args.append(n_elements)
    if disordered is not None:
        where.append("c.is_disordered = ?")
        args.append(1 if disordered else 0)

    if subsystem:
        elements = sorted(set(subsystem))
        if exact_subsystem:
            where.append("c.chemsys = ?")
            args.append("-".join(elements))
        else:
            placeholders = ",".join("?" * len(elements))
            where.append(
                f"c.id IN (SELECT cif_id FROM cif_elements WHERE element IN ({placeholders}) "
                f"GROUP BY cif_id HAVING COUNT(DISTINCT element) = ?)"
            )
            args.extend(elements)
            args.append(len(elements))

    sql = "SELECT c.* FROM cifs c"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY c.icsd_id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, args).fetchall()


def get_cif_text(icsd_id: int) -> str:
    """Return raw CIF text for an ICSD id."""
    conn = _get_conn()
    row = conn.execute("SELECT filename FROM cifs WHERE icsd_id = ?", (icsd_id,)).fetchone()
    if not row:
        raise KeyError(f"ICSD {icsd_id} not in database")
    with zipfile.ZipFile(_zip_path) as z:
        return z.read(row["filename"]).decode("utf-8", "replace")


def get_structure(icsd_id: int):
    """Return a pymatgen.Structure for an ICSD id."""
    from pymatgen.io.cif import CifParser
    return CifParser.from_str(get_cif_text(icsd_id)).parse_structures(primitive=False)[0]


def get_atoms(icsd_id: int):
    """Return an ase.Atoms for an ICSD id."""
    from io import StringIO
    from ase.io import read
    return read(StringIO(get_cif_text(icsd_id)), format="cif")


def stats() -> dict:
    conn = _get_conn()
    n = conn.execute("SELECT COUNT(*) FROM cifs").fetchone()[0]
    n_chem = conn.execute("SELECT COUNT(DISTINCT chemsys) FROM cifs").fetchone()[0]
    n_form = conn.execute("SELECT COUNT(DISTINCT reduced_formula) FROM cifs").fetchone()[0]
    return {"rows": n, "distinct_chemsys": n_chem, "distinct_reduced_formula": n_form}
