#!/usr/bin/env python3
"""Search the ICSD SQLite metadata database and extract CIFs from the zip.

Examples:
    icsd-query chemsys In-Ca-O                # exact ternary chemsys
    icsd-query chemsys Li-La-Zr-O             # exact quaternary chemsys
    icsd-query subsystem Li,La,Zr,O           # contains AT LEAST these elements
    icsd-query subsystem Li,La,Zr,O --exact   # equivalent to chemsys lookup
    icsd-query formula LiFePO4                # by reduced formula
    icsd-query sg 14 --chemsys Cr-O-Te        # spacegroup + chemsys
    icsd-query chemsys Ag-Ge-Te --disordered  # only partial-occupancy structures
    icsd-query chemsys Cr-O-Te --ordered      # only fully ordered structures
    icsd-query stats                          # DB statistics
    icsd-query show 1                         # full row + raw CIF for ICSD #1
    icsd-query extract 1 5 6 -o /tmp/out      # extract CIF files to a directory
    icsd-query extract --chemsys Li-O -o /tmp/out
    icsd-query extract --chemsys Ag-Ge-Te --disordered -o /tmp/out

Disorder filter (chemsys / subsystem / formula / sg / extract):
    --disordered   only partial-occupancy (disordered) structures
    --ordered      only fully ordered structures
    (omit both for all; listings add 'dis' and 'minocc' columns)
"""
import argparse
import os
import sqlite3
import sys
import zipfile
from typing import Iterable, List, Sequence, Tuple

ICSD_DIR = os.environ.get("ICSD_DB_DIR") or os.path.expanduser("~/icsd_db")
DB_DEFAULT = os.path.join(ICSD_DIR, "icsd.sqlite")


def open_db(path: str = DB_DEFAULT) -> sqlite3.Connection:
    if not os.path.exists(path):
        sys.exit(f"DB not found: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def get_zip_path(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT value FROM meta WHERE key='zip_path'").fetchone()
    if not row:
        sys.exit("No zip_path in meta table")
    return row[0]


def sort_chemsys(spec: str) -> str:
    return "-".join(sorted(s for s in spec.split("-") if s))


def parse_elem_list(spec: str) -> List[str]:
    return [s.strip() for s in spec.replace(",", " ").split() if s.strip()]


def add_disorder_flags(p) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--disordered", action="store_true",
                   help="only partial-occupancy (disordered) structures")
    g.add_argument("--ordered", action="store_true",
                   help="only fully ordered structures")


def disorder_tristate(args):
    """Map --disordered/--ordered flags to True/False/None."""
    if getattr(args, "disordered", False):
        return True
    if getattr(args, "ordered", False):
        return False
    return None


# ------------------------- query primitives ------------------------- #

LIST_COLS = (
    "icsd_id, reduced_formula, chemsys, n_elements, "
    "spacegroup_number, spacegroup_symbol, "
    "a, b, c, alpha, beta, gamma, volume, density, "
    "min_occupancy, is_disordered"
)


def disorder_clause(disordered):
    """Return (sql_fragment, args) for a `disordered` tri-state (None/True/False)."""
    if disordered is None:
        return "", []
    return " AND is_disordered = ?", [1 if disordered else 0]


def q_chemsys(conn, chemsys: str, sg: int = None, disordered=None, limit: int = None) -> List[sqlite3.Row]:
    chemsys = sort_chemsys(chemsys)
    sql = f"SELECT {LIST_COLS} FROM cifs WHERE chemsys = ?"
    args: List = [chemsys]
    if sg is not None:
        sql += " AND spacegroup_number = ?"
        args.append(sg)
    frag, fargs = disorder_clause(disordered)
    sql += frag
    args += fargs
    sql += " ORDER BY icsd_id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, args).fetchall()


def q_subsystem(
    conn, elements: Sequence[str], exact: bool = False, sg: int = None,
    disordered=None, limit: int = None
) -> List[sqlite3.Row]:
    elements = sorted(set(elements))
    if exact:
        return q_chemsys(conn, "-".join(elements), sg=sg, disordered=disordered, limit=limit)
    placeholders = ",".join("?" * len(elements))
    sql = (
        f"SELECT {LIST_COLS} FROM cifs c "
        f"WHERE c.id IN ("
        f"  SELECT cif_id FROM cif_elements "
        f"  WHERE element IN ({placeholders}) "
        f"  GROUP BY cif_id HAVING COUNT(DISTINCT element) = ?"
        f")"
    )
    args: List = list(elements) + [len(elements)]
    if sg is not None:
        sql += " AND c.spacegroup_number = ?"
        args.append(sg)
    frag, fargs = disorder_clause(disordered)
    sql += frag
    args += fargs
    sql += " ORDER BY c.icsd_id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, args).fetchall()


def q_formula(conn, formula: str, sg: int = None, disordered=None, limit: int = None) -> List[sqlite3.Row]:
    sql = f"SELECT {LIST_COLS} FROM cifs WHERE reduced_formula = ?"
    args: List = [formula]
    if sg is not None:
        sql += " AND spacegroup_number = ?"
        args.append(sg)
    frag, fargs = disorder_clause(disordered)
    sql += frag
    args += fargs
    sql += " ORDER BY icsd_id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, args).fetchall()


def q_stats(conn) -> dict:
    n = conn.execute("SELECT COUNT(*) FROM cifs").fetchone()[0]
    n_err = conn.execute("SELECT COUNT(*) FROM cifs WHERE parse_error IS NOT NULL").fetchone()[0]
    n_chem = conn.execute("SELECT COUNT(DISTINCT chemsys) FROM cifs").fetchone()[0]
    n_form = conn.execute("SELECT COUNT(DISTINCT reduced_formula) FROM cifs").fetchone()[0]
    by_nel = conn.execute(
        "SELECT n_elements, COUNT(*) FROM cifs GROUP BY n_elements ORDER BY n_elements"
    ).fetchall()
    out = {
        "rows": n,
        "parse_errors": n_err,
        "distinct_chemsys": n_chem,
        "distinct_reduced_formula": n_form,
        "by_n_elements": [tuple(r) for r in by_nel],
    }
    try:
        out["disordered"] = conn.execute(
            "SELECT COUNT(*) FROM cifs WHERE is_disordered = 1"
        ).fetchone()[0]
        out["ordered"] = conn.execute(
            "SELECT COUNT(*) FROM cifs WHERE is_disordered = 0"
        ).fetchone()[0]
        out["occ_unknown"] = conn.execute(
            "SELECT COUNT(*) FROM cifs WHERE is_disordered IS NULL"
        ).fetchone()[0]
    except sqlite3.OperationalError:
        pass  # column not backfilled yet
    return out


# ------------------------- printing ------------------------- #


def print_rows(rows: Iterable[sqlite3.Row]) -> int:
    rows = list(rows)
    if not rows:
        print("(no matches)")
        return 0
    keys = set(rows[0].keys())
    has_dis = "is_disordered" in keys
    header = (
        f"{'ICSD':>7}  {'reduced_formula':<22}  {'chemsys':<22}  "
        f"{'sg':>3}  {'symbol':<14}  {'a':>7} {'b':>7} {'c':>7}  "
        f"{'alpha':>6} {'beta':>6} {'gamma':>6}  {'V':>8}  {'rho':>5}"
    )
    if has_dis:
        header += f"  {'dis':>3}  {'minocc':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        sym = (r["spacegroup_symbol"] or "")[:14]
        line = (
            f"{r['icsd_id']:>7}  "
            f"{(r['reduced_formula'] or ''):<22}  "
            f"{(r['chemsys'] or ''):<22}  "
            f"{r['spacegroup_number'] or 0:>3d}  "
            f"{sym:<14}  "
            f"{(r['a'] or 0):>7.3f} {(r['b'] or 0):>7.3f} {(r['c'] or 0):>7.3f}  "
            f"{(r['alpha'] or 0):>6.2f} {(r['beta'] or 0):>6.2f} {(r['gamma'] or 0):>6.2f}  "
            f"{(r['volume'] or 0):>8.2f}  {(r['density'] or 0):>5.2f}"
        )
        if has_dis:
            d = r["is_disordered"]
            mark = "?" if d is None else ("yes" if d else "no")
            occ = r["min_occupancy"]
            line += f"  {mark:>3}  " + (f"{occ:>6.3f}" if occ is not None else f"{'-':>6}")
        print(line)
    print(f"\n{len(rows)} match(es)")
    return len(rows)


# ------------------------- extract ------------------------- #


def extract_by_filenames(zip_path: str, filenames: Sequence[str], outdir: str) -> int:
    os.makedirs(outdir, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for fn in filenames:
            try:
                data = z.read(fn)
            except KeyError:
                continue
            base = os.path.basename(fn)
            with open(os.path.join(outdir, base), "wb") as f:
                f.write(data)
            n += 1
    return n


def filenames_for_icsd_ids(conn, icsd_ids: Sequence[int]) -> List[str]:
    placeholders = ",".join("?" * len(icsd_ids))
    sql = f"SELECT filename FROM cifs WHERE icsd_id IN ({placeholders}) ORDER BY icsd_id"
    return [r[0] for r in conn.execute(sql, list(icsd_ids)).fetchall()]


def filenames_for_query(
    conn, chemsys: str = None, formula: str = None, subsystem: List[str] = None,
    exact_subsystem: bool = False, sg: int = None, disordered=None,
) -> List[str]:
    dfrag, dargs = disorder_clause(disordered)
    if chemsys:
        cs = sort_chemsys(chemsys)
        sql = "SELECT filename FROM cifs WHERE chemsys = ?"
        args: List = [cs]
        if sg is not None:
            sql += " AND spacegroup_number = ?"
            args.append(sg)
        return [r[0] for r in conn.execute(sql + dfrag, args + dargs).fetchall()]
    if formula:
        sql = "SELECT filename FROM cifs WHERE reduced_formula = ?"
        args = [formula]
        if sg is not None:
            sql += " AND spacegroup_number = ?"
            args.append(sg)
        return [r[0] for r in conn.execute(sql + dfrag, args + dargs).fetchall()]
    if subsystem:
        rows = q_subsystem(conn, subsystem, exact=exact_subsystem, sg=sg, disordered=disordered)
        # need filename column - re-query
        ids = [r["icsd_id"] for r in rows]
        if not ids:
            return []
        return filenames_for_icsd_ids(conn, ids)
    return []


# ------------------------- CLI ------------------------- #


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser(prog="icsd-query", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_DEFAULT, help=f"SQLite path (default {DB_DEFAULT})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("chemsys", help="exact chemsys match (e.g. In-Ca-O)")
    p.add_argument("spec")
    p.add_argument("--sg", type=int, help="spacegroup number filter")
    add_disorder_flags(p)
    p.add_argument("--limit", type=int)

    p = sub.add_parser("subsystem", help="CIFs containing AT LEAST these elements")
    p.add_argument("spec", help="comma- or space-separated element list, e.g. 'Li,La,Zr,O'")
    p.add_argument("--exact", action="store_true", help="require exactly this set (= chemsys)")
    p.add_argument("--sg", type=int, help="spacegroup number filter")
    add_disorder_flags(p)
    p.add_argument("--limit", type=int)

    p = sub.add_parser("formula", help="search by reduced formula (e.g. LiFePO4)")
    p.add_argument("spec")
    p.add_argument("--sg", type=int)
    add_disorder_flags(p)
    p.add_argument("--limit", type=int)

    p = sub.add_parser("sg", help="search by spacegroup number (with optional chemsys)")
    p.add_argument("number", type=int)
    p.add_argument("--chemsys")
    add_disorder_flags(p)
    p.add_argument("--limit", type=int)

    p = sub.add_parser("show", help="show full record + raw CIF for an ICSD id")
    p.add_argument("icsd_id", type=int)
    p.add_argument("--no-cif", action="store_true")

    p = sub.add_parser("extract", help="extract CIFs to a directory")
    p.add_argument("ids", type=int, nargs="*", help="ICSD ids to extract")
    p.add_argument("--chemsys", help="extract everything in this chemsys")
    p.add_argument("--formula", help="extract everything matching this reduced formula")
    p.add_argument("--subsystem", help="extract everything containing these elements")
    p.add_argument("--exact", action="store_true")
    p.add_argument("--sg", type=int)
    add_disorder_flags(p)
    p.add_argument("-o", "--out", required=True, help="output directory")

    sub.add_parser("stats", help="print DB statistics")

    args = ap.parse_args(argv)
    conn = open_db(args.db)

    dis = disorder_tristate(args)
    if args.cmd == "chemsys":
        rows = q_chemsys(conn, args.spec, sg=args.sg, disordered=dis, limit=args.limit)
        print_rows(rows)
    elif args.cmd == "subsystem":
        els = parse_elem_list(args.spec)
        rows = q_subsystem(conn, els, exact=args.exact, sg=args.sg, disordered=dis, limit=args.limit)
        print_rows(rows)
    elif args.cmd == "formula":
        rows = q_formula(conn, args.spec, sg=args.sg, disordered=dis, limit=args.limit)
        print_rows(rows)
    elif args.cmd == "sg":
        if args.chemsys:
            rows = q_chemsys(conn, args.chemsys, sg=args.number, disordered=dis, limit=args.limit)
        else:
            sql = f"SELECT {LIST_COLS} FROM cifs WHERE spacegroup_number = ?"
            sargs: List = [args.number]
            frag, fargs = disorder_clause(dis)
            sql += frag
            sargs += fargs
            sql += " ORDER BY icsd_id"
            if args.limit:
                sql += f" LIMIT {int(args.limit)}"
            rows = conn.execute(sql, sargs).fetchall()
        print_rows(rows)
    elif args.cmd == "show":
        row = conn.execute("SELECT * FROM cifs WHERE icsd_id = ?", (args.icsd_id,)).fetchone()
        if not row:
            sys.exit(f"ICSD {args.icsd_id} not found")
        for k in row.keys():
            print(f"  {k:<22} {row[k]}")
        if not args.no_cif:
            zp = get_zip_path(conn)
            with zipfile.ZipFile(zp) as z:
                print("\n--- CIF ---")
                print(z.read(row["filename"]).decode("utf-8", "replace"))
    elif args.cmd == "extract":
        zp = get_zip_path(conn)
        if args.ids:
            fns = filenames_for_icsd_ids(conn, args.ids)
        else:
            fns = filenames_for_query(
                conn,
                chemsys=args.chemsys,
                formula=args.formula,
                subsystem=parse_elem_list(args.subsystem) if args.subsystem else None,
                exact_subsystem=args.exact,
                sg=args.sg,
                disordered=dis,
            )
        if not fns:
            sys.exit("No matches to extract")
        n = extract_by_filenames(zp, fns, args.out)
        print(f"Extracted {n} CIF(s) to {args.out}")
    elif args.cmd == "stats":
        s = q_stats(conn)
        print(f"  rows                     {s['rows']:,}")
        print(f"  parse_errors             {s['parse_errors']:,}")
        print(f"  distinct chemsys         {s['distinct_chemsys']:,}")
        print(f"  distinct reduced_formula {s['distinct_reduced_formula']:,}")
        if "disordered" in s:
            print(f"  disordered               {s['disordered']:,}")
            print(f"  ordered                  {s['ordered']:,}")
            print(f"  occupancy unknown        {s['occ_unknown']:,}")
        print(f"  by # elements:")
        for nel, c in s["by_n_elements"]:
            print(f"    {nel}: {c:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
