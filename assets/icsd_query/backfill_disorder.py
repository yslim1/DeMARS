#!/usr/bin/env python3
"""Backfill `min_occupancy` and `is_disordered` into an already-built ICSD DB.

Adds the two columns to the existing `cifs` table (if absent), then reads each
CIF from the zip once and populates them — no full metadata rebuild needed.

Usage:
    python3 backfill_disorder.py
"""
import multiprocessing as mp
import os
import sqlite3
import sys
import zipfile

from tqdm import tqdm

from cif_occupancy import is_disordered, min_site_occupancy

ICSD_DIR = os.environ.get("ICSD_DB_DIR") or os.path.expanduser("~/icsd_db")
DB = os.path.join(ICSD_DIR, "icsd.sqlite")
WORKERS = min(mp.cpu_count(), 16)
BATCH = 1000


def add_columns(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(cifs)").fetchall()}
    if "min_occupancy" not in cols:
        conn.execute("ALTER TABLE cifs ADD COLUMN min_occupancy REAL")
    if "is_disordered" not in cols:
        conn.execute("ALTER TABLE cifs ADD COLUMN is_disordered INTEGER")
    conn.commit()


def worker_slice(args):
    """(zip_path, [(id, filename), ...]) -> [(min_occ, is_disordered, id), ...]"""
    zp, rows = args
    out = []
    with zipfile.ZipFile(zp) as z:
        for cif_id, fn in rows:
            try:
                text = z.read(fn).decode("utf-8", "replace")
                occ = min_site_occupancy(text)
                dis = is_disordered(occ)
                out.append((occ, None if dis is None else int(dis), cif_id))
            except Exception:
                out.append((None, None, cif_id))
    return out


def main() -> int:
    conn = sqlite3.connect(DB)
    conn.executescript(
        "PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF; PRAGMA temp_store = MEMORY;"
    )
    add_columns(conn)

    zp = conn.execute("SELECT value FROM meta WHERE key='zip_path'").fetchone()[0]
    rows = conn.execute("SELECT id, filename FROM cifs ORDER BY id").fetchall()
    print(f"Rows to backfill: {len(rows):,}  (zip: {zp})")

    chunks = [rows[i : i + BATCH] for i in range(0, len(rows), BATCH)]
    args = [(zp, [(r[0], r[1]) for r in c]) for c in chunks]

    cur = conn.cursor()
    n_dis = n_unknown = 0
    with mp.Pool(WORKERS) as pool:
        for batch in tqdm(
            pool.imap_unordered(worker_slice, args), total=len(args), desc="backfill"
        ):
            cur.executemany(
                "UPDATE cifs SET min_occupancy = ?, is_disordered = ? WHERE id = ?", batch
            )
            for occ, dis, _ in batch:
                if dis == 1:
                    n_dis += 1
                elif dis is None:
                    n_unknown += 1
    conn.commit()

    print("Rebuilding disorder index…")
    conn.execute("DROP INDEX IF EXISTS idx_disord")
    conn.execute("CREATE INDEX idx_disord ON cifs(is_disordered)")
    conn.execute("ANALYZE")
    conn.commit()

    total = len(rows)
    print(
        f"Done. disordered={n_dis:,}  ordered={total - n_dis - n_unknown:,}  "
        f"unknown(no occ column)={n_unknown:,}  of {total:,}"
    )
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
