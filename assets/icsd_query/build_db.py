#!/usr/bin/env python3
"""Build SQLite metadata database from the ICSD CIF zip.

Extracts metadata (formula, chemsys, spacegroup, lattice, density, ...) via
regex from each CIF (no full structure parse needed — ICSD CIFs are clean).
"""
import multiprocessing as mp
import os
import re
import sqlite3
import sys
import zipfile

from pymatgen.core.composition import Composition
from tqdm import tqdm

from cif_occupancy import is_disordered, min_site_occupancy

# Everything lives in one directory, named by $ICSD_DB_DIR -- the same variable DeMARS'
# sibling search reads (see assets/demars.yaml.example, `paths.icsd_db`).
ICSD_DIR = os.environ.get("ICSD_DB_DIR") or os.path.expanduser("~/icsd_db")
ZIP = os.path.join(ICSD_DIR, "icsd_cif.zip")
DB = os.path.join(ICSD_DIR, "icsd.sqlite")
WORKERS = min(mp.cpu_count(), 16)
BATCH = 1000

CELL_KEYS = [
    ("_cell_length_a", "a"),
    ("_cell_length_b", "b"),
    ("_cell_length_c", "c"),
    ("_cell_angle_alpha", "alpha"),
    ("_cell_angle_beta", "beta"),
    ("_cell_angle_gamma", "gamma"),
    ("_cell_volume", "volume"),
    ("_exptl_crystal_density_diffrn", "density"),
    ("_cell_formula_units_Z", "z_form"),
]

NULL_FIELDS = (
    "formula_sum chemsys n_elements reduced_formula "
    "spacegroup_number spacegroup_symbol"
).split()


def parse_one(name, text):
    rec = {"filename": name, "icsd_id": None, "elements": [], "error": None}
    try:
        m = re.search(r"icsd_(\d+)\.cif", name)
        rec["icsd_id"] = int(m.group(1)) if m else None

        # ICSD CIFs store the formula in one of three forms:
        #   _chemical_formula_sum 'multi token string'    ← quoted
        #   _chemical_formula_sum Cu1                     ← bare token (elementals)
        #   _chemical_formula_sum\n;\n<lines>\n;          ← text block
        m = re.search(
            r"_chemical_formula_sum\s+(?:'([^']+)'|;\s*([^;]+?)\s*;|(?!;)(\S+))",
            text,
            re.DOTALL,
        )
        formula = None
        if m:
            raw = m.group(1) or m.group(2) or m.group(3)
            formula = " ".join(raw.split())  # collapse whitespace
        rec["formula_sum"] = formula

        if formula:
            comp = Composition(formula)
            els = sorted({e.symbol for e in comp.elements})
            rec["elements"] = els
            rec["chemsys"] = "-".join(els)
            rec["n_elements"] = len(els)
            rec["reduced_formula"] = comp.reduced_formula
        else:
            rec["chemsys"] = None
            rec["n_elements"] = None
            rec["reduced_formula"] = None

        for key, dst in CELL_KEYS:
            mm = re.search(rf"{key}\s+(\S+)", text)
            if mm:
                v = mm.group(1).split("(")[0].rstrip(".")
                try:
                    rec[dst] = float(v)
                except ValueError:
                    rec[dst] = None
            else:
                rec[dst] = None

        mm = re.search(r"_space_group_IT_number\s+(\d+)", text)
        rec["spacegroup_number"] = int(mm.group(1)) if mm else None
        mm = re.search(r"_space_group_name_H-M_alt\s+'([^']+)'", text) or re.search(
            r"_symmetry_space_group_name_H-M\s+'([^']+)'", text
        )
        rec["spacegroup_symbol"] = mm.group(1).strip() if mm else None

        occ = min_site_occupancy(text)
        rec["min_occupancy"] = occ
        dis = is_disordered(occ)
        rec["is_disordered"] = None if dis is None else int(dis)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        for k in NULL_FIELDS:
            rec.setdefault(k, None)
        for _, dst in CELL_KEYS:
            rec.setdefault(dst, None)
    return rec


def worker_slice(args):
    zp, names = args
    out = []
    with zipfile.ZipFile(zp) as z:
        for n in names:
            try:
                t = z.read(n).decode("utf-8", errors="replace")
                out.append(parse_one(n, t))
            except Exception as e:
                rec = {
                    "filename": n,
                    "error": f"read failed: {e}",
                    "elements": [],
                    "icsd_id": None,
                }
                for k in NULL_FIELDS:
                    rec[k] = None
                for _, dst in CELL_KEYS:
                    rec[dst] = None
                out.append(rec)
    return out


SCHEMA = """
CREATE TABLE cifs (
    id INTEGER PRIMARY KEY,
    icsd_id INTEGER,
    filename TEXT NOT NULL UNIQUE,
    formula_sum TEXT,
    reduced_formula TEXT,
    chemsys TEXT,
    n_elements INTEGER,
    spacegroup_number INTEGER,
    spacegroup_symbol TEXT,
    a REAL, b REAL, c REAL,
    alpha REAL, beta REAL, gamma REAL,
    volume REAL,
    density REAL,
    z_form REAL,
    min_occupancy REAL,
    is_disordered INTEGER,
    parse_error TEXT
);
CREATE TABLE cif_elements (
    cif_id INTEGER NOT NULL,
    element TEXT NOT NULL,
    PRIMARY KEY (element, cif_id)
) WITHOUT ROWID;
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""

INDEXES = """
CREATE INDEX idx_chemsys ON cifs(chemsys);
CREATE INDEX idx_reduced ON cifs(reduced_formula);
CREATE INDEX idx_sg      ON cifs(spacegroup_number);
CREATE INDEX idx_nel     ON cifs(n_elements);
CREATE INDEX idx_icsd    ON cifs(icsd_id);
CREATE INDEX idx_disord  ON cifs(is_disordered);
"""

INSERT_MAIN = """INSERT INTO cifs(
    icsd_id, filename, formula_sum, reduced_formula, chemsys, n_elements,
    spacegroup_number, spacegroup_symbol,
    a, b, c, alpha, beta, gamma, volume, density, z_form,
    min_occupancy, is_disordered, parse_error)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""


def main():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    if os.path.exists(DB):
        os.remove(DB)
    conn = sqlite3.connect(DB)
    conn.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous  = OFF;
        PRAGMA temp_store   = MEMORY;
        PRAGMA cache_size   = -200000;
        """
        + SCHEMA
    )
    conn.execute("INSERT INTO meta VALUES ('zip_path', ?)", (ZIP,))
    conn.execute("INSERT INTO meta VALUES ('built_on', datetime('now'))")

    with zipfile.ZipFile(ZIP) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".cif"))
    print(f"Total CIFs: {len(names)}")

    chunks = [names[i : i + BATCH] for i in range(0, len(names), BATCH)]
    args = [(ZIP, c) for c in chunks]

    cur = conn.cursor()
    n_ok = n_err = 0
    with mp.Pool(WORKERS) as pool:
        for batch in tqdm(
            pool.imap_unordered(worker_slice, args),
            total=len(args),
            desc="parse+insert",
        ):
            for rec in batch:
                row = (
                    rec["icsd_id"],
                    rec["filename"],
                    rec.get("formula_sum"),
                    rec.get("reduced_formula"),
                    rec.get("chemsys"),
                    rec.get("n_elements"),
                    rec.get("spacegroup_number"),
                    rec.get("spacegroup_symbol"),
                    rec.get("a"),
                    rec.get("b"),
                    rec.get("c"),
                    rec.get("alpha"),
                    rec.get("beta"),
                    rec.get("gamma"),
                    rec.get("volume"),
                    rec.get("density"),
                    rec.get("z_form"),
                    rec.get("min_occupancy"),
                    rec.get("is_disordered"),
                    rec.get("error"),
                )
                cur.execute(INSERT_MAIN, row)
                rowid = cur.lastrowid
                for el in rec.get("elements", []):
                    cur.execute(
                        "INSERT INTO cif_elements(cif_id, element) VALUES (?, ?)",
                        (rowid, el),
                    )
                if rec.get("error"):
                    n_err += 1
                else:
                    n_ok += 1
    conn.commit()
    print(f"Inserted: {n_ok} OK, {n_err} errors")
    print("Building indexes…")
    conn.executescript(INDEXES)
    print("VACUUM + ANALYZE…")
    conn.execute("VACUUM")
    conn.execute("ANALYZE")
    conn.close()
    sz = os.path.getsize(DB) / 1024 / 1024
    print(f"DB size: {sz:.1f} MB at {DB}")


if __name__ == "__main__":
    sys.exit(main())
