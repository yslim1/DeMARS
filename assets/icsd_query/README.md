# `icsd-query` — local ICSD metadata index

A SQLite index over a local ICSD CIF archive, plus a CLI and a small Python API. DeMARS uses it
for two things: pulling input CIFs by id (`icsd-query extract`), and the **ordered-sibling
search** that `mar_evidence` runs during stage ①a.

**The CIF archive itself is not here and cannot be redistributed.** ICSD content is licensed;
`icsd_cif.zip` (~352 MB) and the `icsd.sqlite` built from it (~76 MB, licensed metadata for
241k entries) are both excluded on purpose. What is here is the code that turns *your own*
licensed archive into the index — bring the zip, run one command, get the database.

## What is in this directory

| file | role |
|---|---|
| `build_db.py` | zip → `icsd.sqlite`. Regex metadata extraction, multiprocess, one pass. |
| `cif_occupancy.py` | occupancy parser shared by the two builders — decides `is_disordered`. |
| `backfill_disorder.py` | adds `min_occupancy` / `is_disordered` to an **already built** DB. Only needed for a DB predating those columns; a fresh `build_db.py` includes them. |
| `query.py` | the CLI (`chemsys` · `subsystem` · `formula` · `sg` · `show` · `extract` · `stats`). |
| `icsd_db.py` | Python API — `search()`, `get_cif_text()`, `get_structure()`, `get_atoms()`, `stats()`. |
| `icsd-query` | launcher to drop in `~/.local/bin/`. |

## Install

Everything is keyed off **one variable, `ICSD_DB_DIR`** — the directory that holds the code, the
zip, and the database side by side. It is the same variable DeMARS reads (`paths.icsd_db` in
`demars.yaml`), so setting it once covers both. Unset, everything defaults to `~/icsd_db`.

```bash
# 1. lay out the directory
export ICSD_DB_DIR="$HOME/icsd_db"          # or wherever; put it in your profile
mkdir -p "$ICSD_DB_DIR"
cp assets/icsd_query/*.py "$ICSD_DB_DIR/"

# 2. add YOUR licensed archive, under exactly this name
cp /path/to/your/icsd_cif.zip "$ICSD_DB_DIR/icsd_cif.zip"

# 3. build the index  (~241k CIFs; minutes, scales with cores; needs pymatgen + tqdm)
cd "$ICSD_DB_DIR" && python build_db.py

# 4. put the CLI on PATH
install -m 755 assets/icsd_query/icsd-query "$HOME/.local/bin/icsd-query"
icsd-query stats
```

`build_db.py` writes the zip's absolute path into the DB's `meta` table (`zip_path`), and
`show` / `extract` read the CIFs back through it. **Move the zip and those two break** — rebuild,
or update `meta` by hand.

Requirements: Python 3.9+, `pymatgen`, `tqdm`. `icsd_db.get_atoms()` also wants `ase`.
The DeMARS conda env (`demars`) already has all of them.

### Expected result

```
  rows                     241,152
  parse_errors             70
  distinct chemsys         58,460
  distinct reduced_formula 148,040
  disordered               118,170
  ordered                  122,982
  occupancy unknown        0
```

`disordered` counts entries with any site occupancy below `DISORDER_TOL = 0.99`
(`cif_occupancy.py`) — vacancies, split sites, and mixed sites alike. That column is what
DeMARS filters the corpus on, so a mismatch here means a different archive vintage, not a bug.
The 70 `parse_errors` are CIFs whose formula line does not match any of the three ICSD spellings;
they stay in the table with null metadata rather than being dropped.

## Wiring it into DeMARS

```yaml
# demars.yaml
paths:
  icsd_db: /home/<you>/icsd_db     # $ICSD_DB_DIR overrides
```

`tools/_icsd_env.py` exports that as `ICSD_DB_DIR`, and `tools/ordered_lookup.py` opens
**`$ICSD_DB_DIR/icsd.sqlite` and `$ICSD_DB_DIR/icsd_cif.zip`** — both, by those exact names, in
that one directory. The zip is not optional for DeMARS: the sibling search reads candidate CIFs
to check whether their site tables are fully ordered.

⚠️ **Anything importing `demars_core` must `import _icsd_env` first**, or the sibling bootstrap
silently degrades. See `AGENTS.md` 「Facts that bite」.

⚠️ **Leaving `icsd_db` unset is a supported, honest configuration** — the record then reports
`"none (no sibling DB configured)"`, which means **unchecked**. That is a different claim from
`"none — no fully-ordered ICSD entry of this composition"`, which means **checked and absent**.
Never report the first as the second.

## Usage

```bash
icsd-query chemsys Li-La-Zr-O                  # exact chemical system
icsd-query subsystem Li,La,Zr,O                # contains AT LEAST these elements
icsd-query subsystem Li,La,Zr,O --exact        # == chemsys
icsd-query formula LiFePO4                     # by reduced formula
icsd-query sg 230 --chemsys Li-La-Zr-O         # spacegroup + chemsys
icsd-query chemsys Ag-Ge-Te --disordered       # partial-occupancy only
icsd-query chemsys Cr-O-Te --ordered           # fully ordered only
icsd-query show 174086                         # full row + raw CIF  (--no-cif to skip)
icsd-query extract 174086 258159 -o tmp/cifs/  # by id
icsd-query extract --chemsys Li-O --disordered -o tmp/cifs/
icsd-query stats
```

`--db /path/to/other.sqlite` points any subcommand at a different index.

```python
import os; os.environ.setdefault("ICSD_DB_DIR", "/home/you/icsd_db")
from icsd_db import search, get_structure

for r in search(chemsys="Li-La-Zr-O", disordered=True, limit=5):
    s = get_structure(r["icsd_id"])            # pymatgen.Structure
    print(r["icsd_id"], r["reduced_formula"], s.is_ordered)
```

## Notes

- **`build_db.py` deletes and rebuilds `icsd.sqlite` from scratch** every run. Cheap and
  idempotent, but it drops anything you added by hand.
- Metadata comes from regex over the CIF text, not a structure parse — ICSD CIFs are uniform
  enough for it and it is orders of magnitude faster. `is_disordered` is therefore a *site-table*
  judgement, which is exactly what DeMARS wants: it is asking whether the deposited model is
  averaged, not whether the real material is.
- `n_elements: None` rows (72) are the parse failures above, kept for accounting.
