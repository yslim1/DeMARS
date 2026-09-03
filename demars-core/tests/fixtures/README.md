# Test fixtures — disordered crystals (COD, public domain)

Every `cod_*.cif` here is from the **Crystallography Open Database**
(https://www.crystallography.net), where all data are placed in the **public
domain (CC0)** — freely redistributable. **No ICSD data is used anywhere in this
package**, by design (ICSD is redistribution-restricted).

Each file is `cod_<COD-id>.cif`; source it at
`https://www.crystallography.net/cod/<COD-id>.cif`. Metadata (formula, mechanism,
complexity) is in `manifest.json`, which the tests read.

One fixture is **not** from a database: `synth_bcc_full_mixed.cif` is a
pymatgen-written idealisation (a = 3.83 A, one mixed Tm/Mg site at occupancy
1.0) that `test_backtest_regressions.py` uses for D43 — a fully-occupied
same-element orbit the split must not merge. It carries no `manifest.json`
entry, because it stands for a shape rather than for a real crystal.

## Single-mechanism (`simple`)

| file | formula | mechanism |
|------|---------|-----------|
| `cod_1521474.cif` | Y₀.₁₉Zr₁.₈₁O₃.₉ | aliovalent Y→Zr substitution + O vacancy (YSZ, fluorite) |
| `cod_1011163.cif` | Fe₃.₆₄O₄ | cation-vacancy disorder (non-stoichiometric Fe oxide) |
| `cod_9003141.cif` | K₃.₆Na₀.₄Cl₄ | (K,Na) substitutional solid solution (halide) |
| `cod_1006128.cif` | Sr₀.₅La₃.₅Mn₄O₁₂ | (La,Sr) A-site solid solution (perovskite) |
| `cod_1010497.cif` | Mg₇.₂Fe₀.₈Si₄O₁₆ | (Mg,Fe) solid solution, two M sites (olivine) |
| `cod_1520848.cif` | Ti₁.₃₇Fe₁₀.₁₈O₁₈ | mixed Fe/Ti cation site + vacancy (Fe-Ti oxide) |

## Complex (multi-sublattice / split-site / coupled)

| file | formula | mechanism |
|------|---------|-----------|
| `cod_4000330.cif` | K₀.₄₈Mn₁.₉₄O₅.₁₈ | split tunnel-K sites + mixed-valence Mn framework (hollandite) |
| `cod_9005182.cif` | Ca₁.₈Mg₃.₆₄Al₁.₁₆Si₇.₄O₂₄ | partial Ca site + Mg/Al mixed tetrahedral sites (melilite) |
| `cod_1544358.cif` | Sr₃Fe₃O₈ | split O sites + O-vacancy + mixed-valence Fe (Sr-Fe oxide) |

To refresh or extend these, re-run the discovery approach in the commit that added
them (COD REST `result?el1=…&format=json` → download → keep `not is_ordered`).
