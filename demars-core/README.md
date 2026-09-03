# demars-core

**De-average a disordered crystal into a Minimal Atomistic Representation (MAR)** —
from any CIF, with a pluggable universal-MLIP backend.

A disordered crystal structure (partial occupancies, split sites, mixed sublattices)
is an *average* over many local arrangements — you can't run it directly as an
ordered cell. `demars-core` turns such a structure into an **ensemble of ordered,
relaxed decorations** at the ideal neutral stoichiometry, with a **representative
(typical) configuration** you can hand to a property calculation, an MLIP training
set, or a generative model.

It's the deterministic engine layer of DeMARS
(**De**averaged **M**inimal **A**tomistic **R**epresentation **S**ystem) — the
reproducible ~80% that needs no LLM and no proprietary data.

It is built for the **SevenNet ecosystem**: structures are relaxed by the
[SevenNet](https://github.com/MDIL-SNU/SevenNet) universal MLIP through the
[`torch-sim`](https://torchsim.github.io/torch-sim/) batched-relaxation engine — the
intended, supported backend. (Any ASE calculator can be used instead — see below —
but SevenNet + torch-sim is the default.)

---

## Install

demars-core is part of the **SevenNet ecosystem**. Installing it pulls the
[`torch-sim`](https://torchsim.github.io/torch-sim/) relaxation engine
automatically (`torch-sim-atomistic`), plus numpy / ase / pymatgen. The
**`[sevennet]` extra** additionally pulls [`sevenn`](https://github.com/MDIL-SNU/SevenNet)
(the SevenNet potential). **`torch` is a transitive dependency of `sevenn` and is
CUDA-specific** — for GPU, install a CUDA-matched `torch` *first*, then:

```bash
# GPU: install a torch matched to your CUDA first (see SevenNet / PyTorch docs), then:
pip install -e ".[sevennet]"        # demars-core + torch-sim (auto) + sevenn
pip install -e ".[sevennet,test]"   # also pytest, to run the suite
```

Prefer to manage `sevenn` / `torch` yourself? Install them per the SevenNet docs and
drop the extra (`pip install -e .`). A non-SevenNet ASE calculator can be used
instead — see below — but SevenNet + torch-sim is the supported default.

## Quickstart

**Python:**
```python
from demars_core import deaverage

rec = deaverage("disordered.cif", calculator="sevennet", out_dir="./mar")
print(rec.summary())
# -> MAR de-averaged (sevennet); ensemble 30/30 relaxed; spread 12 meV/at; ...
```

**Command line:**
```bash
demars run disordered.cif -o results/foo        # de-average one structure
demars gallery results --serve                  # browse a pile of results (below)
```

`source` accepts a **CIF path, structure-file path, raw CIF text, a pymatgen
`Structure`, or an `ase.Atoms`**. With `out_dir` it writes:

| file | what |
|------|------|
| `ensemble.xyz` | every sampled decoration, energy-marked (OVITO-native trajectory) |
| `representative.cif` / `.vasp` | the typical (lowest sampled) configuration |
| `deaverage_output.json` | full result: distribution, representative, engine flags, gates. NOT `record.json` -- that name belongs to the stage-6 mar-1.0 record, a different schema |

## MLIP backend — SevenNet (default), or any ASE calculator

Structures are relaxed by **SevenNet** by default (batched GPU relax via `torch-sim`,
with the nano-sample → omni-mpa-final tiering). The engine stays agnostic enough to
drive any ASE calculator if you need a different potential, but SevenNet + torch-sim
is the intended, supported path:

```python
# SevenNet — the intended backend
deaverage(cif, calculator="sevennet")          # or "7net-nano" / "7net-omni" / "<ckpt.pth>"

# any ASE calculator instance -> generic serial-FIRE path (secondary)
from mace.calculators import mace_mp
deaverage(cif, calculator=mace_mp())
```

Dispersion (D3) is **off by default** — DeMARS reproduces dispersion-free PBE labels;
pass `d3=True` only when the chemistry genuinely needs it. `d3=True` switches the model
to `sevenn.torchsim.SevenNetD3Model` (D3 with Becke-Johnson damping + PBE parameters)
and **requires a CUDA GPU**.

SevenNet specs use torch-sim's batched GPU relaxer (FIRE + Frechet cell filter). GPU memory is
handled by `InFlightAutoBatcher`: the budget is **probed once per model and cached**, held at 80 % of
the probed limit for headroom, and **shrunk 0.7x and retried up to three times on a runtime OOM**, so
one oversized batch degrades instead of killing the run. **An OOM that survives every retry is
raised, never recorded as "no valid relaxations"** — an environment failure must not reach the record
as a chemical one. `d3=True` likewise raises on CPU rather than quietly returning dispersion-free
energies. Any ASE `Calculator` instance is routed through a serial FIRE relaxer instead.

## Pile up results → a browsable gallery

Run many structures, then build a self-contained, **offline** web gallery:

```bash
demars run a.cif -o results/a
demars run b.cif -o results/b
demars gallery results --serve        # http://localhost:8000/
```

- **Filterable index** — one card per result; search by formula / element / name; status dropdown; live count.
- **Per-result detail page** — ensemble energy histogram (Δ above ground state) + the full record + raw JSON.
- **3-D viewer** — click through to the ensemble trajectory in the bundled view3d viewer (3Dmol.js, no CDN).

Re-run `demars gallery` any time to refresh as the pile grows. Everything is static —
works under any `python -m http.server`, no internet, no accounts.

## How to read a MAR (important)

- The **representative is the *typical* configuration, not a ground state.** A
  0 K MLIP energy is internal energy `E`, blind to `−TS`; a disordered MAR sitting
  *above* the ordered structure in 0 K energy is *expected* (that's the entropy),
  not a bug.
- The engine targets the **ideal, charge-neutral formula stoichiometry** — a CIF's
  refined occupancies are a *reference*, not ground truth.
- Results carry the **uncertainty of whichever MLIP you plug in**.

## Testing

```bash
pytest                 # 386 tests (the end-to-end ones need an MLIP backend)
pytest -m "not mlip"   # detection-only, no GPU/backend
```

Fixtures are **9 disordered crystals from the Crystallography Open Database (COD,
public-domain)** — 6 single-mechanism + 3 complex (multi-sublattice / split-site /
coupled). **No ICSD data is used anywhere**, by design.

## Repository layout

```
demars_core/
  api.py          deaverage() — the public entry point
  io.py           any input → CIF text (occupancies + oxidation states preserved)
  calculators.py  SevenNet-vs-generic MLIP dispatch + serial ASE relaxer
  models.py       checkpoint pinning + the compute provenance stamped into every record
  record.py       MARRecord (distribution, representative, ensemble writer, JSON)
  connectivity.py audit_frame() — the polyanion/rigid-unit gate (charge & fidelity are blind to it)
  disorder_class.py classify() — the Antypov O/S/V/P orbit descriptor (pure CIF, no relaxation)
  hull.py         compute_hull() — E_above_hull / decomposition (needs a Materials Project key)
  gallery.py      build_gallery() — the filterable pile-up web gallery
  cli.py          `demars run` / `demars gallery`
  _engine/        the DeMARS engine (mar_engine / mar_evidence / mar_record)
  viewer/         VENDORED view3d 3-D viewer (offline, self-contained)
  _torchsim.py    batched relaxation backend (SevenNet via torch-sim)
  _checkpoint.py  crash-resumable relaxation (keeps the configs that already finished)
  _smoketest.py   end-to-end check, no ICSD
tests/
  test_disorder_detection.py   fast, no GPU
  test_deaverage.py            end-to-end (needs an MLIP backend)
  test_*.py                    the rest: the gates, the hull, H placement, model provenance,
                               and the D<n> defect regressions (test_backtest_regressions.py)
  fixtures/                    disordered COD CIFs (public domain) + manifest,
                               plus one synthetic fixture (see fixtures/README.md)
```

## Python API

### `deaverage(source, calculator="sevennet", **opts) -> MARRecord`

The single public entry point.

| parameter | type | default | meaning |
|-----------|------|---------|---------|
| `source` | path / text / Structure / Atoms | — | the disordered crystal (below) |
| `calculator` | str or ASE Calculator | `"sevennet"` | MLIP backend (see *MLIP backend* above) |
| `n_samples` | int | `30` | decorations sampled + relaxed per round |
| `min_cell` | float | `15.0` | minimum supercell edge, in Angstrom |
| `excl` | float | `1.1` | cross-element exclusion cutoff (A) for split sites |
| `d3` | bool | `False` | D3 dispersion — **off by default** (DeMARS uses none); **needs a CUDA GPU** |
| `max_rounds` | int | `3` | self-driving re-enumeration rounds |
| `final` | bool | `True` | SevenNet only: recompute the winner at the omni-mpa modal |
| `out_dir` | str or None | `None` | if set, write `ensemble.xyz` + `representative.cif` + `deaverage_output.json` |
| `same_excl` | float or None | `None` | override the data-derived same-element cutoff |
| `couple_cut` | float or None | `None` | override the former-anion coupling cutoff |

Returns a `MARRecord` (below). If the input has no disorder, the record has
`status == "no-disorder"`; on a relaxation failure, `status == "error"`.

Production defaults (`n_samples=30`, `min_cell=15.0`) are for real runs. For quick
tests use smaller values (see *Testing* below).

### `MARRecord`

The structured result. Selected fields:

| field | meaning |
|-------|---------|
| `status` | `"de-averaged"` \| `"no-disorder"` \| `"error"` |
| `calculator` | backend used |
| `formula` | reduced (empirical) formula of the representative |
| `oxidation_states` | per-element oxidation states used for charge balance |
| `supercell` | supercell multiplicity `[a, b, c]` |
| `n_atoms_template` | atoms in the enumeration template |
| `distribution` | the ensemble result (below) |
| `engine` | engine flags (exclusion-merge, coordination integrity, …) |
| `final_MAR` | omni-mpa recompute of the representative (if `final=True`) |
| `ensemble_files` | paths written when `out_dir` is given |

`distribution` contains: `n_relaxed`, `n_total`, `ground_E_per_atom`,
`spread_meV`, `std_meV`, `energies_sorted`, and `representative` (a dict with
`label`, `E_per_atom`, `charge`, `min_dist_A`, `n_atoms`, …).

Methods:

- `rec.summary()` — a short human-readable string.
- `rec.to_dict()` — the full record as a JSON-serialisable dict.
- `rec.write_ensemble(out_dir)` — write the deliverable files (called
  automatically when `deaverage(..., out_dir=...)`).

### Input forms

`source` is normalised to CIF text internally, preserving occupancies and
oxidation states. Accepted:

- a **CIF path** (`"structure.cif"`);
- a **structure-file path** (POSCAR, xyz — read via pymatgen);
- **raw CIF text**;
- a **pymatgen `Structure`** (may carry partial occupancies);
- an **ase.Atoms** (ordered only — ASE cannot hold partial occupancies).

## Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'`** on `import demars_core`. The
  SevenNet runtime is required; install it (`pip install -e ".[sevennet]"`, after a
  CUDA-matched torch for GPU).
- **Checkpoint not found.** Point the backend at your SevenNet checkpoint via
  `SPINNER_NANO_CKPT=/path/to/checkpoint.pth`.
- **`GPU ... is not supported by this torch build` / falls back to CPU.** Your torch
  wheel has no kernels for this GPU's compute capability — install a matching torch.
  `DEMARS_DEVICE=cuda` forces the GPU anyway; `DEMARS_DEVICE=cpu` silences the notice.
- **`status == "no-disorder"`.** The input had no partial occupancies / split sites;
  nothing to de-average.
- **Broad ensemble spread flagged.** Often a missed coupled/rigid unit or strong
  ordering — inspect the `engine` flags in the record; consider `same_excl` /
  `couple_cut` overrides.
- **Slow relaxation.** Reduce `n_samples` and `min_cell` for tests; ensure the GPU
  path (SevenNet) is active rather than a CPU ASE calculator.

## Data & licensing

- **Code:** MIT. (The earlier batched-relaxation backend torch-sim replaced left only its
  `use_model` / `batched_fire_relax` names and the `(E, rel, val)` contract -- see
  `demars_core/_torchsim.py`; none of its implementation ships here.)
- **Vendored viewer:** `demars_core/viewer/js/3Dmol-min.js` is 3Dmol.js, BSD-3-Clause --
  see `demars_core/viewer/PROVENANCE.txt` and `js/3Dmol-LICENSE.txt`.
- **No ICSD data ships** — ICSD is redistribution-restricted; the engine imports
  `icsd_db` lazily and only its (unused-here) id-wrappers ever touch it.
- **Test fixtures:** COD, public domain (CC0).
- **MLIP checkpoints** (SevenNet, etc.): under their own licenses — you download them.

## Status

Prototype. The deterministic engine +
pluggable MLIP + gallery are working and tested; the LLM-driven judgment layer of
the full DeMARS pipeline (the ~20% of hard, custom cases) is a separate component
and is **not** part of this package.
