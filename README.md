# DeMARS

**De-average disordered crystals into Minimal Atomistic Representations (MARs).**

A crystal structure with partial occupancies, split sites, or mixed sublattices is an *average* over
many local arrangements. You cannot run it: no ordered cell, no forces, no property calculation.
DeMARS turns such a structure into an **ensemble of ordered, relaxed decorations** at the ideal
neutral stoichiometry and picks a **representative** one — a MAR — that you can hand to a property
calculation, an MLIP training set, or a generative model.

The package has two layers, and the split is the point:

| layer | what it is | where |
|---|---|---|
| **engine** | deterministic. Enumerates decorations, relaxes them on a universal MLIP, recomputes the gates. No LLM, no proprietary data. | `demars-core/` (pip-installable `demars_core`), `tools/` (file-based stage CLIs) |
| **judgment** | an LLM analyst that reads the CIF's own evidence, decides the disorder mechanism and how to build for it, plus an adversarial reviewer that tries to refute the result | `.claude/skills/`, `.claude/agents/` |

**Code proves; the analyst judges.** Every number in a record comes from the engine; the skills
decide what to build and what it means, and the reviewer independently attacks it.

You can use the engine completely on its own. The judgment layer needs [Claude
Code](https://claude.com/claude-code) and is what turns a hard structure into a defensible one.

---

## Install

In Claude Code, `/setup` walks through everything below — env, `demars.yaml`, the optional ICSD
index, and a verification pass (`assets/setup/`, scripts usable on their own). By hand:

```bash
git clone https://github.com/yslim1/DeMARS-deploy.git && cd DeMARS-deploy
conda create -n demars python=3.12 && conda activate demars

# GPU: install a CUDA-matched torch FIRST (see the SevenNet / PyTorch docs), then:
pip install -e "./demars-core[sevennet]"        # engine + torch-sim + SevenNet
pip install -e "./demars-core[sevennet,test]"   # ... and pytest
```

Any ASE calculator works instead of SevenNet (MACE, CHGNet, ORB, …) via the serial path — the engine
never learns which potential it is talking to. SevenNet + torch-sim is the supported default.

Optional extras, both **off by default** because they build from source against a CUDA toolchain:
`[oeq]` and `[cueq]` install SevenNet's fused-kernel backends for `compute.accelerator` in the config.

**Distribution is this repository** (a clone, or its Zenodo archive) — not a standalone wheel. The
methodology lives in `.claude/`, which a wheel does not carry.

### Configure `demars.yaml`

Copy `assets/demars.yaml.example`. It holds the model checkpoints, the fused-kernel accelerator, the
optional local ICSD database, and an optional Materials Project key. Every record is stamped with the
resolved spec **and the sha256 of the weights actually loaded**, so a record says which checkpoint
produced it, not merely which path was configured.

> `./demars.yaml` is resolved **relative to the current directory**. Run from the repo root, or set
> `$DEMARS_CONFIG`, or keep a copy in `~/.config/demars/`. Without it the engine fails with an
> actionable message rather than guessing.

`paths.python` in that file is the **absolute path to this env's interpreter**, and it is the one
place the env location is written down. `tools/py` resolves it, so every stage command below runs as
`tools/py tools/<stage>.py …` rather than a bare `python`. Use it: an agent's shell is re-initialized
from your profile on every call, so a bare `python` is whatever that profile leaves behind — usually
an interpreter with no `demars_core`.

## Check that your install is right

```bash
assets/setup/doctor.sh                # config, interpreter, GPU, models, ICSD, MP key, version
tools/py tools/demars_reference.py    # runs bundled COD structures, compares to stored answers
pytest                                # the whole suite -- RUN FROM THE REPO ROOT
```

The reference check separates what is comparable from what is not: enumeration facts (supercell,
composition, charge, how many decorations exist) are compared **exactly** because they do not depend
on the potential or the hardware; energies are compared in a **wide band** that catches wrong weights
but not hardware noise; `n_relaxed` is **reported, never failed on**. Running with a different
potential than the reference was stamped with is normal and supported — the energy block is then
skipped with a note instead of silently passing.

> **Run `pytest` from the repo root.** From `demars-core/` you get that subtree only, and two tests
> fail there because they read `tools/…` by a relative path. Nothing is wrong with your install.

## Run one structure — the engine alone

```bash
demars run demars-core/tests/fixtures/cod_9003141.cif -o /tmp/kna --n-samples 6 --min-cell 8

# what lands in /tmp/kna
#   ensemble.xyz              every relaxed decoration, energy-marked (OVITO-native)
#   representative.cif/.vasp  the MAR
#   deaverage_output.json     the engine's own record  (NOT `record.json` -- see below)

demars gallery /tmp/kna --serve        # browsable pages + a 3-D viewer
```

This path has no analyst and no gates beyond the engine's own. It is the right tool when the disorder
is simple — one substitutional sublattice, no molecular ions, no split sites you need judged.

## Run one structure — the full pipeline

Evidence → mechanism → build → gates → adversarial review → a `record.json`. In Claude Code, from the
repo root:

```
deaverage demars-core/tests/fixtures/cod_4000330.cif
```

`CLAUDE.md` defines that trigger: analyst → reviewer, **one round** → a stage-⑥ record (see
*Keep the one-round rule* below). Single stages are documented in `tools/README.md`.

> **Three files, three names, no shared stem.** `engine.json` = the stage-③ contract ·
> `deaverage_output.json` = the engine's own deterministic output · `record.json` = the stage-⑥
> mar-1.0 record, the *only* authority on which structure is the MAR. Never resolve the MAR by
> globbing a run directory: rival `representative_final.cif` files live there, and one of them can be
> an ordered sibling that must never be adopted.

## Read the result without misreading it

`record.json` is the deliverable. Four fields decide whether you can trust it, and each of them
distinguishes *"checked and fine"* from *"not checked"* — a distinction the whole design turns on.

**`gates`** — five of them. `charge` and `fidelity` prove composition; `connectivity` proves that
discrete units (oxo-anions, molecular ions, octahedra) are intact; `sqs` proves the shipped frame is
not one of the enumeration's bracket probes; `hull` proves the phase is thermodynamically reachable
at all — the one question a perfectly faithful cell can still fail.

`connectivity` and `sqs` re-run themselves inside stage ⑥. **`hull` does not**: it needs a Materials
Project call and a relaxation, so it stays `not_run` until stage ④b's `hull.json` is handed in with
`--hull`. Configure the key (`$MP_API_KEY` or `materials_project.api_key`) or that gate is
permanently unchecked.

| `state` | means | `pass` |
|---|---|---|
| `derived` | it ran and examined something | `true` / `false` |
| `vacuous` | it ran; there was nothing of that kind to check | **`null`** |
| `ambiguous` | it ran but cannot separate "clean" from "examined nothing" | **`null`** |
| `not_run` | it could not run (`basis` says why) | **`null`** |

**`pass: true` is the only pass.** A `vacuous` connectivity gate means no element in that structure
centres a unit — not that the units are fine. Read `connectivity.not_examined` too: the gate covers a
fixed set of centre elements and names the cations it never looked at.

**`gates.hull` reports and does not judge, until you give it a line.** `hull.tol_eV_per_atom` in
`demars.yaml` (or `--hull-tol`) is `null` by default, and then that gate carries `E_above_hull` with
`pass: null` — a number for a human to read, not a verdict, and there is no default threshold to
inherit. The two terms that would set one are system-dependent: the configurational entropy the
0 K hull omits (a MAR is an *ordered approximant* of an entropy-stabilised phase, so it sits above
the hull legitimately) and the energy-scale error, which differs between the two hull modes. Measure
your corpus, then set it.

The hull's **energy scale** is stamped in `gates.hull.corrections` — `mp2020` (default: MP's own
hull scale, the one its published `energy_above_hull` is on; 75 of 79 Fe-Ti-O materials reproduce
there and none on the raw scale) or `none` (raw PBE(+U), internally consistent but not MP's hull).
The reference energies are recomputed at the MAR's own tier by default (`mode: self-consistent`), so
the MLIP-vs-DFT offset cancels instead of loading onto the MAR — measured, that offset was most of
what the cheaper `mp-direct` mode reported.

MP applies a correction only where the element is present *as an anion*, and GGA+U only for oxides
and fluorides of V/Cr/Mn/Fe/Co/Ni/W/Mo — so the offsets cancel in a hull decomposition unless it
crosses that role boundary. A mixed-valence oxide does cross it, and the scales can then differ by
~0.1 eV/atom and name different decomposition products.

**`review`** — `null` means **UNREVIEWED**, which is not reviewed-and-clean. A record with a null
review must never be reported as confirmed. When present it is a ledger of every analyst→reviewer
round, with `final_verdict` and any `unresolved_blocking` objections derived from the rounds rather
than copied from the reviewer's prose.

**`ordered_sibling`** — two different sentences that must never be merged:
`"none — no fully-ordered ICSD entry of this composition"` = *checked and absent*;
`"none (no sibling DB configured)"` = *not checked*. A sibling is only ever a ΔE comparison; the
representative is always DeMARS's own de-averaged structure.

**`representative.source`** — `engine-lowest` (the enumeration's own pick), `engine-repick` (a
different frame of the same ensemble, with `frame_index` and `pick_reason`), or `custom-build` (a
structure the analyst constructed, with `build_recipe`). `pick_unresolved` means a repick was
requested and could not be resolved, so the record fell back to the frame the analyst rejected.

`generation_recipe` carries what the build actually did, including `exclusion_merge` (the cutoffs, per
element) and per-orbit `group_orbits[*].exclusion.mode`. On that last one: `merged: false` means *not
clique-merged*, **not** "no exclusion" — `mode: pairwise` means the excluded pairs are enforced at
decoration.

## Run many structures

More than one structure is a **batch**: use the `mar-batch` skill. It partitions the list, logs every
result and every failure to an append-only ledger, continues past errors, and is resumable. Do not
improvise a loop — the ledger is what makes a long campaign auditable.

## Continuing maturation

DeMARS ships matured for its own scope, not finished. The judgment layer is **text you are meant to
edit**, and the apparatus that turns a failure into a permanent improvement is here rather than in a
maintainer's private notes. If you run it on chemistry it has not seen, expect to close that loop
yourself.

**What the campaign tells you.** `runs/_batch/batch.jsonl` records one row per entry, verdict
included. The **revise rate** is the number to watch: it says whether the *analyst* layer or the
*review* layer needs work. `demars gallery <rundir>` renders the same campaign as an audit page —
mechanism class, the five gate chips, the review verdict, any unresolved blocking objection, and
which entries are unreviewed. Read those before reading individual records.

**Keep the one-round rule.** An entry gets analyst → reviewer once; a `revise` verdict is logged and
the entry is done. Re-running the analyst to satisfy the reviewer optimises the record's prose
rather than the chemistry, and it destroys the only measurement you have — a revise rate you grind
to zero has stopped being a signal.

**Where a finding should land.** A reviewer objection is evidence, not yet an artifact. Convert it,
in this order of preference:

| if the decision is… | put it in | example in this tree |
|---|---|---|
| reducible to a rule | deterministic code | `mar_engine.ship_candidates()` |
| checkable after the fact | a gate | `gates.connectivity`, `gates.sqs` |
| a judgment the engine cannot make | the doctrine in `.claude/skills/` | `strategy.md`, `taxonomy.md` |
| a procedure a run must follow | `CLAUDE.md` / the batch skill | the one-round rule itself |

**Record the episode too, and record it first.** `episodes/` is where an experience is kept in the
form that can be applied to a new entry: keyed by the engine flag that should call it up
(`dummy_species`, `full_occ_clash`, `h_restore`, …), with `lesson` — what applies — beside
`shipped_as`, which says where the de-identified wording ended up in the tree. Write it there before
you write the doctrine, because the doctrine has to be **de-identified**: the skills and engine
comments name no source entries, so that an analyst meeting one in a corpus reasons about it instead
of recognising it. The entries a lesson was measured on live in its `provenance`, which exists to
audit that de-identification and not to apply the lesson. `episodes/README.md` has the contract.

The discipline is *name the chemistry, not the entry*, and it fails the moment it is left to memory
— an identifier once came back into prose that had already been scanned clean, and stayed until the
next scan. If the lesson does not survive losing the number, it is not a lesson yet.

**Then pin it.** Whatever you change, add a regression test, and give the failure a number in
`demars-core/docs/DEFECTS.md` — that file is where the numbering is defined, and the numbers are
permanent because a test uses them to say what it is guarding.
`demars-core/tests/test_backtest_regressions.py` is the worked format: a
`# ---- D<n>: <one-line statement of the failure>` header carrying the narrative, followed by the
assertions that make it impossible to reintroduce. A rule written only in prose is a rule that stops
being followed — several of the artifacts in this tree exist because that happened. Run
`pytest demars-core/tests tools/tests` before and after.

**Nothing is exempt from the gates.** A doctrine change that makes the analyst confident without
making it right will show up as a gate that stops firing, so watch the gate states across a campaign,
not just the verdicts.

## Known gaps

- **The convex hull needs a Materials Project key.** `demars_core.hull` ships installed — its client
  `mp-api` is a hard dependency, because a gate whose machinery may or may not be present makes
  "all gates clean" mean different things on different boxes. What it still needs is the key, in
  `$MP_API_KEY` or `materials_project.api_key`; it is the only DeMARS stage that reaches an external
  service. Without the key it reports `state: "not_run"` with the reason, which is UNCHECKED — so
  where no ordered sibling exists, an unconfigured install still has no stability reference.
- **The ordered-sibling search needs a local ICSD database** you provide. Build it with
  `assets/icsd_query/` — the indexer, CLI and install steps are there
  (`assets/icsd_query/README.md`); bring your own licensed `icsd_cif.zip`. DeMARS also calls that CLI
  (`icsd-query extract`) to pull input CIFs by id, so a corpus run wants it on `PATH`.
- **No ICSD data is redistributed here.** The bundled fixtures are COD (CC0).

## Documentation map

| file | for |
|---|---|
| `demars-core/README.md` | the engine: Python API, CLI, MLIP backends, troubleshooting |
| `tools/README.md` | the stage-①→⑥ file contracts the agents orchestrate |
| `CLAUDE.md` | the trigger, the defaults, the facts that bite |
| `version/README.md` | when protected code may change: the `frozen` / `mutable` modes, release tags, and the hooks in `.claude/settings.json` that check them |
| `assets/` | `demars.yaml.example`, `setup/` (env, config, doctor), `slurm/` (job templates), `icsd_query/` (the local ICSD indexer and CLI) |
| `reference/` | the stored answers `tools/demars_reference.py` checks a fresh install against |
| `.claude/skills/mar-analyst/` | the methodology: `taxonomy.md` (mechanism classes A–F), `strategy.md`, `principles.md`, `tools.md` — **edit these to change how the analyst judges** |
| `.claude/skills/mar-reviewer/` | what an adversarial review must attack — **edit this to change what it refuses to accept** |
| `.claude/skills/mar-batch/` | the batch protocol — partitioning, the append-only ledger, resuming after an interruption; **use it for more than one structure** |
| `episodes/` | what earlier entries taught it, keyed by the engine flag that calls each one up. `lesson` applies; `provenance` records the entries it was measured on, for auditing the de-identification |
| `demars-core/docs/DEFECTS.md` | what every `D<n>` cited in the code and skills means, and which five numbers are burned |
| `demars-core/tests/test_backtest_regressions.py`, `tools/tests/test_stage_contracts.py` | where the fixed defects are pinned: each `# ---- D<n>:` section states the failure it guards, and is the format to copy when you pin your own |

## License

**MIT** — see `LICENSE` at the repository root. It covers the whole distribution unit: the engine
(`demars-core/`), the stage tools (`tools/`), the judgment layer (`.claude/`) and the episodes
(`episodes/`).

Runtime dependencies are MIT/BSD, except ASE (LGPL-2.1-or-later), which imposes nothing on a
Python importer.

Redistributed third-party code:

- `demars-core/demars_core/viewer/js/3Dmol-min.js` — 3Dmol.js, **BSD-3-Clause** (© 2014 University
  of Pittsburgh and contributors), bundling GLmol, Three.js and jQuery. Notices:
  `demars-core/demars_core/viewer/PROVENANCE.txt`, license: `…/viewer/js/3Dmol-LICENSE.txt`.
- `demars-core/tests/fixtures/*.cif` — Crystallography Open Database, **CC0**.
