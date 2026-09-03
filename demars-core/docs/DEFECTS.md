# Known defects

DeMARS carries a numbered defect ledger. Comments and tests across the tree cite those numbers —
`D19`, `D32`, `D37` and others — and **this file is where the numbering is defined.** If you meet a
`D<n>` in a comment and want to know what it means, look here first.

## How the numbering works

- **Numbers are permanent.** A defect keeps its number after it is fixed; the number is how a
  regression test says what it is guarding against.
- **A number resolves here, whatever its state.** Where a regression test pins the defect, that test
  is the real record — it fails if the defect returns.
- **Five numbers are burned and must never be reused: D30 · D31 · D38 · D39 · D41.** They were
  allocated to findings later reclassified as procedure or environment rather than code defects.
  There was a period when older provisional citations were being converted into these numbers, so a
  reused number would silently resolve to the wrong thing.
- **D4 and D20 are unaccounted for.** They are in neither the table below nor the burned list,
  and no comment, test or episode in this repository cites either number. Do not reuse them
  until someone establishes what they were.
- Defect numbering is about *this program's* behaviour. Entries from the source database are
  referred to by chemistry, never by identifier — see `CLAUDE.md`, "name the chemistry, not the
  entry."

---

## What each number means

One line per number. Where a test pins it, that test is the authoritative description. `TBR` =
`demars-core/tests/test_backtest_regressions.py`, `TSC` = `tools/tests/test_stage_contracts.py`.

| # | What it was | Pinned by |
|---|---|---|
| D1 | per-atom energy comparison across samples of differing composition and occupancy | — |
| D2 | connectivity gate false positives — it invented units that were not there | TBR |
| D3 | model repair silently changed the sibling lookup key | — |
| D5 | the `couple_cut` basis string did not describe what was actually used | TBR |
| D6 | self-driving chased stoichiometrically unreachable targets | — |
| D7 | analyst output had no machine-readable layout contract; `record.json` is now the only basis for the MAR | — |
| D8 | the representative was picked from 13 % of the requested ensemble and recorded as success | — |
| D9 | `representative.composition` on engine-lowest records held a config label, not a composition | — |
| D10 | a failed final tier reported `final_MAR: {}`, indistinguishable from "not requested" — the autobatcher's own calibration could yield a budget smaller than one system, and the resulting `ValueError` was laundered into `val=False` | — |
| D11 | `engine.json` recorded the sibling search as both run and not run | TSC:228 — value and provenance must move together |
| D12 | `COORD_FORMERS` was a fixed element list, so anions of any former outside it read as dangling | — |
| D13 | the atom budget shrank the cell silently while the record kept the *requested* value | — |
| D14 | long relaxations were all-or-nothing — an interruption discarded completed work | — |
| D15 | rigid-unit pre-relaxation cost more than the free relaxation it protected | — |
| D16 | custom builds were not held to the file contract | — |
| D17 | the fused-kernel accelerator changed numbers and the record did not say so | `test_accelerator_provenance.py` |
| D18 | the skill told subagents to background a relaxation and not what to do next | — |
| D19 | the only way to ship a frame other than the engine's lowest was to call it a custom build | TBR:938 |
| D21 | recording the review deleted what the review asked for | TBR:751 |
| D22 | nano was not released on tier transition → final-tier OOM | — |
| D23 | a requested final tier that failed returned `{}`, indistinguishable from "not requested" | TSC:342 |
| D24 | the memory probe OOM'd building its own trial batch and killed the run | — |
| D25 | the connectivity gate ignored cations outside `COORD_FORMERS` | — |
| D26 | the charge gate could not correct an integer-declared species from the CIF | — |
| D27 | `dist_stats` had no `n==1` guard and produced an SRO reading from a single point | — |
| D28 | the gate over-flagged, so analysts declared it inapplicable where it applied and was clean | TBR:858 |
| D29 | the record could not prove its own construction | TBR:1204 |
| D32 | charge balancing erased the disorder and the one place a gate looks did not see it | TBR:1057 |
| D33 | the sibling search filters on exact chemsys, so an ordered prototype in a narrower chemsys is unreachable | — |
| D34 | the bracket probes were offered to take-lowest, so a probe could ship as the representative of a solid solution | `test_ship_candidates.py` |
| D35 | (with D29) the record could not prove its own construction | TBR:1204 |
| D36 | the `--from` path crashed on the marker the final tier returns | TSC:336 |
| D37 | the `COORD_FORMERS` gap bit the *build*, and no cutoff value could reach it | TBR:1142 |
| D40 | `provenance.spacegroup` is empty on the repaired-CIF path | — |
| D42 | the same-element cut inspected only pairs under a 1.3 Å window, so a split pair above it left no cut set — **half closed** | TBR:224 |
| D43 | closing D42 through a covalent radius merged fully-occupied metallic sublattices, short-filling every decoration | TBR:1278 |

**D42 is only half closed.** The impossibility split reaches a pair under `0.9 * 2 r_cov`, wherever the widest gap falls. A pair above BOTH that and the 1.3 Å window is still invisible to distance alone — a face-sharing cation pair at ~2.9 Å reads exactly like an isolated one, and one such case needed `--same-excl` passed by hand. That residue is reported rather than solved: it lands in `out_of_window`, which is NOT confident, and the chemistry is left to the analyst. TBR:224 pins the reporting, not a fix.

**D43 is the cost of closing the half of D42 that could be closed, and the pair should be read together.** The impossibility split D42 needed reads chemistry through `0.9 * 2 r_cov` — a COVALENT single-bond radius. Metallic nearest-neighbour spacing at CN 8–12 sits below that for much of the transition-metal and rare-earth block: nine elements fail on the pure metal alone, and 38 of 65 sit within 0.3 Å of the threshold, a margin ordinary compound compression erases. On a fully-occupied *mixed* sublattice the split fired, union-find swallowed the sublattice into one exclusion network, and every decoration on every seed short-filled. The guard is gated on the split having actually fired, so distance still selects the candidates and occupancy then vetoes them — an alternate needs a vacancy to alternate into. A pair the split never reached is untouched and stays in `out_of_window`. The engine was already computing that occupancy sum for `suspicious_merges`, whose own note said no code branched on it. `--same-excl` remains the escape hatch and still outranks the guard. Measured on a metal-rich corpus that is held out, so the entries are not named here; the pinned fixture is a synthetic idealisation of the same shape.

**D32 is only half closed.** The blind spot it closed for the adjusted branch remains one
branch over: a rollup read `n_unconfident` as 0 because the values were CONFIDENT, so the gate saw
nothing. See TBR:1129.

Provenance, where it is recorded: **D22 · D23 · D25 · D27** — a **hydrous Fe/Mn phosphate**
(tier-memory release, the requested-vs-unrequested final tier, a former outside the coordination
tables leaving 16 of 62 cations ungated, and the single-point SRO reading). **D24 · D26 · D27** — a
**Mo-substituted lithium lanthanum zirconate garnet** (the memory probe OOM'ing on its own trial
batch, and an integer-declared species the charge gate could not correct). **D32** — a **lithium
rhodium oxide**. **D37** — a **gallium telluride**.

> **Descriptions for D28 · D29 · D32 · D35 · D36 · D37 were reconstructed from their regression
> tests.** The original prose lived in a working note that is not part of this repository. Where a
> row above says `—`, no test pins that defect by number and the description comes from the historical
> ledger; treat it as a label, not a specification.
