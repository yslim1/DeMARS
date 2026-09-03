# tools/ — the judgment layer's instrument shelf

Thin CLI wrappers over the installed `demars_core` package, giving the `mar-analyst` /
`mar-reviewer` agents (`.claude/`) the file-based stage contracts they orchestrate.
Every one of them is keyed by a structure **file path**, never an ICSD id.

    cd <repo root> && tools/py tools/<tool>.py …

`tools/py` runs the tool under the interpreter named by `paths.python` in `demars.yaml`
(`$DEMARS_PY` overrides it) — `demars_core` is installed editable from `./demars-core` into that
env. An interactive shell with the env already activated can call `python` directly; an agent
cannot, because its shell is re-initialized from the profile on every call and a profile that
initializes conda restores its own default env, leaving a bare `python` without `demars_core`.

| tool | stage |
|---|---|
| `demars_evidence.py <cif> [--summary] [--out DIR] [--icsd-id N]` | ①a evidence bundle (incl. ordered-sibling search) |
| `demars_engine.py <cif> --out DIR [flags] [--from S]` | ③ enumeration engine |
| `demars_connectivity.py <struct> [--json]`            | ⑤ polyanion gate |
| `demars_hull.py <representative_final> [--calculator omni] [--out DIR]` | ④b convex hull → `gates.hull` (needs an MP key). Writes `hull.json` **and** the tier-relaxed `representative_final.*`, so `--out` belongs in `_work/<tag>/` |
| `demars_record.py <cif> --engine E --judgment J --hull H [--connectivity C] --out DIR` | ⑥ mar-1.0 assembler (connectivity + SQS run themselves; **hull does not**) |
| `demars_reference.py [name] [--generate]` | self-check against the bundled COD references |

`demars_engine.py` writes `engine.json` in the shape `demars_record.py` consumes, and renames
`deaverage()`'s own deterministic output to `deaverage_output.json` so it cannot be confused with
stage ⑥'s mar-1.0 `record.json` (and fixes up `ensemble_files.record`, which recorded the pre-rename
name and so dangled in every CLI run). Three files, three distinct names, no shared stem:
`engine.json` = the stage-③ contract · `deaverage_output.json` = the engine's own record ·
`record.json` = stage ⑥.

**Run every engine into `<rundir>/_work/<tag>/`, never the run dir**, because `--out` is where
`representative*` land: a refinement rerun plus an ordered-sibling relaxation otherwise leaves several
rival `representative_final.cif` side by side (one set-B entry had three, one of them the sibling that must
never be adopted and 12× smaller than the real MAR). **A run is adopted by passing its `engine.json`
to `demars_record.py` — nothing is copied or renamed.** `record.json`'s `representative.file` comes
from the engine's own `ensemble_files`, so the record is the single code-proven authority on which
structure is the MAR; never resolve that by globbing.

**A repicked frame needs its own final tier.** The engine relaxes the enumeration's *lowest* at the
final tier, so a `representative_pick` ships a structure that has only the nano energy — every other
path (take-lowest, a custom build via `--from`) carries a final one. Relax the picked frame with
`--from <frame> --final` into its own `_work` tag and name that run in
`judgment.representative_pick.final_from`; stage ⑥ reads back `E_final` and `representative_final`
from it, refuses it if it finalized a different frame, and warns when it is missing. The ensemble and
the gate basis still come from the one `--engine` adopted, so D19(b) stays fixed.

Those paths are stored **relative to the record** when the file sits under the run directory (absolute
when it comes from a corpus elsewhere), so the whole entry directory can be moved or renamed and stays
internally consistent. Stage ⑥ also re-roots a stale path: if the recorded one no longer exists it
looks for the same tail under the record's own directory, which repairs an archive that was moved
after the fact — re-run ⑥ (cheap, no relaxation) and the paths and `gates.connectivity` come back.

All of them are thin: argparse + file I/O + printing, with the logic in `demars_core`. That
includes the ⑤ gate's audit — the one check that catches what charge and fidelity cannot lives in
`demars_core.connectivity.audit_frame`, not here, so a standalone install has it and the package's
own tests reach it without a `sys.path` detour.

`ordered_lookup.py` is not a CLI: it backs `mar_evidence`'s `_sib` bootstrap (exec'd via the
`DEMARS_ORDERED_LOOKUP` env var, which `demars_evidence.py` sets before importing the package).
Its paths are driven by `ICSD_DB_DIR` (set from
`paths.icsd_db` in `demars.yaml`; there is no default -- unset raises rather than guessing, and
the record then says the search did not run). Fetch a sibling's CIF for the ΔE
comparison with `icsd-query extract <id> -o <dir>`, then `demars_engine.py --from`.

**Browse the results:** `demars gallery <rundir> --serve` (or
`tools/py -m demars_core.cli gallery <rundir>`) renders every `record.json` it finds — one card per
entry with the mechanism class, the five gate chips, the review verdict and `E_above_hull`, and a
detail page carrying the full record and the 3-D ensemble. Read-only: it writes to
`<rundir>/_gallery` and never touches a record. The chips keep the four-state vocabulary, so
`not_run` / `vacuous` / a gate the record does not carry never look like a pass.

**Tests:** `tools/py -m pytest tools/tests` (23 contract tests, no MLIP/GPU, ~8 s) — or
`tools/py -m pytest` from the repo root, which collects these plus `demars-core`'s suite. They pin the seams that live *here*
rather than in the package: `engine.json`'s shape and its round trip into `build_record`, the
`record.json` → `deaverage_output.json` rename and the path fixup, the `mode` tag vocabulary, and
the `built` 5-tuple `--from` hand-assembles. See the INTERNAL CONTRACT note in
`demars-core/demars_core/api.py` for the full list of private names this directory depends on.

**Slaved-pair and orientation-unit detection** are `mar_evidence.slaved_units` /
`orientation_units` — neutral facts in the stage-①a bundle under `disorder`. They are what computes
class **E — Slaved** of `taxonomy.md`; without them `cross_orbit_contacts` gives distances and the
analyst has to draw the pairing. Detection only — building is `mar_engine`'s job.
`orientation_units` is the weaker of the two: its criterion needs a full-occupancy anchor of the
same element, disjoint coverage and ≥2 alternatives per anchor, and it judges position by position
without first merging same-element alternates into one group, so it under-reports rather than over-reports.

**H reconstruction.** The engine's H-restore keeps located H and adds only the formula's deficit.
Placement is **clash-checked with retry**, over an acceptor list beyond O/N (`F, Cl, S, Se`, with
their X–H lengths), and a blocked host re-offers its protons instead of costing the structure the H
the formula demands. There is **no standalone unlocated-H allocator** in this package (`mar_h_alloc`,
`mar_h_topup`): H restoration exists only inside the engine's build, off the formula deficit.

Acceptor ranking has a **fallback**: a continuous Pauling bond-strength
deficit (`|q̄_X| − Σ_M q̄_M/CN_M`), reached only where the capacity model finds nothing placeable and
no orientational risk was flagged. It matters because the capacity model counts an ionic Ca–O contact
as a full covalent bond: on the shared oxygen of a Si tetrahedron and a Ca octahedron it reports
capacity 0 while the bond-strength deficit is 0.87. Without the fallback, hydrous silicates and
phosphates come back `deficit N not reconciled by safe acceptors -> custom` with the protons
unrestored. It declines outright when the CIF carries no oxidation
states — that model has no other input, and inventing one would be worse than saying so.

**The convex hull** is `demars_core.hull` + `tools/demars_hull.py` (stage ④b). Its key is the
**file**, and the tier is explicit
(`--calculator omni` = the DeMARS final tier) because an `E_above_hull` computed at a different
level than the structure it describes is not a stability number. Default `--mode self-consistent`
re-relaxes MP's reference structures at that same tier so the MLIP-vs-DFT offset cancels rather than
loading onto the MAR; `--mode mp-direct` takes MP's own energies instead (omni-mpa only, ~2x cheaper).
Needs an MP key; without one it reports `state: "not_run"` with the reason — UNCHECKED, never
"on the hull". The energy scale is stamped in `corrections`, default `mp2020` (MP's own hull scale; `none` is raw PBE(+U), internally consistent
but not the hull MP publishes).

It **is** `gates.hull`, the fifth gate — which **reports rather than judges** unless a threshold is
configured (`hull.tol_eV_per_atom` in `demars.yaml`, or `--hull-tol`): unset, `pass` is `null` and
you read `E_above_hull` yourself. It is also the only gate that does
**not** re-run itself in stage ⑥,
because it needs a network call and a relaxation. Pass `--hull <rundir>/hull.json` every time;
without it the gate is `not_run`, which is not a pass. `mp-api` is therefore a hard dependency
(a gate whose machinery may or may not be installed makes "all gates clean" mean different things on
different boxes); the *key* stays deployment config, and its absence is reported, not raised.

**The Antypov O/S/V/P orbit descriptor** is `demars_core.disorder_class`. Its key is the **file**,
and stage ⑥ runs it itself off `evidence['source']` — so `disorder_descriptor` carries a real
`disorder_set` / `multiset` / `no_full_backbone`, which is the structural half of the
`no_full_backbone` triage row in `strategy.md`. A record built from an evidence bundle with no
`source` still reports `{"available": false, "reason": ...}` — **not `null`**, so a reader can tell
"could not be computed here" from "computed, came back empty". Run it by hand with
`tools/py -m demars_core.disorder_class <structure.cif>`.
