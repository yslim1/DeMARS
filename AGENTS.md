# DeMARS

De-average disordered crystals into Minimal Atomistic Representations (MARs).

- `demars-core/` — the deterministic engine, installed editable as `demars_core`. **Never guess the
  env**: `paths.python` in `demars.yaml` is the only place it is written down, and `tools/py` reads it.
- `tools/` — file-based CLI wrappers over it (the stage ①→⑥ contracts). See `tools/README.md`.
- `.agents/skills/` and `.codex/agents/` — the LLM judgment layer: `mar-analyst` (builds the MAR)
  and `mar-reviewer` (adversarial check). Code proves; the analyst judges.
- `episodes/` — what earlier entries taught the program, keyed by the engine flag that calls each
  one up. `lesson` is what applies; `provenance` is the audit trail. See `episodes/README.md`.
- `demars-core/docs/DEFECTS.md` — what every `D<n>` cited in the code and tests means.
- `version/` — when protected code may change (`state.json.mode`), enforced by the hooks in
  `.codex/hooks.json`. See `version/README.md`.
- `assets/` — `demars.yaml.example`, setup scripts, SLURM templates, the local ICSD indexer.
- `reference/` — stored answers for `tools/demars_reference.py`'s install self-check.

## Version-controlled changes

Before any maturation request that may lead to changes in protected code, **first read
`version/README.md` fully and follow it**. The version layer controls when production code may
change; it does not prescribe how memories are accumulated, reviewed, or selected for maturation.

- Observation, memory collection, and analysis may continue while the version is `frozen`. Those
  activities alone do not authorize a mode change.
- Before editing a protected file, confirm that the human has authorized the change. If the current
  mode is `frozen`, change only `version/state.json.mode` to `mutable`, retain the existing
  `release_tag` as the baseline, and commit that state-only transition with its reason before making
  implementation changes.
- While `mutable`, production Analyst and Reviewer runs are blocked. Use a rehearsal only when its
  prompt contains the exact line `VERSION_MODE: rehearsal`.
- Do not create, replace, or move a release tag without explicit human instruction. Once the release
  scope is known, use the promotion procedure in `version/README.md` to freeze the next version.

## Trigger: "deaverage <file>"

When the user says **"deaverage <file>"** — or any equivalent ("de-average X.cif", "run the MAR
pipeline on X", "build a MAR for X", "what's the disorder in X") — run the full pipeline without
asking follow-up questions. Fill in the defaults below and go.

1. **Spawn the project custom agent `mar-analyst`** on that file, in the background, with:
   - out dir `runs/<cif-stem>/` (create it; `<cif-stem>` = filename without extension)
   - production settings: `--nr 30 --min-nm 1.5 --final`
   - `--icsd-id N` **if** the filename or the user identifies it as an ICSD entry
     (e.g. `icsd_123456.cif` → `--icsd-id 123456`)
   - instruction: if `ordered_sibling_ids` is non-empty, fetch the sibling with
     `icsd-query extract` and report the ΔE on the omni-mpa tier
   - instruction: **run stage ④b** (`tools/py tools/demars_hull.py <shipped representative_final>
     --calculator omni --out <rundir>/_work/<tag>`) once the representative is final — `--out` goes
     in `_work/<tag>/`, not the run dir, because this stage also writes a tier-relaxed
     `representative_final.*`. `gates.hull` is the one gate that cannot run itself, so skipping this
     leaves it `not_run` = UNCHECKED. It **reports and does not judge** unless
     `hull.tol_eV_per_atom` is set, so quote `E_above_hull` and interpret it; never call a
     `pass: null` hull gate clean. If the artifact comes back `not_run` (no MP key), say that.
2. **Then spawn a fresh project custom agent `mar-reviewer`** on the finished
   `runs/<cif-stem>/record.json`.
   Fresh context — never pass it the analyst's reasoning. Save its output verbatim to
   `runs/<cif-stem>/review.json`.
3. **ONE round. Report the verdict — do NOT re-run the analyst.** analyst → reviewer, and stop.
   No round 2, no round 3. A second analyst launched to answer the review optimises the record's
   prose to satisfy the reviewer rather than the chemistry, and a surviving objection is the
   review-layer equivalent of the engine's *"residual spread is genuine SRO"* — a fact to report,
   not a defect to grind away.
   - **`confirm`, no blocking objection** → done.
   - **`revise`** → **done too, and reported as `revise`** with the reviewer's `objections` listed
     verbatim. The objections are the deliverable; whether to rebuild is the user's call, and they
     can ask for another pass. Do not launch a second analyst on your own.
   - **`escalate`** → stop and say so. It needs literature or a human.
   In every case the record keeps the review (step 4) and the report names the verdict.
4. **Stamp the review into the record.** Pass step 2's `runs/<cif-stem>/review.json` (the
   reviewer's own object — no array, there is one round) and re-run stage ⑥ with `--review`:
   ```bash
   tools/py tools/demars_record.py <cif> --engine <rundir>/_work/<tag>/engine.json \
            --judgment <rundir>/judgment.json --review <rundir>/review.json \
            --hull <rundir>/_work/<tag>/hull.json --out <rundir>
   ```
   `<cif>` is **the file the MAR was actually built from**: if the analyst repaired the input, pass
   the repaired CIF, not the original — the charge gate is recomputed here from that file's own
   oxidation labels, so the wrong source flips a passing gate to `pass: false`.
   Cheap — no relaxation. **Skipping it leaves `"review": null` in the record, which means
   UNREVIEWED**, and a null review must never be reported as confirmed. Re-running ⑥ also repairs
   structure paths after a run directory has been moved: they are stored relative to the record, and
   a stale one is re-rooted under it.
   `gates.connectivity` and `gates.sqs` **re-run themselves** here, so nothing has to be handed in
   again for them. Add `--connectivity <rundir>/connectivity.json` only when the stage-⑤ artifact
   carries what the auto-run cannot (`--expect`, or a multi-frame `ensemble.xyz` audit).
   **`--hull` is different: `gates.hull` cannot re-run itself** (it needs a Materials Project call
   and a relaxation), so pass step 1's `hull.json` every time — without it the gate is `not_run`.
5. **Report back**: mechanism class + one-line interpretation, the five gates
   (charge / fidelity / connectivity / sqs / hull — connectivity and sqs are automatic, **hull is
   not**; report every `state`, and never read `vacuous` or `not_run` as a pass), the
   `E_above_hull` **with your reading of it** when the hull gate is `derived` (that gate leaves
   `pass: null` unless a threshold is configured — a number, not a verdict), sibling ΔE if any, and
   the reviewer's verdict — **quoting its `objections` verbatim when the verdict is `revise` or
   `escalate`**, since that is where the loop used to go and the user decides instead. Plus anything
   the analyst escalated. Give the run dir path.

Deviate from the defaults only when the user states a different budget, output location, or
calculator. If they say "screening" or "quick", drop to `--nr 12 --min-nm 1.0` and tell the
analyst to record that in `decision_trace` and cap `confidence`.

**More than one structure** — a directory ("deaverage the structures in `A/`"), a glob, or several
named files — is a batch: **invoke the `$mar-batch` skill** and follow it. It partitions the list and
may run entries in parallel (each entry standalone; analyst → reviewer stays ordered within one),
logs every result and failure as it goes, and continues past errors. Do not improvise a batch loop
without it.

## Running things yourself

Only when the user asks for a single stage rather than the pipeline:

Run everything from the repo root through `tools/py` — **never a bare `python`, never
`conda activate`**: the Bash tool re-initializes the shell from the profile on every call, which
restores whatever env that profile sets up, so a bare `python` has no `demars_core` and an
activation of your own is undone by the next call.

```bash
cd "${DEMARS_ROOT:-$PWD}"          # tools/py = paths.python from demars.yaml ($DEMARS_PY overrides)
tools/py tools/demars_evidence.py <cif> --summary --out <rundir>     # ①a evidence
tools/py tools/demars_engine.py   <cif> --out <rundir>/_work/r0 --final  # ③ engine
tools/py tools/demars_connectivity.py <struct> [--json] [--expect S=4]  # ⑤ gate, by hand
tools/py tools/demars_hull.py <rep_final> --out <rundir>/_work/<tag>  # ④b hull (needs an MP key)
tools/py tools/demars_record.py   <cif> --engine <rundir>/_work/r0/engine.json \
         --judgment <rundir>/judgment.json --out <rundir>   # ⑥ record (⑤ + SQS run themselves)
icsd-query show|extract|chemsys|formula ...                          # local ICSD (user-provided)
```

## Facts that bite

- **The ICSD sibling search IS wired up** (`tools/ordered_lookup.py` + `ICSD_DB_DIR`). But
  `"none — no fully-ordered ICSD entry of this composition"` means *checked and absent*, while
  `"none (no sibling DB configured)"` means *unchecked*. Never report either as "this phase has
  no ordered form".
- **Never adopt an ordered sibling as the MAR.** It is a ΔE comparison only; the representative is
  always our own de-averaged structure.
- **A new lesson goes into `episodes/` FIRST, and into the prose only de-identified.** The skills and
  engine comments carry no ICSD ids by design: an id plus its answer in a file the analyst reads
  turns inference into lookup and silently contaminates the corpus. So **name the chemistry, not the
  entry** — the lesson has to hold without the number, and if it doesn't, it isn't a lesson yet.
  **This rule has been broken by writing up a fix**, more than once; the last time, an identifier
  came *back* into prose that had already been swept clean, and nobody was watching. Schema and the
  full contract are in `episodes/README.md`; defect measurements go to `DEFECTS.md`, not here.
  **The analyst never writes `episodes.json` — it proposes in `escalate` and stops**; an episode added
  mid-run re-bases the doctrine partway through the batch.
- `record.json` is the stage-⑥ mar-1.0 record (judgment + recomputed gates) and nothing else.
  `deaverage()`'s own deterministic output is `deaverage_output.json`.
- Anything importing `demars_core` must `import _icsd_env` **first**, or the sibling search
  silently degrades.
