---
name: mar-batch
description: >
  Run the DeMARS de-averaging pipeline over MANY structures — a directory, a glob, or several
  named files. Each structure is processed standalone — partitioned across workers, optionally in
  parallel — logging every result and every failure to an append-only ledger, continuing past
  errors, and resumable after an interruption. Use whenever the user points the pipeline at more
  than one structure.
---

# MAR batch runner

You orchestrate the per-structure cycle over a set of structures. The cycle itself is unchanged —
`mar-analyst` builds the MAR, `mar-reviewer` checks it — once each, no revise loop. Your job is the
**ledger discipline**:
sequential execution, an honest ledger, and continuing past failures.

## Three non-negotiables

**SEQUENTIAL BY DEFAULT. Parallel only on measured headroom — and measure the CPU, not the GPU.**
Entries are standalone (next rule), so nothing about *correctness* requires running them one at a time.
But parallelism buys nothing unless there are cores to run it on, and on a pinned single core it is
strictly worse than serial. Two things hold whichever way you run:

- **Within one entry, analyst → reviewer stays strictly ordered.** The reviewer reviews a finished record.
- **If you do go parallel, split the file list into per-worker slices before you start.** Never let
  workers pick from a shared queue by consulting the ledger — two of them will claim the same file.
  Partition, then never re-check.

**Before starting, run this. If it says 1, run one entry at a time:**
```bash
nproc; taskset -cp $$; nvidia-smi --query-gpu=name,memory.free --format=csv
```
`nproc` is what you actually get, not what `/proc/cpuinfo` advertises — the shell can be pinned to a
single core of a 40-core box, and **every process you launch inherits that affinity**. `taskset -c 0-39`
does *not* widen it: it returns success and changes nothing (`sched_getaffinity` still reports 1).

**Measured 2026-08-12, set B, `n007.hpc` (RTX 3090, 24 GB, `nproc=1`):** three entries at once ran at
21–25 % CPU each with loadavg ≈ 5 — one core split three ways, so no throughput was gained and each
entry took 2–4× its pilot wall-clock. Worse, the concurrency caused the failures: **2 of 3 entries hit
CUDA OOM in the `--final` omni-mpa step**, and an empty `final_MAR` is indistinguishable in the record
from "the final tier was never requested". One 360-atom entry was still relaxing after
**68 minutes** and never finished. GPU headroom was never the constraint — 14 GB stayed free throughout.

So: the old "the cap is GPU, not integrity" reading was wrong on this node. Relaxation does auto-size
its batch to *free* VRAM and back off on OOM rather than crashing, but that back-off is what turns a
contended run into an hours-long one, and the OOM retry can still exhaust its 3 attempts. Go parallel
(3–4 entries) only when `nproc` genuinely reports several cores **and** the GPU is idle; confirm the
card is an **A5000/3090-class device, not the TITAN Xp** — that card falls back to CPU and parallelism
there buys nothing at all.

The ledger is safe to write concurrently: `batch_log.py` appends one line per call with `open(..., 'a')`,
and the reader already tolerates a torn line.

⚠️ **Redefine the failure circuit-breaker if you run parallel.** "Stop after 3 consecutive failures"
is meaningless across interleaved workers — use *3 failures within one worker's slice*, or a total count.

**EVERY STRUCTURE IS STANDALONE. You are the leak.**
The analyst and reviewer each run in a fresh, isolated context, so they cannot see each other or the
previous entry — **but you can, and you write their launch prompts.** By entry 50 you are holding 49
finished answers, and one helpful clause ("another A-site disordered perovskite like the last few")
contaminates an agent that is otherwise perfectly isolated.

So the launch prompt is **fixed**: the file path, the out dir, and the `CLAUDE.md` defaults — nothing
else. No mention of any previous entry, its class, its verdict, or a pattern you think you are seeing
across the batch. Do not summarise the run so far *to* an agent. If you notice a trend, it goes in
your final report to the user, never into a prompt.

Why this is not fussiness: the campaign reports a **distribution of mechanisms**. Entries nudged by
what their neighbours were classified as make that distribution self-confirming — it would corroborate
the very result the campaign exists to measure. (For the same reason, never set `memory:` on the
analyst or reviewer subagent: it would carry learning across entries by design.)

**LOG EVERY STRUCTURE THE MOMENT IT FINISHES.** Success or failure, one `batch_log.py` call each,
written before you start the next file. Never accumulate results and write them at the end: a batch
that dies at file 15 of 40 must still leave a truthful record of the 14 that completed.

## Procedure

### 1. Enumerate and announce
List the `.cif` files (depth 1 unless told otherwise), sort them, and **state the count before
starting**. Each structure gets `<root>/<cif-stem>/` and the ledger lives in `<root>/_batch/`, where
`<root>` is the out dir the caller named — **`runs/` unless told otherwise**. Substitute it into every
command below; never split the entries and the ledger across two roots.

If the set is large enough that the user should know the cost up front (say >10 structures at
production settings), say so and give a rough wall-clock before you begin.

### 2. Resume check
`python tools/…` below means the demars env's interpreter — run it as
`cd "${DEMARS_ROOT:-$PWD}" && tools/py tools/…`; a bare `python` is the profile's and has no
`demars_core`.
```bash
python tools/batch_log.py <root>/_batch --done
```
Skip any file it prints — those are FINISHED (last row `ok`, `declined` or `skipped`) — unless the user said to
redo them. Say how many you're skipping. An entry whose last row is `error` is deliberately NOT in
that list: it has no record, so a resume runs it again. So is an entry with a run directory but no
ledger row at all, which is what a session that died mid-entry leaves behind.

### 3. Per structure, in order
Announce progress as you go — `[7/23] <filename>.cif`.

Run the normal cycle (analyst → reviewer, **one round — no revise loop**, defaults from
`CLAUDE.md`), and log **immediately after the review**, one row per entry (`--round 1`):

```bash
python tools/batch_log.py <root>/_batch --file <cif> --status ok --rundir <root>/<stem> \
       --class "<class>" --confidence <high|medium|[screening]> \
       --gates "charge=pass,fidelity=pass,connectivity=clean,sqs=pass,hull=derived" \
       --verdict <confirm|revise|escalate> --round <N> [--escalate "<kind>: <one clause>"]
```

**`--escalate` is one line, and it introduces NO new fact.** It is an index entry — an
`engine:`/`tool:`/`gate:`/`chem:` prefix plus one clause, restating something that already stands in
the entry's own `record.escalate` or `review.objections`. A finding that exists only in the ledger
has no provenance and nobody re-checks it, so put it in the record (or `episodes/`) first and cite it
here second. Reviewer minor objections are NOT escalations: they are already verbatim in
`review.json`, and `verdict` is what carries them.

**Record all five gate states, `hull` included.** You do not run any gate — the entry's own cycle
does — but `hull` is the only one that cannot run itself, so an entry whose stage ④b was skipped
comes back `hull=not_run`, which is UNCHECKED and never a pass. If that state never reaches the row,
the campaign summary cannot show that it happened to every entry.

A `revise` verdict is logged as `revise` and the entry is **done** — the analyst is not re-run
(`CLAUDE.md` §3). That keeps the revise rate measurable across the campaign, which is the number
that says whether the analyst layer or the review layer needs work. `--round` stays in the schema so
a ledger from a re-run pass (one the user asked for) can still be told apart.

On failure at any stage:

```bash
python tools/batch_log.py <root>/_batch --file <cif> --status error \
       --stage <evidence|engine|connectivity|record|review> --error "<real error text>"
```

When the analyst reaches a CONCLUSION that no reliable MAR can be built — the evidence does not
determine the structure, and no setting fixes that — the row is `declined`, not `error`:

```bash
python tools/batch_log.py <root>/_batch --file <cif> --status declined \
       --stage <stage> --error "no reliable MAR: <what a human or the literature has to supply>"
```

### 4. Finish
```bash
python tools/batch_log.py <root>/_batch --summary
```
Then state plainly: how many succeeded, how many failed, which need a rerun, and which were
escalated. **Report the failures out loud.** A batch with failures must never be summarized as
though it completed.

## Error discipline

- **An error NEVER aborts the batch.** Log it and move to the next file.
- **Capture the real error text**, not a paraphrase — that string is what makes the rerun
  diagnosable. Include the failing stage.
- **Retry at most once**, and only for something transient (a transport/OOM blip). A genuine
  chemistry or gate failure is a *result*, not a retry candidate.
- **Never quietly weaken settings to force a pass.** Dropping `--nr` or the cell to make a
  stubborn structure succeed, without saying so, turns a failure into a bad record. If you do
  reduce anything, it goes in `decision_trace` AND the ledger note, and `confidence` drops.
- **A failing gate is not a batch error.** `--status ok` with `gates=charge=FAIL` is the correct
  row: the pipeline ran, the structure has a problem. Reserve `--status error` for a stage that
  could not complete.
- **A refusal is not a batch error either.** `error` means something broke and a retry might work;
  `declined` means the pipeline ran to a conclusion and the conclusion is that this evidence cannot
  produce a reliable MAR. `--done` counts `declined` as FINISHED, so a resume leaves it alone —
  which is the point: re-running reproduces the same refusal. One entry logged as `error` sent a
  driver's retry loop back into it for hours and the loop could not terminate. Judge by whether
  another pass could plausibly differ, not by whether a structure came out.
- If the *same* failure hits several structures in a row (3+), stop and tell the user — that is an
  environment problem, not a per-structure one, and burning through the rest wastes hours.

## The ledger

`<root>/_batch/batch.jsonl` (machine-readable, one row per structure) and `<root>/_batch/batch.log`
(human-readable). Fields: `ts, file, status, stage, rundir, klass, confidence, gates, verdict,
escalate, error`. `--summary` prints a table plus a rerun list; `--done` prints the finished files
(last row `ok`, `declined` or `skipped`; an `error` entry is left out so a resume retries it).
