# Slurm job scripts

Copy one, edit the marked lines, submit **from the repo root** — every path inside is relative to
it, and `tools/py` finds the interpreter through `./demars.yaml`.

```bash
sbatch assets/slurm/single.j path/to/structure.cif   # one structure
sbatch assets/slurm/batch.j                          # a directory (edit the two paths inside)
```

| | `single.j` | `batch.j` |
|---|---|---|
| output | `runs/<cif-stem>/` | `runs/<name>/<stem>/`, ledger in `runs/<name>/_batch/` |
| resume | resubmit, it redoes the entry | resubmit, the ledger says what to skip |

## Edit before submitting

**`--partition=CHANGE_ME`** — cluster-specific, so both scripts ship unusable on purpose: sbatch
rejects an unknown partition rather than quietly running somewhere wrong. `sinfo -s` lists yours.

Pick one whose GPU your torch build has kernels for. This is the one that bites: an unsupported
arch **falls back to CPU silently** — the run still finishes, many times slower, and nothing in the
record says so. Check both sides:

```bash
sinfo -p <partition> -o '%N %G'                               # e.g. gpu:pro6000:8  (sm_120)
python -c "import torch; print(torch.cuda.get_arch_list())"   # must contain that arch
```

**`--time`** — budget from a pilot. Production settings (`--nr 30 --min-nm 1.5`) have run
~25 min/entry on a mid-size cell; a 1000-atom cell is several times that.

**`TARGET_DIR` / `ROOT`** (batch only) — one ledger per root. Never split entries and ledger
across two roots.

## While a batch runs

Leave `episodes/episodes.json` and the three `.claude/skills/*/SKILL.md` (plus `taxonomy.md`)
alone. The analyst and reviewer read them on *every* entry, so editing one mid-run re-bases the
taxonomy partway through and the mechanism distribution stops being comparable across entries.
A queued job inherits whatever is on disk when it *starts*, not when you submitted it.
