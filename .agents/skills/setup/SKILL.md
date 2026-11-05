---
name: setup
description: >
  First-run setup for a DeMARS checkout — create the interpreter environment and install
  demars-core, generate demars.yaml with the interpreter path that env actually produced, lay out
  the optional local ICSD index from the user's own licensed zip, and verify the deployment.
  Use when the user says "setup", "install", "set up DeMARS", "new machine", asks why `tools/py`
  or the sibling search does not find anything, or when a fresh clone has no `demars.yaml`.
---

# DeMARS setup

You turn a clone into a working deployment. Four scripts in `assets/setup/` do the deterministic
work; your job is to **ask the questions the scripts cannot answer, run them in order, and report
what the resulting deployment can and cannot check**. Never edit `demars.yaml` by hand when
`write_config.py` can write it — the file's comments are the documentation, and the script keeps
them.

| step | script | what it needs from the user |
|---|---|---|
| 1 env | `assets/setup/make_env.sh` | conda or venv; env name; GPU → which CUDA torch |
| 2 config | `assets/setup/write_config.py` | SevenNet-nano checkpoint dir; MP key (optional) |
| 3 ICSD | `assets/setup/build_icsd.sh` | path to **their** `icsd_cif.zip`, or "don't have one" |
| 4 verify | `assets/setup/doctor.sh` | nothing |

## Rules that hold throughout

- **Never guess the environment.** `paths.python` is the one place the env is written down, and
  `tools/py` reads it. It comes from make_env.sh's `PYTHON=` line, or — if the user already has an
  env — from a path they name and you verify (`<py> -c 'import demars_core'`). Not from
  `which python`, not from the conda default, not from the shell you happen to be in.
- **Do not activate anything.** Every call goes through the env's own interpreter. The shell is
  re-initialized between calls, so `conda activate` is undone before the next command runs.
- **Absent is honest; unchecked is not "absent".** No ICSD zip → `paths.icsd_db: null` and the
  records say the sibling search did **not run**. No MP key → `gates.hull` is `not_run`. Both are
  supported configurations. Say plainly which gates this deployment cannot check; never present a
  WARN from doctor.sh as a pass.
- **Licensed data moves, it is never fetched.** The ICSD archive is the user's; build_icsd.sh
  copies or links the file they point at. The nano checkpoint is likewise a file they download
  (SevenNet-nano weights) — if it is not in the directory they name, report the placeholder and
  where to put the file, do not substitute a name you hope sevenn resolves.
- **An existing `demars.yaml` is a reconfigure, not a setup.** Read it, tell the user what is set,
  and change only what they ask. `write_config.py` refuses to overwrite without `--force`; pass
  `--force` only after they have said so.
- **Never during a run.** If `runs/*/_batch/` has a live ledger or an analyst is running, stop:
  a reinstalled `demars_core` or a re-pointed checkpoint mid-campaign changes the provenance of the
  second half of the batch.

## Procedure

**0. Look before asking.** From the repo root:
```bash
ls demars.yaml 2>/dev/null; conda env list 2>/dev/null | grep -v '^#'; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
ls ~/icsd_db 2>/dev/null; python3 version/check_version.py verify
```
If `demars.yaml` exists → run `assets/setup/doctor.sh` first and go straight to whatever it flags.

**1. Environment.** Ask once for: conda or venv; env name
(default `demars`); GPU present → CUDA torch tag (`cu128` is right for current drivers; `cpu` for
a CPU-only box; `skip` if they insist torch is already handled). If they already have a demars env,
skip creation and verify the interpreter they name instead.
```bash
assets/setup/make_env.sh --manager conda --name demars --torch cu128      # prints PYTHON=…
```
Takes minutes (torch is large). Capture the `PYTHON=` line; it is the only thing you carry forward.

**2. Config.** Ask: where the SevenNet checkpoints live (the nano file `checkpoint_7net_nano_5.5.pth`
must be there); whether they have a Materials Project key (optional; it enables `gates.hull`);
whether to run a fused-kernel accelerator (default `none` — say so, do not ask unless they bring
it up). ICSD is asked in step 3, so write the config **after** step 3 or rerun with `--force`:
```bash
python3 assets/setup/write_config.py --python "$PYTHON" --checkpoint-dir /path/to/sevennet_ckpt \
        [--icsd-db "$ICSD_DB_DIR"] [--mp-key "$KEY"]
```
Exit 1 with `WARN` lines means the file was written but something is a placeholder — report each.

**3. ICSD (optional).** Ask: "Do you have a licensed `icsd_cif.zip`? If so, where?" Offer the
default dir `~/icsd_db` and `--link` (symlink, no 350 MB copy). If they don't have one, skip and
leave `icsd_db` unset — and tell them what that means for the records.
```bash
assets/setup/build_icsd.sh --zip /path/to/icsd_cif.zip --python "$PYTHON" [--dir ~/icsd_db] [--link]
```
Indexing 241k CIFs takes minutes and scales with cores. It prints `icsd-query stats` at the end;
compare with the expected table in `assets/icsd_query/README.md` — a different `rows` count is a
different archive vintage, not a bug. If `~/.local/bin` is not on PATH the script says so; the
user has to fix their profile, you cannot.

**4. Verify.**
```bash
assets/setup/doctor.sh              # seconds
assets/setup/doctor.sh --reference  # + bundled COD structures vs stored answers (minutes)
```
Offer `--reference` and run it when they say yes; it is the only check that proves the potential
gives the right answers, not merely that it loads.

## Report

A short table of doctor.sh's lines, then in prose: which of the five gates this deployment can
run (hull needs the MP key), whether the sibling search is checked or unchecked, the interpreter
path that is now in `paths.python`, and the exact `--icsd-id`/`icsd-query` capability
(present only if `icsd-query` is on PATH). End with the one command that starts real work:
`deaverage <file>`.
