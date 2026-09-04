#!/bin/bash
#SBATCH -J demars-batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=04-00:00
#SBATCH --partition=CHANGE_ME
#SBATCH -o STDOUT.%N.%j.out
#SBATCH -e STDERR.%N.%j.err
#
# Many structures -> one resumable ledger. Set --partition above (see README.md) and the two
# paths below, then submit FROM THE REPO ROOT:
#
#     sbatch assets/slurm/batch.j
#
# One pass, then exit. Resubmit to resume: the skill reads the ledger and skips what is finished.

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
[ -f demars.yaml ] || { echo "submit from the repo root"; exit 1; }

TARGET_DIR="./testcases/my_set/cifs"   # EDIT -- a directory of .cif files
ROOT="./runs/my_set"                   # EDIT -- entries in $ROOT/<stem>/, ledger in $ROOT/_batch/

TOTAL=$(ls "$TARGET_DIR"/*.cif | wc -l)
echo "start: $(tools/py tools/batch_log.py "$ROOT/_batch" --done 2>/dev/null | wc -l)/$TOTAL done"

codex exec --approve-for-me --dangerously-bypass-hook-trust \
  "deaverage the structures in $TARGET_DIR (out root $ROOT)"

tools/py tools/batch_log.py "$ROOT/_batch" --summary 2>/dev/null

# Compare against the INPUT. --summary lists the rows that exist; it cannot list the ones that are
# missing, and a session that dies mid-entry leaves no row at all -- so a truncated batch reads as
# a clean one until someone counts the inputs by hand.
echo "end: $(tools/py tools/batch_log.py "$ROOT/_batch" --done 2>/dev/null | wc -l)/$TOTAL done"
