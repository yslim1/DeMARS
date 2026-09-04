#!/bin/bash
#SBATCH -J demars-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=CHANGE_ME
#SBATCH -o STDOUT.%N.%j.out
#SBATCH -e STDERR.%N.%j.err
#
# One structure -> one MAR. Output lands in runs/<cif-stem>/.
# Set --partition above (see README.md), then submit FROM THE REPO ROOT:
#
#     sbatch assets/slurm/single.j path/to/structure.cif

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
[ -f demars.yaml ] || { echo "submit from the repo root"; exit 1; }

CIF="${1:?usage: sbatch single.j <structure.cif>}"

codex exec --approve-for-me --dangerously-bypass-hook-trust \
  "deaverage $CIF"
