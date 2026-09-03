#!/bin/bash
# Create (or reuse) the DeMARS interpreter environment and install demars-core into it.
#
#   assets/setup/make_env.sh --manager conda --name demars --torch cu128
#   assets/setup/make_env.sh --manager venv  --dir .venv    --torch cpu
#
# Prints exactly one machine-readable line at the end:   PYTHON=/abs/path/to/python
# That path is what `write_config.py --python` takes -- the env is never guessed, it is the one
# this script actually produced.
#
# --torch  cu128 | cu126 | cu121 | cpu | skip
#          A CUDA-matched torch must be installed BEFORE sevenn pulls its own (README "Install").
#          `skip` leaves torch to sevenn's dependency resolution -- fine on CPU-only boxes.
# Nothing here activates a shell: every install goes through the env's own interpreter, so it
# works from an agent whose shell is re-initialized between calls.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
manager=conda; name=demars; dir=""; pyver=3.12; torch=skip; extras="sevennet,test"

while [ $# -gt 0 ]; do
    case "$1" in
        --manager) manager="$2"; shift 2 ;;
        --name)    name="$2";    shift 2 ;;
        --dir)     dir="$2";     shift 2 ;;
        --python)  pyver="$2";   shift 2 ;;
        --torch)   torch="$2";   shift 2 ;;
        --extras)  extras="$2";  shift 2 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "make_env.sh: unknown argument $1" >&2; exit 2 ;;
    esac
done

case "$manager" in
    conda)
        command -v conda >/dev/null || { echo "make_env.sh: conda not on PATH" >&2; exit 1; }
        if conda env list | awk '{print $1}' | grep -qx "$name"; then
            echo "make_env.sh: conda env '$name' already exists -- reusing it" >&2
        else
            conda create -y -n "$name" "python=$pyver" >&2
        fi
        py="$(conda run -n "$name" python -c 'import sys; print(sys.executable)')"
        ;;
    venv)
        dir="${dir:-$ROOT/.venv}"
        if [ -x "$dir/bin/python" ]; then
            echo "make_env.sh: venv at $dir already exists -- reusing it" >&2
        else
            "python$pyver" -m venv "$dir" >&2 2>/dev/null || python3 -m venv "$dir" >&2
        fi
        py="$(cd "$dir" && pwd)/bin/python"
        ;;
    *) echo "make_env.sh: --manager must be conda or venv" >&2; exit 2 ;;
esac

[ -x "$py" ] || { echo "make_env.sh: interpreter not found at $py" >&2; exit 1; }
"$py" -m pip install --upgrade pip >&2

case "$torch" in
    skip) ;;
    cpu|cu*) "$py" -m pip install torch --index-url "https://download.pytorch.org/whl/$torch" >&2 ;;
    *) echo "make_env.sh: --torch must be cuNNN, cpu or skip" >&2; exit 2 ;;
esac

"$py" -m pip install -e "$ROOT/demars-core[$extras]" >&2

# Prove the install before handing the path on.
"$py" - >&2 <<'PY'
import demars_core, torch
print(f"make_env.sh: demars_core OK; torch {torch.__version__}, cuda={torch.cuda.is_available()}")
PY

echo "PYTHON=$py"
