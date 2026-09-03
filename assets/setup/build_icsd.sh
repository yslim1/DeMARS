#!/bin/bash
# Lay out ICSD_DB_DIR from the user's own licensed icsd_cif.zip and build the icsd-query index.
# Automates assets/icsd_query/README.md "Install" -- read that for what each step is.
#
#   assets/setup/build_icsd.sh --zip /path/to/icsd_cif.zip --python /abs/envs/demars/bin/python \
#                              [--dir ~/icsd_db] [--link] [--bin ~/.local/bin] [--rebuild]
#
# --python  the demars env interpreter (build_db.py needs pymatgen + tqdm, which it has).
# --link    symlink the zip instead of copying its ~350 MB. build_db.py records the zip's absolute
#           path in the DB; a symlink is fine as long as the target stays put.
# The archive is licensed content: this script moves it into place, never fetches it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/assets/icsd_query"
zip=""; dir="$HOME/icsd_db"; py=""; link=0; bin="$HOME/.local/bin"; rebuild=0

while [ $# -gt 0 ]; do
    case "$1" in
        --zip)     zip="$2"; shift 2 ;;
        --dir)     dir="$2"; shift 2 ;;
        --python)  py="$2";  shift 2 ;;
        --bin)     bin="$2"; shift 2 ;;
        --link)    link=1;   shift ;;
        --rebuild) rebuild=1; shift ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "build_icsd.sh: unknown argument $1" >&2; exit 2 ;;
    esac
done
[ -n "$zip" ] && [ -f "$zip" ] || { echo "build_icsd.sh: --zip must name an existing icsd_cif.zip" >&2; exit 2; }
[ -n "$py" ] && [ -x "$py" ]   || { echo "build_icsd.sh: --python must be the demars env interpreter" >&2; exit 2; }
dir="$(mkdir -p "$dir" && cd "$dir" && pwd)"
zip="$(cd "$(dirname "$zip")" && pwd)/$(basename "$zip")"

cp "$SRC"/*.py "$dir/"
if [ "$zip" != "$dir/icsd_cif.zip" ]; then
    if [ $link -eq 1 ]; then ln -sfn "$zip" "$dir/icsd_cif.zip"; else cp "$zip" "$dir/icsd_cif.zip"; fi
fi

if [ -f "$dir/icsd.sqlite" ] && [ $rebuild -eq 0 ]; then
    echo "build_icsd.sh: $dir/icsd.sqlite exists -- keeping it (--rebuild to redo)" >&2
else
    [ $rebuild -eq 1 ] && rm -f "$dir/icsd.sqlite"
    echo "build_icsd.sh: indexing $dir/icsd_cif.zip (minutes, scales with cores)" >&2
    ( cd "$dir" && ICSD_DB_DIR="$dir" "$py" build_db.py ) >&2
fi

mkdir -p "$bin"
install -m 755 "$SRC/icsd-query" "$bin/icsd-query"
case ":$PATH:" in *":$bin:"*) ;; *)
    echo "build_icsd.sh: NOTE $bin is not on PATH -- add it, or 'icsd-query extract' will not be found" >&2 ;;
esac

ICSD_DB_DIR="$dir" DEMARS_PY="$py" "$bin/icsd-query" stats
echo "ICSD_DB_DIR=$dir"
