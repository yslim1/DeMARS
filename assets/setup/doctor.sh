#!/bin/bash
# Report whether a DeMARS deployment is complete. Prints one PASS/WARN/FAIL line per check and
# exits non-zero on any FAIL. WARN marks an honest-but-unchecked configuration (no ICSD, no MP
# key) -- the pipeline runs, and the records will say those gates/searches did not run.
#
#   assets/setup/doctor.sh                # config, interpreter, imports, GPU, ICSD, MP, version
#   assets/setup/doctor.sh --reference    # ... plus tools/demars_reference.py (minutes on CPU)
#   assets/setup/doctor.sh --pytest       # ... plus the full suite (~4 min), from the repo root
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ref=0; pyt=0
for a in "$@"; do case "$a" in --reference) ref=1 ;; --pytest) pyt=1 ;; esac; done

fails=0
ok()   { printf 'PASS  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; fails=$((fails+1)); }

# 1. config file
cfg="${DEMARS_CONFIG:-}"
for c in "$cfg" "$ROOT/demars.yaml" "$HOME/.config/demars/demars.yaml"; do
    [ -n "$c" ] && [ -f "$c" ] && { cfg="$c"; break; }; cfg=""
done
if [ -n "$cfg" ]; then ok "config: $cfg"; else fail "config: no demars.yaml (write_config.py)"; fi

# 2. interpreter via tools/py, exactly as the stages resolve it
py="$(tools/py -c 'import sys; print(sys.executable)' 2>/dev/null)"
if [ -z "$py" ]; then
    fail "interpreter: tools/py could not resolve paths.python"
else
    if PYTHONPATH=tools tools/py -c 'import _icsd_env, demars_core, torch' 2>/dev/null; then
        ok "interpreter: $py imports demars_core + torch"
    else
        fail "interpreter: $py cannot import demars_core/torch (make_env.sh)"
    fi
    # The engine's own device pick: CUDA only if this torch build has kernels for the installed GPU.
    # A mismatch falls back to CPU silently at run time, many times slower, with nothing in the record.
    gpu="$(PYTHONPATH=tools tools/py - 2>/dev/null <<'PY'
import _icsd_env, torch  # noqa
from demars_core import _torchsim
dev = _torchsim._pick_device()
if not torch.cuda.is_available():
    print('WARN  gpu: torch sees no CUDA device -- relaxations run on CPU, many times slower')
elif dev.type == 'cpu':
    M, m = torch.cuda.get_device_capability(0)
    print(f'WARN  gpu: {torch.cuda.get_device_name(0)} is sm_{M}{m} but torch {torch.__version__} has '
          f'{sorted(torch.cuda.get_arch_list())} -- the engine will FALL BACK TO CPU. Install a matching torch.')
else:
    print(f'PASS  gpu: {torch.cuda.get_device_name(0)} via {dev} (torch {torch.__version__})')
PY
)"
    echo "${gpu:-FAIL  gpu: could not query torch}"
    case "$gpu" in FAIL*) fails=$((fails+1)) ;; esac
fi

# 3. models
if [ -n "$py" ]; then
    out="$(tools/py - <<'PY'
import os, sys
sys.path.insert(0, 'tools'); import _icsd_env  # noqa
from demars_core import models
for alias in ('nano', 'omni'):
    m, _ = models.resolve(alias)
    if not m:
        print(f'FAIL  models.{alias}: not declared in the config'); continue
    spec = m['spec']
    if '/' in spec and not os.path.isfile(spec):
        print(f'FAIL  models.{alias}: checkpoint file missing: {spec}')
    elif '/' in spec:
        print(f'PASS  models.{alias}: {spec}  sha256={models.hash_file(spec)[:12]}')
    else:
        print(f'PASS  models.{alias}: {spec} (sevenn pretrained name; fetched on first use)')
try:
    models.accelerator_kwargs()
    print(f'PASS  compute.accelerator: {models.accelerator()}')
except Exception as e:
    print(f'FAIL  compute.accelerator: {e}')
PY
)" || out="FAIL  models: config could not be loaded"
    echo "$out"
    fails=$((fails + $(echo "$out" | grep -c '^FAIL')))
fi

# 4. ICSD sibling search
icsd="$(tools/py -c 'import sys; sys.path.insert(0,"tools"); import _icsd_env; from demars_core import models; print(models.icsd_db_dir() or "")' 2>/dev/null)"
if [ -z "$icsd" ]; then
    warn "icsd: paths.icsd_db unset -- sibling search UNCHECKED ('none (no sibling DB configured)')"
elif [ -f "$icsd/icsd.sqlite" ] && [ -f "$icsd/icsd_cif.zip" ]; then
    if PYTHONPATH=tools tools/py -c 'import _icsd_env, sys; sys.exit(0 if _icsd_env.sibling_db_available() else 1)' 2>/dev/null; then
        ok "icsd: $icsd (sqlite + zip; sibling bootstrap loaded)"
    else
        fail "icsd: $icsd has the files but mar_evidence's sibling bootstrap did not load"
    fi
    if command -v icsd-query >/dev/null; then ok "icsd-query: $(command -v icsd-query)"
    else warn "icsd-query: not on PATH -- 'icsd-query extract' (input CIFs by id) will not work"; fi
else
    fail "icsd: $icsd is set but icsd.sqlite / icsd_cif.zip missing (build_icsd.sh)"
fi

# 5. Materials Project key
if tools/py -c 'import sys; sys.path.insert(0,"tools"); import _icsd_env; from demars_core import models; sys.exit(0 if models.mp_api_key() else 1)' 2>/dev/null; then
    ok "materials_project: key present (gates.hull can run)"
else
    warn "materials_project: no key -- gates.hull will be not_run (UNCHECKED)"
fi

# 6. version layer
if out="$(python3 version/check_version.py verify 2>&1)"; then
    ok "version: $(echo "$out" | grep -E '^(Version|Mode):' | tr '\n' ' ')"
else
    fail "version: $(echo "$out" | tail -1)"
fi

# 7. optional deep checks
if [ $ref -eq 1 ]; then
    if tools/py tools/demars_reference.py; then ok "reference: bundled COD structures match"; else fail "reference: see output above"; fi
fi
if [ $pyt -eq 1 ]; then
    if tools/py -m pytest -q; then ok "pytest"; else fail "pytest: see output above"; fi
fi

echo
if [ $fails -eq 0 ]; then echo "doctor: no failures"; else echo "doctor: $fails failure(s)"; fi
exit $(( fails > 0 ))
