# Setup scripts

The deterministic half of the `/setup` skill (`.claude/skills/setup/SKILL.md`). Each is usable
on its own; the skill just asks the questions and runs them in this order.

| script | does | prints |
|---|---|---|
| `make_env.sh` | conda or venv env, CUDA-matched torch first, `pip install -e demars-core[sevennet,test]` | `PYTHON=/abs/path` |
| `write_config.py` | `demars.yaml` from `assets/demars.yaml.example`, keys filled and comments kept; verifies the interpreter imports `demars_core` and the file reads back | what was set, `WARN` per placeholder |
| `build_icsd.sh` | `ICSD_DB_DIR` from **your** licensed `icsd_cif.zip`: copies the indexer, copies/links the zip, builds `icsd.sqlite`, installs `icsd-query` | `icsd-query stats`, `ICSD_DB_DIR=` |
| `doctor.sh` | config → interpreter → GPU → models → ICSD → MP key → version; `--reference`, `--pytest` for the deep checks | `PASS/WARN/FAIL` per line |

`WARN` is an honest gap, not a failure: no ICSD means the sibling search reports *unchecked*, no
MP key means `gates.hull` is `not_run`. The pipeline runs either way and the records say so.

```bash
assets/setup/make_env.sh --manager conda --name demars --torch cu128
python3 assets/setup/write_config.py --python "$PYTHON" --checkpoint-dir /path/to/sevennet_ckpt
assets/setup/build_icsd.sh --zip /path/to/icsd_cif.zip --python "$PYTHON" --link
assets/setup/doctor.sh --reference
```
