"""Import-order preamble: set the environment `demars_core` reads at import time.

`demars_core._engine.mar_evidence` reads DEMARS_ORDERED_LOOKUP **at module import time** and exec's
it to build `_sib`, and that exec'd code reads ICSD_DB_DIR. Any tool that touches demars_core must
therefore `import _icsd_env` BEFORE its first demars_core import, or the sibling search silently
degrades to "none (no sibling DB configured)".

Import this first, always. It is idempotent and never raises: a missing or unreadable config just
leaves the variables unset, and the sibling bootstrap degrades cleanly.

NOTE: this parses demars.yaml itself rather than calling `demars_core.models`, deliberately.
Importing anything from demars_core would pull in mar_evidence -- the very module whose environment
we are still setting up -- so the read has to be self-contained. It is a handful of lines and only
needs two keys.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

os.environ.setdefault('DEMARS_ORDERED_LOOKUP', os.path.join(_HERE, 'ordered_lookup.py'))


def _config():
    """The same discovery order demars_core.models uses, kept minimal and dependency-tolerant."""
    cands = [os.environ.get('DEMARS_CONFIG')]
    for base in ('demars.yaml', 'demars.yml', 'demars.json'):
        cands += [os.path.join(os.getcwd(), base), os.path.join(_ROOT, base),
                  os.path.expanduser(os.path.join('~', '.config', 'demars', base))]
    for path in cands:
        if not (path and os.path.isfile(path)):
            continue
        try:
            with open(path, encoding='utf-8') as fh:
                text = fh.read()
            if path.endswith('.json'):
                import json
                return json.loads(text) or {}
            import yaml
            return yaml.safe_load(text) or {}
        except Exception:
            return {}                       # a broken config must not stop a tool from running
    return {}


_cfg = _config()
_paths = _cfg.get('paths') or {}
_mp = _cfg.get('materials_project') or {}

if isinstance(_paths, dict) and _paths.get('icsd_db'):
    os.environ.setdefault('ICSD_DB_DIR', str(_paths['icsd_db']))
if isinstance(_mp, dict) and _mp.get('api_key'):
    # exported so a hull tool picks it up the same way it would from the shell; the environment
    # still wins if the user already set it
    os.environ.setdefault('MP_API_KEY', str(_mp['api_key']))


def sibling_db_available():
    """True iff mar_evidence's _sib bootstrap actually loaded (imported lazily so this module stays
    safe to import before demars_core)."""
    from demars_core._engine import mar_evidence as MEV
    return MEV._sib is not None
