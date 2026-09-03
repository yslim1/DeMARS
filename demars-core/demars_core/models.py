"""Model pinning and compute provenance.

Two separate problems, both of which have to be solved or a long campaign is not reproducible:

  1. PINNING -- which checkpoint does this run use? Without a single declared answer the choice is
     assembled at call time from `SPINNER_NANO_CKPT` / `DEMARS_NANO_CUTOFF` / `DEMARS_CKPT_DIR` and a
     hardcoded default, so two runs on two machines (or the same machine after a shell change) can
     silently use different weights.
  2. STAMPING -- which checkpoint did this run ACTUALLY use? Pinning alone cannot answer that after
     the fact: edit the config halfway through a thousand-structure campaign and the first and second
     halves differ while their records look identical.

So the config file below declares the models, and `provenance()` records what was resolved into every
record. A path on its own is weak evidence -- the file it points at can be overwritten or repointed --
so the config may also carry a `sha256`, which is recorded and (optionally) verified.

The same file also carries the machine-specific settings that used to be hardcoded in the source --
the local checkpoint directory, the ICSD database directory, and the Materials Project key -- so a
new deployment is one file rather than a grep through the package.

Config discovery, first hit wins:
    $DEMARS_CONFIG                             explicit path
    ./demars.yaml | .yml | .json               the working directory of the run
    ~/.config/demars/demars.yaml | .yml | .json    per user
    (none)                                     built-in defaults -- previous behaviour, unchanged

See `demars.yaml` at the repository root for the annotated template. `spec` is anything
SevenNetModel accepts: a pretrained name ('7net-omni', '7net-nano-6.0') or a checkpoint path.
Environment variables still override the file -- they are the more specific signal -- but any
override shows up in the stamped provenance, so a run can never be silently off-config.
"""
import hashlib
import json
import os

__all__ = ['load_config', 'resolve', 'provenance', 'hash_file', 'config_path',
           'checkpoint_dir', 'icsd_db_dir', 'mp_api_key', 'accelerator', 'accelerator_kwargs']

_ENV_CONFIG = 'DEMARS_CONFIG'
_BASENAMES = ('demars.yaml', 'demars.yml', 'demars.json')

_cache = {'path': None, 'data': None, 'loaded': False}
_hash_cache = {}


def _candidates():
    out = [os.environ.get(_ENV_CONFIG)]
    out += [os.path.join(os.getcwd(), b) for b in _BASENAMES]
    out += [os.path.expanduser(os.path.join('~', '.config', 'demars', b)) for b in _BASENAMES]
    return out


def _parse(path):
    """YAML or JSON by extension. YAML is the documented format -- a deployment config needs
    comments -- and JSON stays accepted for anything that generates one."""
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    if path.endswith('.json'):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f'{path} is not valid JSON: {e}') from e
    try:
        import yaml
    except ImportError as e:                     # pragma: no cover
        raise ImportError(f'{path} needs PyYAML (pip install pyyaml), '
                          'or use a .json config instead') from e
    try:
        return yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise ValueError(f'{path} is not valid YAML: {e}') from e


def config_path():
    """-> the config file actually in use, or None when running on built-in defaults."""
    load_config()
    return _cache['path']


def load_config(reload=False):
    """-> the parsed config dict (empty when no file was found). Cached; `reload=True` re-reads.

    A malformed file is a hard error rather than a silent fallback: a campaign that thinks it is
    pinned but is not is worse than one that refuses to start.
    """
    if _cache['loaded'] and not reload:
        return _cache['data']
    data, path = {}, None
    for cand in _candidates():
        if cand and os.path.isfile(cand):
            data = _parse(cand)
            if not isinstance(data, dict):
                raise ValueError(f'{cand}: top level must be a mapping')
            if not isinstance(data.get('models', {}), dict):
                raise ValueError(f'{cand}: "models" must be a mapping')
            path = cand
            break
    _cache.update(path=path, data=data, loaded=True)
    return data


def _paths():
    p = load_config().get('paths') or {}
    return p if isinstance(p, dict) else {}


def checkpoint_dir(default=None):
    """Directory of local SevenNet checkpoints, used when a model spec is a bare name.
    $DEMARS_CKPT_DIR overrides the config. Unset simply means "no local checkpoint directory"."""
    return os.environ.get('DEMARS_CKPT_DIR') or _paths().get('checkpoint_dir') or default


def icsd_db_dir(default=None):
    """Directory of the local ICSD database (icsd.sqlite + icsd_cif.zip) backing the ordered-sibling
    search. Absent -> the search reports 'none (no sibling DB configured)', which the analyst must
    read as UNCHECKED, not as 'no ordered form exists'."""
    return os.environ.get('ICSD_DB_DIR') or _paths().get('icsd_db') or default


def mp_api_key(default=None):
    """Materials Project API key. $MP_API_KEY wins over the config file -- a key sitting in a file is
    easy to commit by accident, so the environment must be able to override it.

    NOT used by the ordered-sibling search, which is ICSD-only by design: an MP entry says a
    structure was COMPUTED, not that an ordered form was ever observed. This is for the convex hull."""
    mp = load_config().get('materials_project') or {}
    key = mp.get('api_key') if isinstance(mp, dict) else None
    return os.environ.get('MP_API_KEY') or key or default


# Fused-kernel backends SevenNet can run its equivariant tensor products on. The name is the
# vocabulary; the value is how sevenn is told; the module is what has to be importable for the
# claim to be true.
_ACCELERATORS = {
    'none': ({}, None),
    'cueq': ({'enable_cueq': True}, 'cuequivariance_torch'),   # NVIDIA cuEquivariance
    'oeq': ({'enable_oeq': True}, 'openequivariance'),
    'flash': ({'enable_flash': True}, None),                   # no separate distribution
}


def accelerator(default='none'):
    """-> which fused-kernel backend to run SevenNet's tensor products on: 'none'|'cueq'|'oeq'|'flash'.

    $DEMARS_ACCELERATOR overrides the config's `compute.accelerator`, same rule as every other
    setting here. The older $DEMARS_ENABLE_CUEQ=1 still means 'cueq'.

    This belongs in the config file and not only in the environment because it CHANGES THE NUMBERS.
    Fused kernels sum in a different order, so forces move in the last float32 digits (measured:
    4.8e-6 eV/A on a 576-atom cell, ~3e-7 relative). That is chemically nothing and reproducibly
    something -- and a setting that lives only in a shell variable differs between two windows on
    one machine, which is the worst possible place for anything a record has to explain.
    """
    env = os.environ.get('DEMARS_ACCELERATOR')
    if env is None and os.environ.get('DEMARS_ENABLE_CUEQ') == '1':
        env = 'cueq'
    cfg = load_config().get('compute') or {}
    name = env or (cfg.get('accelerator') if isinstance(cfg, dict) else None) or default
    name = str(name).strip().lower()
    if name in ('', 'null', 'none', 'off', 'false', 'no'):
        return 'none'
    if name not in _ACCELERATORS:
        raise ValueError(f'unknown accelerator {name!r}; expected one of '
                         f'{", ".join(sorted(_ACCELERATORS))}')
    return name


def accelerator_kwargs(name=None):
    """-> the sevenn constructor kwargs for `name` (default: the resolved accelerator).

    Raises if the backend was asked for but is not installed. Falling back silently would put an
    environment fault into a record as a chemical result -- the same laundering d3 refuses (see
    _torchsim._build_model): a tier that COULD NOT run must never read as a tier that was never
    asked for.
    """
    name = accelerator() if name is None else name
    kw, module = _ACCELERATORS[name]
    if module is not None:
        try:
            __import__(module)
        except ImportError as e:
            raise RuntimeError(
                f'accelerator {name!r} needs {module}, which is not installed '
                f'(pip install {module.replace("_", "-")}). Set compute.accelerator to "none" '
                f'in the config, or install it -- but do not run half-configured: the records '
                f'would not say which kernels produced them.') from e
    return dict(kw)


def resolve(alias):
    """-> ({'spec', 'modal'}, source) for a model alias ('nano' / 'omni'), or (None, None).

    `source` names where the answer came from, so it can be stamped.
    """
    cfg = load_config()
    entry = (cfg.get('models') or {}).get(alias)
    if not isinstance(entry, dict) or not entry.get('spec'):
        return None, None
    return {'spec': entry['spec'], 'modal': entry.get('modal')}, _cache['path']


def hash_file(path, _chunk=1 << 20):
    """sha256 of a local checkpoint, cached per process. None for anything that is not a file
    (a pretrained NAME has nothing local to hash).

    Computed automatically and stamped into the record rather than declared in the config: the path
    says WHERE the weights came from, the hash says WHICH weights they were, and only the second
    lets a reader confirm they hold the same file. SevenNet-Nano checkpoints are a couple of MB, so
    this costs milliseconds -- there is nothing to optimise away.
    """
    if not (isinstance(path, str) and os.path.isfile(path)):
        return None
    path = os.path.abspath(path)
    if path in _hash_cache:
        return _hash_cache[path]
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(_chunk), b''):
            h.update(block)
    _hash_cache[path] = h.hexdigest()
    return _hash_cache[path]


def _version(module):
    """Best-effort version of an installed dependency. Tries the module attribute first (cheap when
    already imported), then the distribution metadata -- several packages no longer expose
    __version__ at top level."""
    try:
        mod = __import__(module)
        v = getattr(mod, '__version__', None)
        if v:
            return v
    except Exception:
        pass
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(module.replace('_', '-'))
        except PackageNotFoundError:
            return None
    except Exception:
        return None


def provenance(spec=None, tag=None, modal=None, d3=None, device=None, source=None,
               accel=None):
    """The compute-provenance block stamped into a record.

    Enough to answer "would this run reproduce?" without access to the machine that produced it:
    which weights (name or path, plus the sha256 of the local file), which modal, whether dispersion
    was on, what resolved it, and the versions of everything that can change the numbers underneath.

    `accel` names the fused-kernel backend, and it is stamped because the weights hash CANNOT stand
    in for it: cuEquivariance does not touch the checkpoint, so sha256 matches while the numbers
    move slightly -- and `tools/demars_reference.py` compares energies precisely WHEN the sha
    matches. Without this field a kernel change is indistinguishable from a chemistry change.
    """
    accel = accelerator() if accel is None else accel
    versions = {'demars_core': _version('demars_core'), 'sevenn': _version('sevenn'),
                'torch': _version('torch'), 'torch_sim': _version('torch_sim'),
                'ase': _version('ase'), 'pymatgen': _version('pymatgen')}
    module = _ACCELERATORS.get(accel, ({}, None))[1]
    if module:
        versions[module] = _version(module)
    return {
        'model_spec': spec, 'model_tag': tag, 'modal': modal, 'd3': d3,
        'checkpoint_sha256': hash_file(spec),
        'resolved_from': source or 'built-in defaults',
        'config_file': config_path(),
        'device': device,
        'accelerator': accel,
        'versions': versions,
    }
