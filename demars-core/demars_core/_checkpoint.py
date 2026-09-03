"""Crash-resumable relaxation -- keep the configs that finished (defect D14).

Nothing used to be written until every config of a round had relaxed
(`api.deaverage` -> `MARRecord.write_ensemble`, after the loop), so a timeout or an OOM at
config 29/30 discarded all 29 and the next attempt started from zero.

This wraps the engine's `relax` callable instead of touching the engine: relax in chunks, append
each finished config to an extxyz checkpoint, hand back on the next run what is already there.

Two decisions:
  * The cache key is the DECORATION (symbols + positions + cell) plus a tag naming the potential,
    not the sample label -- labels like `rand7` repeat across rounds and settings, so a
    label-keyed cache would hand a round-1 energy to a round-2 config. A changed seed/cut/cell
    simply misses; the tag is what stops a nano energy being reused for an omni run.
  * ONLY converged results are cached. A `val=False` can be the environment (OOM, a CUDA build
    with no kernels for this GPU) rather than the structure, and caching it would make a
    transient fault permanent -- the laundering D10 is about. Failures are recomputed.
"""
import hashlib
import os
import sys
import warnings
from io import StringIO

import numpy as np

__all__ = ['config_key', 'relaxer_tag', 'read_checkpoint', 'append_result', 'checkpointed',
           'discard']

_KEY = 'demars_ckpt_key'
_ENE = 'demars_ckpt_E'


def config_key(atoms, tag=''):
    """-> stable hex id for (this exact decoration, this potential)."""
    def _norm(a):
        # `+ 0.0` so -0.0 and +0.0 hash alike: they compare equal but have different bits.
        return (np.round(np.asarray(a, dtype=float), 6) + 0.0).tobytes()

    h = hashlib.sha1()
    h.update(str(tag).encode())
    h.update(' '.join(atoms.get_chemical_symbols()).encode())
    h.update(_norm(atoms.get_positions()))
    h.update(_norm(atoms.get_cell()))
    return h.hexdigest()


def relaxer_tag(mode, calculator, d3):
    """-> stable name for the potential behind a relax callable, for `config_key`.

    MUST NOT vary between processes. `str(<an ASE Calculator instance>)` carries a memory address,
    so a tag built from it names a different potential every run and NOTHING EVER RESUMES -- and
    silently, because a cache miss is indistinguishable from work still to do. A class name is the
    most an injected object can honestly be identified by (a second checkpoint of the same class
    is not distinguished; native SevenNet specs are strings and carry their own identity).
    """
    cal = calculator if isinstance(calculator, str) else type(calculator).__name__
    return f'{mode}|{cal}|d3={bool(d3)}'


def _complete_frames(text):
    """-> [frame_text, ...] for every COMPLETE extxyz frame, stopping at a torn tail.

    A kill can leave the last frame half-written. `ase.io.read(index=':')` raises on that and
    would cost the whole file; counting `natoms` headers ourselves costs the torn frame only.
    """
    lines = text.splitlines(keepends=True)
    out, i, n = [], 0, len(lines)
    while i < n:
        try:
            na = int(lines[i].strip())
        except ValueError:
            break
        end = i + 2 + na
        if end > n or not lines[end - 1].endswith('\n'):
            break
        out.append(''.join(lines[i:end]))
        i = end
    return out


def read_checkpoint(path):
    """-> {key: (relaxed Atoms, total energy)}; {} when there is nothing usable."""
    if not path or not os.path.exists(path):
        return {}
    from ase.io import read as aseread
    try:
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
    except OSError as exc:
        warnings.warn(f'checkpoint {path} unreadable ({exc}); starting from scratch', stacklevel=2)
        return {}
    out = {}
    for frame in _complete_frames(text):
        try:
            at = aseread(StringIO(frame), format='extxyz')
        except Exception:
            break                       # a frame we cannot parse ends our trust in the rest
        k, e = at.info.pop(_KEY, None), at.info.pop(_ENE, None)
        if k is not None and e is not None:
            out[str(k)] = (at, float(e))
    return out


def append_result(path, atoms, key, energy):
    """Append one finished config -- one frame per call, so a kill tears at most the frame in
    flight, never one already reported as done."""
    from ase.io import write as asewrite
    at = atoms.copy()                   # .copy() also drops the attached calculator
    at.info = {k: v for k, v in atoms.info.items() if isinstance(v, (str, int, float, bool))}
    at.info[_KEY] = str(key)
    at.info[_ENE] = float(energy)
    buf = StringIO()
    asewrite(buf, at, format='extxyz')
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(buf.getvalue())
        fh.flush()
        os.fsync(fh.fileno())


def discard(path):
    """Remove a checkpoint once `ensemble.xyz` exists -- a second, possibly stale copy of the
    ensemble in the run dir is the ambiguity the `_work/` layout rule exists to prevent."""
    try:
        os.remove(path)
    except OSError:
        pass


def checkpointed(relax, path, tag='', chunk=1):
    """Wrap `relax(confs) -> (E, rel, val)` so finished configs survive a kill.

    chunk : configs handed to the inner relaxer at a time = the LOSS UNIT. 1 for the serial ASE
        paths (they relax one at a time anyway, so it costs nothing); a small batch for the
        batched backend, whose autobatcher budget is probed once and cached per model
        (`_torchsim._SCALER_CACHE`) -- splitting a round costs per-call setup, not re-calibration.
    """
    cache = read_checkpoint(path)
    if cache:
        print(f'[demars] checkpoint {path}: {len(cache)} relaxed config(s) available',
              file=sys.stderr)

    def _relax(confs):
        keys = [config_key(a, tag) for a in confs]
        todo = [i for i, k in enumerate(keys) if k not in cache]
        if len(todo) < len(confs):
            print(f'[demars] resuming: {len(confs) - len(todo)}/{len(confs)} config(s) reused, '
                  f'{len(todo)} to relax', file=sys.stderr)
        failed = {}
        step = max(1, int(chunk))
        for s in range(0, len(todo), step):
            idx = todo[s:s + step]
            E, rel, val = relax([confs[i] for i in idx])
            for j, i in enumerate(idx):
                if bool(val[j]) and np.isfinite(E[j]):
                    cache[keys[i]] = (rel[j], float(E[j]))
                    append_result(path, rel[j], keys[i], float(E[j]))
                else:
                    failed[i] = (float(E[j]), rel[j])     # never cached -- see module docstring
        Eo, relo, valo = [], [], []
        for i, k in enumerate(keys):
            if k in cache:
                at, e = cache[k]
                Eo.append(e); relo.append(at); valo.append(True)
            else:
                e, at = failed.get(i, (float('nan'), confs[i]))
                Eo.append(e); relo.append(at); valo.append(False)
        return np.asarray(Eo, dtype=float), relo, valo

    return _relax
