"""The batched relaxation backend -- SevenNet driven by `torch-sim`.

This is DeMARS's native fast path. `mar_engine` imports it as `S` and uses exactly
two entry points; `api.py` builds the engine's injected `relax=` callable from the
same pair:

    use_model(model=, modal=, d3=)          -> select the MLIP
    batched_fire_relax(atoms_list, **kw)    -> (E, rel, val)

The (E, rel, val) contract every caller depends on:
    E   : np.ndarray of TOTAL potential energies (eV), one per input, NaN on failure
    rel : list of relaxed ase.Atoms, input order
    val : np.ndarray of bool (False = did not converge / failed)

(This replaced an earlier batched-relaxation backend, whose `use_model` /
`batched_fire_relax` names and (E, rel, val) contract are kept -- the engine's call
sites are unchanged.)

Design notes:

  * Memory is managed by torch-sim's `InFlightAutoBatcher`, with the budget probed
    once per model, cached, padded for headroom, and shrunk-and-retried on a runtime
    OOM (see `_max_memory_scaler` / `_optimize_with_oom_fallback`). This replaced a
    fixed 6000-atom pre-chunk. All of it is CUDA-only, so it is off on CPU.
  * `val` is derived post-hoc from converged forces. torch-sim marks
    max-steps-exceeded systems as "converged" only to evict them from the batch,
    so the optimizer loop's own flag cannot be trusted as a success signal.
  * Variable-cell relaxation uses torch-sim's Frechet cell filter, matching the
    FrechetCellFilter the generic ASE path already prefers (`calculators.py`).
"""
import logging
import os
import warnings

import numpy as np

# The vendored engine calls warnings.filterwarnings('ignore') at import
# (mar_engine.py), which would make a warnings-only failure channel invisible.
# Log as well, so a failed batch is never silent.
_log = logging.getLogger(__name__)


def _warn(msg, stacklevel=3):
    _log.warning(msg)
    warnings.warn(msg, stacklevel=stacklevel)

__all__ = ['use_model', 'batched_fire_relax', 'active_model_tag', 'active_provenance',
           'clear_model_cache', 'release_models']

# --- nano resolution -----------------------------------------------------------
# SevenNet-Nano is distilled from Omni's mpa task and comes in four cutoff variants,
# named `7net-nano-<cutoff>` (4.5 / 5.0 / 5.5 / 6.0 A) -- single-task, so no modal.
#
# Those names are NOT resolvable in sevenn 0.13.0 (the latest PyPI release):
# `pretrained_name_to_path('7net-nano-5.0')` raises "Not a valid pretrained model
# name", and they are absent from get_available_pretrained_models(). The docs at
# sevennet.readthedocs.io/latest are built ahead of the release. So: try the
# pretrained registry FIRST (a future sevenn will resolve it and this needs no
# change), then fall back to a local checkpoint file.
_NANO_ALIASES = ('7net-nano', 'nano', 'sevennet', '')
_NANO_CUTOFFS = ('4.5', '5.0', '5.5', '6.0')
_NANO_DEFAULT_CUTOFF = '5.0'     # checkpoint_7net_nano_<cutoff>.pth

_MODEL_CACHE = {}
_PROV_CACHE = {}        # cache key -> compute-provenance dict, filled when the model is built
_ACTIVE = None          # (model_obj, tag)

# Errors that are NOT a relaxation failure: an unusable GPU or a missing dependency
# must never be laundered into val=False for every config (that reads as "these
# structures don't relax" when the truth is "this machine can't run the model").
_INFRA_ERROR_MARKERS = (
    'no kernel image', 'not compatible with the current pytorch',
    'cuda error', 'cuda capability', 'cudnn', 'cublas',
    'no cuda gpus are available', 'device-side assert',
    # An OOM that survived every shrink-and-retry is a resource fault, not a set of
    # structures that "failed to relax" -- reporting val=False for all of them would
    # write a misleading "no valid relaxations" into the record.
    'cuda out of memory', 'out of memory', 'failed to allocate',
    # An autobatcher budget smaller than one system's own metric is a CALIBRATION fault, and
    # torch-sim reports it as a plain ValueError ("... is greater than max_metric ..."). It is
    # neither an OOM (shrinking retries make it worse) nor a property of the structure, so
    # letting it reach the val=False path wrote "this structure failed to relax" into the
    # record for what is really a memory-probe artefact.
    'is greater than max_metric',
)


def _is_infra_error(exc):
    if isinstance(exc, (ImportError, FileNotFoundError, MemoryError)):
        return True
    return any(m in str(exc).lower() for m in _INFRA_ERROR_MARKERS)


def _pick_device():
    """-> torch.device for the model. DEMARS_DEVICE overrides; otherwise use CUDA
    only if this torch build actually has kernels for the installed GPU (a
    capability mismatch raises 'no kernel image' deep inside the first op, long
    after it looks like the model loaded fine)."""
    import torch
    env = os.environ.get('DEMARS_DEVICE')
    if env:
        return torch.device(env)
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        major, minor = torch.cuda.get_device_capability(0)
        supported = {int(a.split('_')[1]) for a in torch.cuda.get_arch_list()
                     if a.startswith('sm_')}
        if supported and (major * 10 + minor) < min(supported):
            _warn(f'GPU {torch.cuda.get_device_name(0)} (sm_{major}{minor}) is not '
                  f'supported by this torch build (has sm_{sorted(supported)}); '
                  'falling back to CPU. Install a matching torch, or set '
                  'DEMARS_DEVICE=cuda to override.')
            return torch.device('cpu')
    except Exception:
        pass
    return torch.device('cuda')


def _pretrained_ok(name):
    """True if sevenn can resolve `name` as a pretrained model (downloading it)."""
    try:
        import sevenn.util as u
        u.pretrained_name_to_path(name)
    except Exception:
        return False
    return True


def _resolve_nano(cutoff=None):
    """-> a model spec for SevenNet-Nano at `cutoff` (default DEMARS_NANO_CUTOFF).

    Order: SPINNER_NANO_CKPT -> the pinned config (see models.py) -> the
    `7net-nano-<cutoff>` pretrained name (once a sevenn release ships it) -> a local
    checkpoint_7net_nano_<cutoff>.pth.
    """
    env = os.environ.get('SPINNER_NANO_CKPT')
    if env:
        if not os.path.exists(env):
            raise FileNotFoundError(f'SPINNER_NANO_CKPT does not exist: {env}')
        return env
    cutoff = cutoff or os.environ.get('DEMARS_NANO_CUTOFF', _NANO_DEFAULT_CUTOFF)
    if cutoff not in _NANO_CUTOFFS:
        raise ValueError(f'unknown nano cutoff {cutoff!r}; SevenNet-Nano ships '
                         f'{", ".join(_NANO_CUTOFFS)} A')
    name = f'7net-nano-{cutoff}'
    if _pretrained_ok(name):
        return name                      # sevenn resolves + downloads it
    from . import models as _models
    ckpt_dir = _models.checkpoint_dir()          # $DEMARS_CKPT_DIR -> config paths.checkpoint_dir
    if not ckpt_dir:
        raise FileNotFoundError(
            f'SevenNet-Nano {cutoff} A unavailable: this sevenn cannot resolve {name!r} and no '
            'local checkpoint directory is configured. Set paths.checkpoint_dir in demars.yaml, '
            'or DEMARS_CKPT_DIR / SPINNER_NANO_CKPT.')
    path = os.path.join(ckpt_dir, f'checkpoint_7net_nano_{cutoff}.pth')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'SevenNet-Nano {cutoff} A unavailable: this sevenn cannot resolve '
            f'{name!r} and no checkpoint at {path}. Set SPINNER_NANO_CKPT to your '
            f'checkpoint, or paths.checkpoint_dir / DEMARS_NANO_CUTOFF to locate it.')
    return path


def _config_alias(model):
    """-> the models.json alias a model string belongs to ('nano' / 'omni'), or None.

    Only the BARE tier aliases are pinnable. An explicit request -- a checkpoint path, or a specific
    cutoff variant like '7net-nano-5.0' -- is already unambiguous and must win over the config, or a
    caller asking for a particular model would silently get a different one."""
    low = str(model).lower()
    if low in _NANO_ALIASES:
        return 'nano'
    if low in ('7net-omni', 'omni'):
        return 'omni'
    return None


def _resolve_spec(model):
    """-> (sevennet_model_arg, display_tag, source).

    Maps a model string onto something SevenNetModel accepts (a pretrained name or a checkpoint
    path). A pinned config (demars.yaml) wins over the built-in default resolution, so a long
    campaign uses one declared set of weights instead of whatever the environment happens to
    resolve to; an explicit SPINNER_NANO_CKPT still wins over both, and `source` records which of
    them decided, so an override can never be silent.
    """
    from . import models as _models

    alias = _config_alias(model)
    if alias and not os.environ.get('SPINNER_NANO_CKPT'):
        entry, source = _models.resolve(alias)
        if entry:
            spec = str(entry['spec'])
            # keep the display tag readable: a checkpoint shows as its filename, not its full path
            tag = os.path.basename(spec) if os.sep in spec else spec
            return entry['spec'], tag, source

    low = str(model).lower()
    if low in _NANO_ALIASES:
        return _resolve_nano(), '7net-nano', None
    if low.startswith('7net-nano-'):                 # explicit cutoff variant
        cutoff = low[len('7net-nano-'):]
        return _resolve_nano(cutoff), f'7net-nano-{cutoff}', None
    return model, str(model), None


def _build_model(model, modal, d3):
    """Construct (and cache) the model for this spec.

    d3=True uses sevenn's own `SevenNetD3Model` (Grimme D3, damp_bj/pbe by default,
    batched CUDA kernel) rather than assembling a dispersion term by hand.
    """
    import torch
    from sevenn.torchsim import SevenNetD3Model, SevenNetModel

    spec, tag, source = _resolve_spec(model)
    if modal:
        tag += f'/{modal}'
    device = _pick_device()
    # Record exactly which weights this tier resolved to (provenance() hashes the checkpoint). Done
    # here rather than at call time so the cost is paid once per model, not once per batch.
    from . import models as _models
    accel = _models.accelerator()
    _PROV_CACHE[(str(model), modal, bool(d3))] = _models.provenance(
        spec=str(spec), tag=tag, modal=modal, d3=bool(d3), device=str(device), source=source,
        accel=accel)
    # SevenNet's torch-sim interface is float32-only (it raises on anything else). The accelerator
    # is resolved from one place for BOTH relax paths -- see ase_calculator; two tiers of one entry
    # computed on different kernels, with nothing in the record saying so, is not a saving.
    common = dict(modal=modal, dtype=torch.float32, device=device,
                  **_models.accelerator_kwargs(accel))
    if not d3:
        return SevenNetModel(spec, **common), tag

    # SevenNetD3Model's D3 term is CUDA-only -- fail loudly rather than silently
    # dropping dispersion from an energy the caller asked to include it in.
    if device.type != 'cuda':
        raise RuntimeError(
            f'd3=True needs a CUDA GPU (SevenNetD3Model has no CPU D3 path), but the '
            f'model resolved to {device}. Run without d3, or fix the GPU/torch pairing.')
    return SevenNetD3Model(spec, **common), tag + '+D3'


def use_model(model='7net-nano', modal=None, d3=False, **_ignored):
    """Select the active MLIP.

    Cached by (model, modal, d3): the engine calls this before every relax batch,
    and rebuilding would reload weights from disk each time.
    """
    key = (str(model), modal, bool(d3))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _build_model(model, modal, d3)
    global _ACTIVE
    _ACTIVE = (*_MODEL_CACHE[key], key)
    return _ACTIVE[0]


def ase_calculator(model='7net-nano', modal=None, d3=False):
    """-> (ase.Calculator, provenance) for the SERIAL ASE path.

    torch-sim has no holonomic bond constraint, so a build whose discrete units must be held rigid
    through a pre-relaxation cannot use the batched backend (see calculators.ase_relax_batch). It
    still must not lose its pin or its provenance to get there, so the spec is resolved by exactly
    the same rules as the batched model and the provenance is built the same way.

    Returns the provenance rather than stashing it in _ACTIVE: that slot holds a SevenNetModel for
    the batched path, and putting a calculator in it would be a footgun for any later
    batched_fire_relax call.
    """
    if d3:                                   # reject before resolving anything: an unsupported
        raise NotImplementedError(           # combination should fail fast, not after the work
            'd3=True is not available on the ASE path (SevenNetD3Model is a torch-sim model). '
            'Run without d3, or without rigid-unit constraints.')
    from sevenn.calculator import SevenNetCalculator
    from . import models as _models

    spec, tag, source = _resolve_spec(model)
    if modal:
        tag += f'/{modal}'
    device = _pick_device()
    # Same accelerator as the batched path. It used to be passed only there, so a rigid-unit build
    # -- which is forced onto THIS path -- sampled its ensemble on one set of kernels and took its
    # final tier on another, with nothing in the record distinguishing them.
    accel = _models.accelerator()
    kw = {'device': str(device), **_models.accelerator_kwargs(accel)}
    if modal:
        kw['modal'] = modal
    calc = SevenNetCalculator(spec, **kw)
    return calc, _models.provenance(spec=str(spec), tag=tag, modal=modal, d3=bool(d3),
                                    device=str(device), source=source, accel=accel)


def active_model_tag():
    """-> the display tag of the active model, or None."""
    return None if _ACTIVE is None else _ACTIVE[1]


def active_provenance():
    """-> the compute-provenance block of the ACTIVE model, or None if nothing is loaded.

    This is what gets stamped into a record: the resolved weights, their hash when pinned, what
    decided them, the device, and the versions underneath. Without it a record cannot be checked
    against the run that claims to have produced it."""
    if _ACTIVE is None:
        return None
    return _PROV_CACHE.get(_ACTIVE[2])


# --- autobatcher memory budget -------------------------------------------------
# Adapted from an earlier internal tool that hit the same OOM problem.
# `autobatcher=True` is not enough on its own:
#
#   1. it RE-PROBES the memory limit on every optimize() call, and probing runs real
#      forward passes up to the OOM boundary -- expensive, and DeMARS calls
#      batched_fire_relax once per self-driving round plus once for the final tier;
#   2. the probe is only an estimate, so a later batch whose structures are bigger
#      than the calibration sample can still OOM and kill the whole run.
#
# So: probe ONCE per model, cache it, keep a padding margin, and on a runtime OOM
# shrink the budget and retry instead of failing.
_MAX_MEMORY_PADDING = 0.8    # use 80% of the probed limit -> 20% headroom
_STEPS_MIN = 400             # the historical flat default -- never go below it
_STEPS_MAX = 1000            # bounds the stuck-structure tail; see _auto_steps for what it costs
_OOM_MAX_RETRIES = 3
_OOM_BACKOFF = 0.7           # multiply the budget by this after each OOM
_SCALER_CACHE = {}           # model key -> raw (unpadded) probed max_memory_scaler

_OOM_MARKERS = ('cuda out of memory', 'out of memory', 'failed to allocate')


def _is_oom_error(exc):
    s = str(exc).lower()
    return any(m in s for m in _OOM_MARKERS)


def _max_memory_scaler(model, key, init_state, padding=_MAX_MEMORY_PADDING):
    """-> padded autobatcher budget for `model`, probing once and caching by key."""
    raw = _SCALER_CACHE.get(key)
    if raw is None:
        from torch_sim.autobatching import (calculate_memory_scalers,
                                            estimate_max_memory_scaler)
        scalers = calculate_memory_scalers(init_state, model.memory_scales_with)
        try:
            raw = estimate_max_memory_scaler(init_state, model, scalers)
        except Exception as exc:
            if not _is_oom_error(exc):
                raise
            # The probe climbs batch sizes by 1.6x until a forward pass OOMs -- but it BUILDS each
            # trial batch with `ts.concatenate_states([state] * n)` OUTSIDE its own try/except
            # (torch_sim.autobatching.determine_max_batch_size). So when the allocation that OOMs is
            # the concatenation rather than the forward pass, the error escapes the probe entirely
            # and kills a run whose structures relax perfectly well one at a time. Reproduced on
            # Seen five times, on three GPUs of a 96 GB node including a fully idle one.
            #
            # Fall back to a budget of exactly one largest system. Batches of one are slow, not
            # wrong, and they keep the run alive; anything smaller would trip the refusal for a
            # budget below a single system's own metric.
            raw = float(max(scalers)) / padding
            _warn(f'memory probe raised OOM while building its own trial batch '
                  f'({type(exc).__name__}); falling back to a budget of one system '
                  f'({raw * padding:.3g}) rather than failing the relaxation')
        _SCALER_CACHE[key] = raw
        _log.info('calibrated autobatcher max_memory_scaler=%.3g for %s '
                  '(cached for later batches)', raw, type(model).__name__)
    return raw * padding


def _largest_system_metric(model, init_state):
    """-> the memory metric of the biggest system in `init_state`, in the budget's own units."""
    from torch_sim.autobatching import calculate_memory_scalers
    return float(max(calculate_memory_scalers(init_state, model.memory_scales_with)))


def _optimize_with_oom_fallback(model, key, device, budget, **optimize_kwargs):
    """ts.optimize with an explicit autobatcher budget, shrinking + retrying on OOM.

    -> (final_state, budget_that_worked)
    """
    import torch
    import torch_sim as ts
    from torch_sim.autobatching import InFlightAutoBatcher

    for attempt in range(_OOM_MAX_RETRIES + 1):
        autobatcher = InFlightAutoBatcher(
            model=model, memory_scales_with=model.memory_scales_with,
            max_memory_scaler=budget)
        try:
            state = ts.optimize(model=model, autobatcher=autobatcher, **optimize_kwargs)
            # Persist what actually worked, so the next batch starts from a budget
            # that survived rather than re-hitting the same OOM.
            _SCALER_CACHE[key] = budget / _MAX_MEMORY_PADDING
            return state, budget
        except Exception as exc:
            if not _is_oom_error(exc) or attempt == _OOM_MAX_RETRIES:
                raise
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            budget *= _OOM_BACKOFF
            _warn(f'GPU OOM during relaxation (attempt {attempt + 1}/'
                  f'{_OOM_MAX_RETRIES}); retrying with max_memory_scaler '
                  f'shrunk to {budget:.3g}')


def release_models(keep=None):
    """Drop every cached MLIP except `keep` and hand its GPU memory back. -> n dropped

    DeMARS is TIERED: it samples the ensemble on nano and recomputes the winner on omni-mpa.
    `use_model` caches by (model, modal, d3) so the engine can call it before every batch without
    reloading weights -- which means that when the final tier loads omni, **nano is still resident**,
    together with whatever the caching allocator grew to over the sampling run. Nobody was releasing
    it: `clear_model_cache` existed but had no caller anywhere in the package.

    Measured on a 717-atom cell (100 samples, RTX A5000 24 GB): sampling ran at 9.6 GB, and by
    the time the final tier asked for 3.72 GB the process already held 21.23 GiB of a 23.56 GiB card,
    so it raised torch.OutOfMemoryError. The structure fits; there was no room left for it. That OOM
    is classified as an infra error (correctly -- it is a resource fault, not a claim about the
    structure), so it takes down a run whose 102 relaxed frames were already on disk.

    Not `clear_model_cache()`: that also drops `_PROV_CACHE`, and then the record cannot say which
    weights produced which tier. Provenance is kept here on purpose -- it is a few dicts, and it is
    the thing a reader needs to check a record against the run that claims to have made it.

    The probed budgets of dropped models go too: `_max_memory_scaler` measures the OOM boundary as
    the card stood at probe time, and a value measured with nothing else resident would be wrong on
    reload.
    """
    import gc

    global _ACTIVE
    dropped = [k for k in list(_MODEL_CACHE) if keep is None or k != keep]
    for k in dropped:
        _MODEL_CACHE.pop(k, None)
        _SCALER_CACHE.pop(k, None)
    if _ACTIVE is not None and _ACTIVE[2] in dropped:
        _ACTIVE = None
    if dropped:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return len(dropped)


def clear_model_cache():
    """Drop cached models and free GPU memory (weights are the persistent cost)."""
    global _ACTIVE
    _MODEL_CACHE.clear()
    _SCALER_CACHE.clear()
    _PROV_CACHE.clear()
    _ACTIVE = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _auto_steps(n_atoms):
    """-> optimizer-step budget for a system of `n_atoms`.

    The cap binds ONLY on structures that fail to converge -- FIRE stops at `fmax` whatever the
    ceiling is -- so its single job is deciding how long to hold on to a stuck relaxation. The flat
    400 was too tight (one entry kept 4 of 30 requested configs, a silent convergence failure, not an
    OOM). Scale it with size, then cap it: without the cap one pathological large cell eats 20k
    steps before anyone learns it is stuck, and in a 1000-entry campaign the stuck tail is where
    the wall-clock goes.

    The cap was 5000 and is now 1000. Above ~50 atoms the size term saturates it, so in practice
    every production cell now gets exactly 1000 steps. What that trades:

      * it cannot cost a converged config. FIRE stops at fmax on its own; one 717-atom cell relaxed 6 of 6
        717-atom configs to val=True under the old ceiling, so the ceiling was not binding there.
      * it CAN cost a slow one, and that is that failure mode returning at a higher number.
        The difference from then is that a ceiling hit is no longer silent: ase_relax_batch and the
        batched path both report val=False with the residual force and a warning naming the budget,
        so a too-tight cap shows up as "did not converge in 1000 steps", not as a missing config.

    If entries start reporting step-ceiling non-convergence, this is the number to raise -- and
    `relax_max_force` in the returned Atoms.info says how far off they were.
    """
    return int(np.clip(20 * int(n_atoms), _STEPS_MIN, _STEPS_MAX))


def batched_fire_relax(atoms_list, fmax=0.05, steps=None, *, cell=True,
                       autobatcher=True, **_ignored):
    """Batched variable-cell FIRE relaxation -- the batched-relaxation entry point.

    Parameters
    ----------
    atoms_list : list of ase.Atoms
    fmax : float    force convergence tolerance (eV/A)
    steps : int|None  max optimizer steps; None (default) = `_auto_steps(largest system)`
    cell : bool     relax the cell too (Frechet filter)
    autobatcher : bool
                    True (default) = calibrated autobatcher budget + OOM
                    shrink-and-retry. Forced off on CPU, where torch-sim raises
                    "Memory estimation does not make sense on CPU".

    Returns
    -------
    (E, rel, val) -- see module docstring.
    """
    import torch
    import torch_sim as ts

    atoms_list = list(atoms_list)
    if not atoms_list:
        return np.zeros(0), [], np.zeros(0, dtype=bool)
    if steps is None:                   # one budget for the batch -> size it by the largest member
        steps = _auto_steps(max(len(a) for a in atoms_list))
    if _ACTIVE is None:
        use_model()                     # nano, no modal, no D3 -- the default tier
    model, _tag, key = _ACTIVE
    device = getattr(model, 'device', torch.device('cpu'))

    # Memory estimation is CUDA-only in torch-sim; on CPU it hard-raises. autobatcher
    # =False does not mean "one giant batch" -- optimize() still evicts converged
    # systems, it just assumes unbounded memory.
    if device.type != 'cuda':
        autobatcher = False

    # cell_filter is a fire_init argument, so it must go through init_kwargs;
    # optimize()'s **optimizer_kwargs are forwarded to the step function instead.
    opt_kw = dict(
        optimizer=ts.Optimizer.fire,
        convergence_fn=ts.generate_force_convergence_fn(force_tol=fmax),
        max_steps=steps,
        init_kwargs={'cell_filter': ts.CellFilter.frechet} if cell else {},
    )

    try:
        if autobatcher:
            # Build the state once and reuse it for calibration + the run, so the
            # probe measures the batch we are actually about to relax.
            init_state = ts.initialize_state(atoms_list, device, model.dtype)
            budget = _max_memory_scaler(model, key, init_state)
            # The probe climbs to the OOM boundary and comes back; times the padding, the budget
            # can land BELOW a single system's own metric -- and then torch-sim refuses the whole
            # batch. There is nothing to batch in that case, so relax without the autobatcher
            # rather than report a relaxation failure for a memory-probe artefact (this is the
            # path that left `final_MAR: {}` on structures that relax cleanly when asked directly).
            need = _largest_system_metric(model, init_state)
            if need > budget:
                _warn(f'autobatcher budget {budget:.3g} is below the largest system\'s own '
                      f'metric {need:.3g} (memory probe came back too small); relaxing this '
                      f'batch of {len(atoms_list)} with autobatcher=False instead')
                final = ts.optimize(system=atoms_list, model=model,
                                    autobatcher=False, **opt_kw)
            else:
                final, _ = _optimize_with_oom_fallback(
                    model, key, device, budget, system=init_state, **opt_kw)
        else:
            final = ts.optimize(system=atoms_list, model=model,
                                autobatcher=False, **opt_kw)
    except Exception as exc:
        # An unusable GPU or missing dependency is NOT a relaxation failure -- let it
        # propagate, or the engine records "nothing relaxed" for an environment fault.
        if _is_infra_error(exc):
            raise
        # A genuine batch failure maps onto that backend's per-config invalid contract,
        # but never silently: the engine only sees val=False.
        _warn(f'batched_fire_relax failed for this batch of {len(atoms_list)} '
              f'configs: {type(exc).__name__}: {exc}')
        return (np.full(len(atoms_list), np.nan), list(atoms_list),
                np.zeros(len(atoms_list), dtype=bool))

    # Order is guaranteed: optimize() calls autobatcher.restore_original_order().
    rel = ts.io.state_to_atoms(final)
    E = np.asarray(final.energy.detach().to(torch.float64).cpu().numpy(), dtype=float)
    maxf = ts.system_wise_max_force(final).detach().to(torch.float64).cpu().numpy()
    val = np.asarray((maxf < fmax) & np.isfinite(E), dtype=bool)

    # Hand back the input geometry wherever the relax did not converge, so callers
    # never consume a half-relaxed cell as if it were a minimum.
    for i, ok in enumerate(val):
        if not ok:
            E[i] = np.nan
            rel[i] = atoms_list[i]

    # SAY that the step ceiling was hit. val=False alone reads as "this structure does not
    # relax", which is a statement about the CHEMISTRY; running out of `steps` is a statement
    # about the BUDGET, and the two must not be conflated (the caller may want more steps, not
    # a different structure). Stamped on the atoms as well as warned, because a warning is lost
    # the moment the run dir outlives the terminal.
    nbad = int((~val).sum())
    if nbad:
        for i, ok in enumerate(val):
            if not ok:
                rel[i].info['relax_converged'] = False
                rel[i].info['relax_max_force'] = float(maxf[i])
                rel[i].info['relax_steps_budget'] = int(steps)
        _warn(f'{nbad}/{len(atoms_list)} config(s) did not reach fmax={fmax} within '
              f'steps={steps}; residual max force up to {float(np.nanmax(maxf)):.3f} eV/A. '
              f'Reported as invalid -- a step-ceiling hit is NOT evidence that the structure '
              f'cannot relax.')
    return E, rel, val
