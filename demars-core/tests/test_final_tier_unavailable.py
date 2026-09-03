"""A final tier that could not run must say so, not look like one that was never asked for.

Three outcomes have to stay distinguishable:

    final_MAR is None             no --final; nobody asked
    final_MAR['available'] False  asked for, could not run, `reason` says why
    final_MAR['E_final_...']      it ran

The middle one had no representation. `_final_omni` returned None when the relaxation failed, and
raised when it OOM'd -- so "could not" collapsed into "did not ask", or took the whole run down with
it. One entry lost a completed 102-frame ensemble that way: its 717-atom cell does not fit on a
24 GB card at the omni-mpa tier, which is a fact about the card, not about the structure.
"""
import numpy as np
import pytest
from ase import Atoms

from demars_core import api


def _built(n=2):
    at = Atoms('NaCl', positions=[[0, 0, 0], [2.8, 0, 0]], cell=[6.0] * 3, pbc=True)
    rel = [at.copy() for _ in range(n)]
    samples = [{'label': f'r{i}', 'valid': True, 'E_per_atom': -5.0 - i} for i in range(n)]
    return (rel, rel, [True] * n, samples, {})


@pytest.fixture
def _no_gpu_for_final(monkeypatch):
    """The sampling tier already succeeded; only the final tier fails."""
    monkeypatch.setattr(api.backend, 'release_models', lambda *a, **k: 0)
    monkeypatch.setattr(api.backend, 'use_model', lambda *a, **k: None)


def test_an_oom_at_the_final_tier_no_longer_kills_the_run(_no_gpu_for_final, monkeypatch):
    def _oom(*a, **k):
        raise MemoryError('CUDA out of memory. Tried to allocate 3.72 GiB')

    monkeypatch.setattr(api.backend, 'batched_fire_relax', _oom)
    fm, struct = api._final_omni(_built())
    assert fm['available'] is False and fm['requested'] is True
    assert 'out of memory' in fm['reason'].lower()
    assert struct is None


def test_the_reason_is_verbatim_so_the_cause_stays_diagnosable(_no_gpu_for_final, monkeypatch):
    monkeypatch.setattr(api.backend, 'batched_fire_relax',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('probe said 5.75e+04')))
    fm, _ = api._final_omni(_built())
    assert 'RuntimeError' in fm['reason'] and '5.75e+04' in fm['reason']


def test_non_convergence_is_also_reported_not_nulled(_no_gpu_for_final, monkeypatch):
    monkeypatch.setattr(api.backend, 'batched_fire_relax',
                        lambda confs, **k: (np.array([np.nan]), list(confs), np.array([False])))
    fm, struct = api._final_omni(_built())
    assert fm is not None, 'a non-converged final tier returned None -- indistinguishable from "not requested"'
    assert fm['available'] is False and struct is None


def test_a_successful_final_tier_is_unchanged(_no_gpu_for_final, monkeypatch):
    def _ok(confs, **k):
        return np.array([-20.0]), list(confs), np.array([True])

    monkeypatch.setattr(api.backend, 'batched_fire_relax', _ok)
    fm, struct = api._final_omni(_built())
    assert 'available' not in fm and fm['E_final_per_atom'] == -10.0
    assert struct is not None


def test_the_summary_does_not_print_E_none():
    from demars_core.record import MARRecord

    # the representative line prints its own `E=...`, so fill it in -- otherwise this test fails on
    # that line and says nothing about the final tier, which is what it is here to check.
    rec = MARRecord(source='x', status='de-averaged', calculator='sevennet',
                    distribution={'n_relaxed': 1, 'n_total': 1, 'spread_meV': 0, 'std_meV': 0,
                                  'representative': {'label': 'r0', 'E_per_atom': -5.0,
                                                     'charge': 0, 'min_dist_A': 2.1, 'n_atoms': 2}},
                    final_MAR={'available': False, 'requested': True, 'reason': 'no room on a 24 GB card'})
    out = rec.summary()
    final = out.splitlines()[-1]
    assert 'NOT AVAILABLE' in final and 'no room on a 24 GB card' in final
    assert 'E=None' not in final


def test_never_requested_still_prints_nothing():
    from demars_core.record import _final_line
    assert _final_line(None) == ''


def test_the_engine_path_reports_it_too():
    """`deaverage()` is not the only way into the final tier.

    mar_engine has its own `--final` block, and it used to leave `final_MAR` unset on failure --
    which tools/demars_engine.py then coerces to `{}`, the empty final_MAR of D10. Both entry
    points have to make the same three states distinguishable.
    """
    import inspect

    from demars_core._engine import mar_engine

    src = inspect.getsource(mar_engine)
    i = src.index("if '--final' in args:")
    block = src[i:i + 2000]
    body = '\n'.join(ln.split('#', 1)[0] for ln in block.splitlines())   # comments are not code

    assert 'S.release_models()' in body, 'the engine path loads omni without freeing the sampling tier'
    assert body.index('S.release_models()') < body.index("use_model(model='7net-omni'"), \
        'the sampling tier is freed after omni is loaded, which is too late'
    assert "'available': False" in body or '"available": False' in body, \
        'a failed final tier leaves final_MAR unset, which reads as "never requested"'
    assert 'except Exception' in body, 'an OOM in the engine path still takes the whole run down'
