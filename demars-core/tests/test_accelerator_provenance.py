"""The fused-kernel backend is a declared, stamped setting -- not a shell variable.

cuEquivariance (and friends) do not touch the checkpoint, so `checkpoint_sha256` matches while the
numbers move in the last float32 digits. `tools/demars_reference.py` compares energies exactly WHEN
the sha matches, so without a separate stamp a kernel change is indistinguishable from a chemistry
change -- the same laundering of an environment fact into a chemical claim as D10 and the ASE drift.
"""
import pytest

from demars_core import models


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in ('DEMARS_ACCELERATOR', 'DEMARS_ENABLE_CUEQ', 'DEMARS_CONFIG'):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(models, '_cache', {'path': None, 'data': {}, 'loaded': True})


def test_the_default_is_off():
    assert models.accelerator() == 'none'
    assert models.accelerator_kwargs('none') == {}


def test_the_config_file_declares_it(monkeypatch):
    monkeypatch.setattr(models, '_cache',
                        {'path': '/x/demars.yaml', 'data': {'compute': {'accelerator': 'cueq'}},
                         'loaded': True})
    assert models.accelerator() == 'cueq'


def test_the_environment_still_overrides_the_file(monkeypatch):
    monkeypatch.setattr(models, '_cache',
                        {'path': '/x/demars.yaml', 'data': {'compute': {'accelerator': 'cueq'}},
                         'loaded': True})
    monkeypatch.setenv('DEMARS_ACCELERATOR', 'none')
    assert models.accelerator() == 'none'


def test_the_old_switch_keeps_working(monkeypatch):
    """DEMARS_ENABLE_CUEQ=1 predates the config key; a live campaign must not change meaning."""
    monkeypatch.setenv('DEMARS_ENABLE_CUEQ', '1')
    assert models.accelerator() == 'cueq'


def test_an_unknown_backend_is_refused_not_guessed(monkeypatch):
    monkeypatch.setenv('DEMARS_ACCELERATOR', 'cuquivariance')      # typo
    with pytest.raises(ValueError, match='unknown accelerator'):
        models.accelerator()


def test_a_missing_backend_is_an_error_not_a_silent_fallback(monkeypatch):
    """Falling back quietly would put 'ran on cueq' in a record produced without it."""
    real = __import__

    def _no_cueq(name, *a, **k):
        if name == 'cuequivariance_torch':
            raise ImportError('no module named cuequivariance_torch')
        return real(name, *a, **k)

    monkeypatch.setattr('builtins.__import__', _no_cueq)
    with pytest.raises(RuntimeError, match='cuequivariance_torch'):
        models.accelerator_kwargs('cueq')


def test_the_kwargs_are_what_sevennet_actually_takes():
    import inspect

    sevenn_calc = pytest.importorskip('sevenn.calculator')
    params = inspect.signature(sevenn_calc.SevenNetCalculator.__init__).parameters
    for name, (kw, _mod) in models._ACCELERATORS.items():
        for k in kw:
            assert k in params, f'{name}: SevenNetCalculator has no {k}'


def test_the_record_says_which_kernels_produced_it():
    p = models.provenance(spec='7net-nano', tag='7net-nano', device='cuda', accel='none')
    assert p['accelerator'] == 'none'
    p = models.provenance(spec='7net-nano', tag='7net-nano', device='cuda', accel='cueq')
    assert p['accelerator'] == 'cueq'
    # and the accelerator's own version travels with it -- the kernels are versioned software too
    assert 'cuequivariance_torch' in p['versions']


def test_both_relax_paths_read_the_same_setting():
    """A rigid-unit build samples on the ASE path and finalises on the batched one. If only one
    honoured the setting, one entry's two tiers would run on different kernels, unrecorded."""
    import demars_core._torchsim as T
    src = __import__('inspect').getsource(T)
    batched = src.index('def _build_model')
    serial = src.index('def ase_calculator')
    for start, end, where in ((batched, serial, '_build_model'),
                              (serial, len(src), 'ase_calculator')):
        assert 'accelerator_kwargs' in src[start:end], f'{where} does not apply the accelerator'
        assert 'accel=accel' in src[start:end], f'{where} does not stamp the accelerator'
    assert 'DEMARS_ENABLE_CUEQ' not in src, 'the env var is read in models.accelerator(), not here'
