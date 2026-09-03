"""Shared pytest fixtures for the demars-core suite."""
import os
import pathlib
import sys

import pytest

# --- environment, established before any test module imports demars_core --------------------------
# Two things are read at IMPORT time, so a fixture is too late for both:
#   * `mar_evidence` exec's DEMARS_ORDERED_LOOKUP to build `_sib`, and that code reads ICSD_DB_DIR.
#     Four test modules import `mar_evidence` at module level, so `_sib` is decided at collection.
#     Without this the whole ordered-sibling path was skipped by default (`skipif(_sib is None)`) --
#     only the `tools/` wrappers, which import `_icsd_env` themselves, ever exercised it. The
#     taxonomy's central question runs through that code, so it must not be opt-in.
#   * `demars_core.models` discovers the config relative to the CWD, so running pytest from
#     `demars-core/` rather than the repo root left `paths.checkpoint_dir` unset and skipped every
#     MLIP test with "7net-nano unavailable".
# Both degrade cleanly when the file or DB is absent: the variables stay unset and the skips return.
# `_icsd_env` is reused rather than re-derived -- one definition of the discovery order, not two.
_REPO = pathlib.Path(__file__).resolve().parents[2]

if (_REPO / 'demars.yaml').is_file():
    os.environ.setdefault('DEMARS_CONFIG', str(_REPO / 'demars.yaml'))

try:
    sys.path.insert(0, str(_REPO / 'tools'))
    import _icsd_env                    # noqa: F401  sets DEMARS_ORDERED_LOOKUP + ICSD_DB_DIR
except Exception:                       # pragma: no cover -- may run without the repo's tools/
    pass
finally:
    sys.path.remove(str(_REPO / 'tools'))


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'mlip: test needs a universal-MLIP backend loaded (relaxation); skipped if unavailable')


@pytest.fixture(autouse=True)
def _fresh_config_cache():
    """Re-read the deployment config around every test.

    `demars_core.models` caches the parsed config per process, so a test that points DEMARS_CONFIG at
    a temporary file would otherwise leak that state into unrelated tests in other modules -- which
    is exactly the kind of order-dependent failure that wastes an afternoon.
    """
    from demars_core import models
    models.load_config(reload=True)
    yield
    models.load_config(reload=True)


@pytest.fixture(scope='session')
def calculator():
    """The MLIP backend used by the end-to-end (relaxation) tests.

    Defaults to SevenNet '7net-nano' (native batched path, torch-sim by default).
    If the backend is not usable the test is skipped rather than failed. To test a
    different universal MLIP, return your own ASE Calculator instance here (e.g. a
    MACE or CHGNet calculator) — demars_core routes any ASE calculator through its
    generic relaxer.
    """
    try:
        from demars_core import _torchsim as backend
    except Exception as e:                       # pragma: no cover
        pytest.skip(f'no MLIP backend available ({type(e).__name__}: {e})')
    # nano ships as a separate checkpoint, so an install can be complete yet
    # unable to locate it -- skip instead of failing every relaxation test.
    try:
        backend.use_model(model='7net-nano')
    except Exception as e:                       # pragma: no cover
        pytest.skip(f'7net-nano unavailable via {backend.__name__} '
                    f'({type(e).__name__}: {e})')
    return '7net-nano'
