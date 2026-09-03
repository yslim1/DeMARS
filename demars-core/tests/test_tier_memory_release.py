"""The sampling tier must hand its GPU memory back before the final tier loads.

DeMARS is tiered: sample on nano, recompute the winner on omni-mpa. `use_model` caches models so
the engine can call it before every batch, which means the second tier used to load while the first
was still resident -- on top of whatever the caching allocator had grown to over the sampling run.
Nothing released it; `clear_model_cache` existed with no caller in the package.

Measured (717 atoms, 100 samples, RTX A5000 24 GB): sampling ran at 9.6 GB, then the final tier
asked for 3.72 GB while the process already held 21.23 of 23.56 GiB and raised
torch.OutOfMemoryError. That is an infra error by design, so it killed a run whose 102 relaxed
frames were already checkpointed to disk. The structure fits; there was no room left for it.
"""
import inspect

import pytest

from demars_core import _torchsim as T


@pytest.fixture(autouse=True)
def _clean_caches():
    saved = (dict(T._MODEL_CACHE), dict(T._SCALER_CACHE), dict(T._PROV_CACHE), T._ACTIVE)
    T._MODEL_CACHE.clear(); T._SCALER_CACHE.clear(); T._PROV_CACHE.clear()
    T._ACTIVE = None
    yield
    T._MODEL_CACHE.clear(); T._MODEL_CACHE.update(saved[0])
    T._SCALER_CACHE.clear(); T._SCALER_CACHE.update(saved[1])
    T._PROV_CACHE.clear(); T._PROV_CACHE.update(saved[2])
    T._ACTIVE = saved[3]


def _load(key, model=object()):
    T._MODEL_CACHE[key] = (model, str(key))
    T._SCALER_CACHE[key] = 12345.0
    T._PROV_CACHE[key] = {'model_tag': str(key), 'checkpoint_sha256': 'abc'}
    T._ACTIVE = (model, str(key), key)


def test_the_sampling_model_is_dropped():
    nano = ('7net-nano', None, False)
    _load(nano)
    assert T.release_models() == 1
    assert nano not in T._MODEL_CACHE


def test_provenance_survives_so_the_record_can_still_name_the_weights():
    """clear_model_cache() drops _PROV_CACHE; using it here would blank the record's mlip block."""
    nano = ('7net-nano', None, False)
    _load(nano)
    T.release_models()
    assert T._PROV_CACHE[nano]['checkpoint_sha256'] == 'abc'


def test_the_probed_budget_goes_with_the_model():
    """A budget measured with nothing else resident would be wrong after a reload."""
    nano = ('7net-nano', None, False)
    _load(nano)
    T.release_models()
    assert nano not in T._SCALER_CACHE


def test_keep_spares_exactly_one_model():
    nano, omni = ('7net-nano', None, False), ('7net-omni', 'mpa', False)
    _load(nano); _load(omni)
    assert T.release_models(keep=omni) == 1
    assert omni in T._MODEL_CACHE and nano not in T._MODEL_CACHE


def test_the_active_slot_is_not_left_pointing_at_a_dropped_model():
    """_ACTIVE feeds active_provenance(); a dangling entry would stamp a released tier."""
    nano = ('7net-nano', None, False)
    _load(nano)
    T.release_models()
    assert T._ACTIVE is None


def test_releasing_nothing_is_a_no_op():
    assert T.release_models() == 0


def test_the_final_tier_releases_before_it_loads():
    """The wiring, not just the helper: _final_omni must free the sampling tier FIRST."""
    from demars_core import api

    # Match the CALL, not the word: a comment mentioning release_models would satisfy a
    # substring test while the call itself was gone (this test was written that way first,
    # and a mutation that deleted the call passed it).
    src = '\n'.join(ln.split('#', 1)[0] for ln in inspect.getsource(api._final_omni).splitlines())
    assert 'backend.release_models()' in src, '_final_omni does not release the sampling tier'
    assert src.index('backend.release_models()') < src.index("use_model(model='7net-omni'"), \
        'the sampling tier is released after the final tier is loaded, which is too late'
