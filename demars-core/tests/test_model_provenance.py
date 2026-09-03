"""Model pinning and compute provenance (no MLIP / no GPU).

A thousand-structure campaign is only reproducible if two things hold: the weights are DECLARED in
one place, and whatever was actually used is RECORDED in every result. These tests guard both, plus
the precedence rule that keeps pinning from surprising a caller who asked for something specific.
"""
import json
import os

import pytest

from demars_core import models
from demars_core import _torchsim as backend


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    """A models config pinning both tiers, installed via $DEMARS_CONFIG."""
    cfg = tmp_path / 'models.json'
    ckpt = tmp_path / 'checkpoint_pinned.pth'
    ckpt.write_bytes(b'not really weights, but a real file to hash')
    # a cutoff variant on disk, so an EXPLICIT request has somewhere to resolve to and the
    # precedence rule can be tested rather than masked by a missing file
    (tmp_path / 'checkpoint_7net_nano_5.0.pth').write_bytes(b'the 5.0 variant')
    cfg.write_text(json.dumps({
        'models': {'nano': {'spec': str(ckpt), 'modal': None},
                   'omni': {'spec': '7net-omni', 'modal': 'mpa'}},
        'paths': {'checkpoint_dir': str(tmp_path)}}))
    monkeypatch.setenv('DEMARS_CONFIG', str(cfg))
    monkeypatch.delenv('SPINNER_NANO_CKPT', raising=False)
    monkeypatch.delenv('DEMARS_CKPT_DIR', raising=False)
    models.load_config(reload=True)
    yield {'cfg': cfg, 'ckpt': ckpt, 'dir': tmp_path}
    models.load_config(reload=True)


@pytest.fixture
def unpinned(tmp_path, monkeypatch):
    """No config anywhere -- the package must behave exactly as it did before pinning existed."""
    monkeypatch.delenv('DEMARS_CONFIG', raising=False)
    monkeypatch.delenv('SPINNER_NANO_CKPT', raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))
    models.load_config(reload=True)
    yield
    models.load_config(reload=True)


def test_config_is_discovered_and_parsed(pinned):
    assert models.config_path() == str(pinned['cfg'])
    entry, source = models.resolve('nano')
    assert entry['spec'] == str(pinned['ckpt'])
    assert source == str(pinned['cfg'])


def test_bare_alias_is_pinned(pinned):
    """'sevennet' / '7net-nano' carry no cutoff of their own, so the config decides."""
    for alias in ('sevennet', '7net-nano'):
        spec, tag, source = backend._resolve_spec(alias)
        assert str(spec) == str(pinned['ckpt']), alias
        assert source == str(pinned['cfg'])
        assert os.sep not in tag, 'the display tag should be a filename, not a full path'


def test_explicit_request_is_never_overridden(pinned):
    """A caller asking for a specific cutoff or a specific checkpoint must get exactly that --
    otherwise pinning would silently swap the model out from under them."""
    spec, _, source = backend._resolve_spec('7net-nano-5.0')
    assert '5.0' in str(spec) and str(spec) != str(pinned['ckpt']) and source is None
    spec, _, source = backend._resolve_spec('/somewhere/explicit.pth')
    assert spec == '/somewhere/explicit.pth' and source is None


def test_omni_tier_is_pinnable(pinned):
    spec, _, source = backend._resolve_spec('7net-omni')
    assert spec == '7net-omni' and source == str(pinned['cfg'])


def test_no_config_means_nothing_is_pinned(unpinned):
    assert models.config_path() is None
    assert models.resolve('nano') == (None, None)


def test_missing_checkpoint_dir_fails_with_an_actionable_message(unpinned, monkeypatch):
    """With nothing configured the package must say so, not quietly reach for some other machine's
    directory -- the hardcoded fallback that used to live here."""
    for var in ('SPINNER_NANO_CKPT', 'DEMARS_CKPT_DIR'):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(FileNotFoundError, match='checkpoint_dir'):
        backend._resolve_spec('sevennet')


def test_malformed_config_is_a_hard_error(tmp_path, monkeypatch):
    """A campaign that believes it is pinned but is not is worse than one that refuses to start."""
    bad = tmp_path / 'models.json'
    bad.write_text('{not json')
    monkeypatch.setenv('DEMARS_CONFIG', str(bad))
    with pytest.raises(ValueError):
        models.load_config(reload=True)
    monkeypatch.delenv('DEMARS_CONFIG')
    models.load_config(reload=True)          # restore a clean cache for the rest of the suite


def test_hash_is_recorded_automatically(pinned):
    """The hash is computed from the file, not declared in the config: nothing to fill in by hand and
    no way to forget. A spec that is a pretrained NAME has no local file, so it hashes to None."""
    spec, _, _ = backend._resolve_spec('sevennet')
    p = models.provenance(spec=str(spec))
    assert p['checkpoint_sha256'] == models.hash_file(str(pinned['ckpt']))
    assert len(p['checkpoint_sha256']) == 64
    assert models.provenance(spec='7net-omni')['checkpoint_sha256'] is None


def test_hash_detects_changed_weights(tmp_path):
    """Two records whose checkpoint paths match but whose hashes differ were produced by different
    weights -- exactly the drift a path alone cannot show."""
    a, b = tmp_path / 'a.pth', tmp_path / 'b.pth'
    a.write_bytes(b'weights v1'); b.write_bytes(b'weights v2')
    assert models.hash_file(str(a)) != models.hash_file(str(b))


def test_provenance_carries_what_a_reader_needs():
    p = models.provenance(spec='/x/ckpt.pth', tag='ckpt.pth', modal='mpa', d3=False,
                          device='cuda', source='/cfg.json')
    for k in ('model_spec', 'model_tag', 'modal', 'd3', 'checkpoint_sha256',
              'resolved_from', 'config_file', 'device', 'versions'):
        assert k in p
    assert p['versions']['demars_core']
    assert p['resolved_from'] == '/cfg.json'


def test_provenance_names_the_default_source_when_unpinned(unpinned):
    assert models.provenance()['resolved_from'] == 'built-in defaults'


def test_yaml_config_is_read(tmp_path, monkeypatch):
    """YAML is the documented format precisely because a deployment config needs comments."""
    cfg = tmp_path / 'demars.yaml'
    cfg.write_text('# a comment the JSON format could not carry\n'
                   'models:\n'
                   '  omni:\n'
                   '    spec: 7net-omni\n'
                   '    modal: mpa\n'
                   'paths:\n'
                   '  checkpoint_dir: /ckpts\n'
                   '  icsd_db: /icsd\n'
                   'materials_project:\n'
                   '  api_key: from-file\n')
    monkeypatch.setenv('DEMARS_CONFIG', str(cfg))
    for var in ('DEMARS_CKPT_DIR', 'ICSD_DB_DIR', 'MP_API_KEY'):
        monkeypatch.delenv(var, raising=False)
    models.load_config(reload=True)
    entry, _ = models.resolve('omni')
    assert entry == {'spec': '7net-omni', 'modal': 'mpa'}
    assert models.checkpoint_dir() == '/ckpts'
    assert models.icsd_db_dir() == '/icsd'
    assert models.mp_api_key() == 'from-file'
    monkeypatch.delenv('DEMARS_CONFIG')
    models.load_config(reload=True)


def test_environment_overrides_the_config_paths(tmp_path, monkeypatch):
    """The environment is the more specific signal and must win -- and for the MP key it is also
    the safer place to keep it, since a key in a file is easy to commit by accident."""
    cfg = tmp_path / 'demars.yaml'
    cfg.write_text('paths: {checkpoint_dir: /from-file, icsd_db: /from-file}\n'
                   'materials_project: {api_key: from-file}\n')
    monkeypatch.setenv('DEMARS_CONFIG', str(cfg))
    monkeypatch.setenv('DEMARS_CKPT_DIR', '/from-env')
    monkeypatch.setenv('ICSD_DB_DIR', '/from-env')
    monkeypatch.setenv('MP_API_KEY', 'from-env')
    models.load_config(reload=True)
    assert models.checkpoint_dir() == '/from-env'
    assert models.icsd_db_dir() == '/from-env'
    assert models.mp_api_key() == 'from-env'
    monkeypatch.delenv('DEMARS_CONFIG')
    models.load_config(reload=True)


def test_paths_are_absent_rather_than_hardcoded(unpinned):
    """With no config and no environment the accessors report nothing -- the package must not fall
    back to any one machine's directories."""
    for var in ('DEMARS_CKPT_DIR', 'ICSD_DB_DIR', 'MP_API_KEY'):
        os.environ.pop(var, None)
    assert models.checkpoint_dir() is None
    assert models.icsd_db_dir() is None
    assert models.mp_api_key() is None


def test_record_carries_the_mlip_block():
    """MARRecord must have somewhere to put it, and it must survive serialisation."""
    from demars_core.record import MARRecord
    rec = MARRecord(source='x', status='de-averaged', calculator='sevennet')
    rec.mlip = models.provenance(spec='s', tag='t')
    assert rec.to_dict()['mlip']['model_spec'] == 's'
