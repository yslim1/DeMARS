"""Where a record stores the paths to its own structures (`mar_record.portable_path`).

A record used to keep whatever path the engine emitted, which was relative to the working directory
of the run. Move the tree and every one dangles — measured on an archived 50-record campaign, 49 of
50 no longer resolved, and with them the connectivity gate's only route to the shipped frame. Two
rules: store relative to the record when the file lives beside it, and resolve by re-rooting the
tail under the record's own directory when the recorded path is already stale.

No MLIP, no GPU, no network.
"""
import json
import os

from demars_core._engine.mar_record import build_record, portable_path


def _tree(tmp_path, entry='icsd_000001'):
    """A run directory shaped like the pipeline's: the record beside `_work/<tag>/`."""
    d = tmp_path / entry
    (d / '_work' / 'r0').mkdir(parents=True)
    struct = d / '_work' / 'r0' / 'representative.cif'
    struct.write_text('data_x\n')
    (d / f'{entry}.cif').write_text('data_in\n')
    return d, struct


# ---- storing ----------------------------------------------------------------

def test_a_file_beside_the_record_is_stored_relative_to_it():
    """So the whole entry directory can be moved or renamed and stay internally consistent."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'icsd_1')
        os.makedirs(os.path.join(out, '_work', 'r0'))
        f = os.path.join(out, '_work', 'r0', 'rep.cif')
        open(f, 'w').write('x')
        stored, found = portable_path(f, out)
        assert stored == os.path.join('_work', 'r0', 'rep.cif'), stored
        assert found == os.path.abspath(f)


def test_a_file_outside_the_record_stays_absolute():
    """A relative path to a shared corpus would only survive moving everything at once, so it is
    not an improvement over the absolute one."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'run'); os.makedirs(out)
        elsewhere = os.path.join(tmp, 'corpus'); os.makedirs(elsewhere)
        f = os.path.join(elsewhere, 'in.cif'); open(f, 'w').write('x')
        stored, found = portable_path(f, out)
        assert os.path.isabs(stored), stored
        assert found == os.path.abspath(f)


def test_no_output_directory_leaves_the_value_alone():
    """`build_record` is called without `--out` in tests and by other callers; it must not rewrite
    paths it has no directory to be relative to."""
    stored, found = portable_path('some/where/rep.cif', None)
    assert stored == 'some/where/rep.cif' and found is None


def test_a_missing_file_is_reported_as_unresolved_not_invented():
    stored, found = portable_path('nowhere/rep.cif', None)
    assert found is None
    assert stored == 'nowhere/rep.cif', 'an unresolvable path is left as recorded, not mangled'


# ---- healing a tree that already moved --------------------------------------

def test_a_moved_tree_is_found_again_by_its_tail():
    """The archived campaign's records say `runs/<entry>/_work/r0/...` and the tree now lives
    somewhere else entirely."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'icsd_9')
        os.makedirs(os.path.join(out, '_work', 'r0'))
        f = os.path.join(out, '_work', 'r0', 'rep.cif'); open(f, 'w').write('x')
        stale = 'runs/icsd_9/_work/r0/rep.cif'
        stored, found = portable_path(stale, out)
        assert found == os.path.abspath(f), found
        assert stored == os.path.join('_work', 'r0', 'rep.cif'), stored


def test_the_entry_directory_may_even_have_been_renamed():
    """Only the tail below the entry has to match, so an archive that renames its entries still
    resolves."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'renamed')
        os.makedirs(os.path.join(out, '_work', 'r0'))
        f = os.path.join(out, '_work', 'r0', 'rep.cif'); open(f, 'w').write('x')
        stored, found = portable_path('runs/icsd_9/_work/r0/rep.cif', out)
        assert found == os.path.abspath(f)
        assert stored == os.path.join('_work', 'r0', 'rep.cif')


def test_a_path_that_matches_nothing_under_the_directory_stays_unresolved():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'run'); os.makedirs(out)
        stored, found = portable_path('runs/x/_work/r0/rep.cif', out)
        assert found is None
        assert stored == 'runs/x/_work/r0/rep.cif'


# ---- through the record ------------------------------------------------------

def _evidence():
    from demars_core._engine import mar_evidence as MEV
    fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'fixtures', 'cod_1544358.cif')
    with open(fixture, encoding='utf-8') as fh:
        return MEV.evidence_from_text(fh.read(), iid=None, search_siblings=False)


def test_the_record_stores_the_relative_path_and_the_gate_sees_the_real_file(tmp_path):
    """Both halves in one: what lands in `record.json` is portable, and the connectivity gate still
    got an absolute path it could actually read."""
    d, struct = _tree(tmp_path)
    engine = {'engine': {}, 'distribution': {},
              'ensemble_files': {'representative': f'runs/icsd_000001/_work/r0/{struct.name}'}}
    rec = build_record(None, engine, {}, evidence=_evidence(), out_dir=str(d))

    assert rec['representative']['file'] == os.path.join('_work', 'r0', struct.name)
    assert os.path.exists(os.path.join(str(d), rec['representative']['file']))
    # the gate ran on it rather than reporting the frame unreadable
    assert rec['gates']['connectivity']['state'] != 'not_run', rec['gates']['connectivity']


def test_without_out_dir_the_record_keeps_the_engines_own_path(tmp_path):
    """Backwards compatible: a caller that gives no output directory sees no rewriting."""
    engine = {'engine': {}, 'distribution': {},
              'ensemble_files': {'representative': 'runs/icsd_000001/_work/r0/rep.cif'}}
    rec = build_record(None, engine, {}, evidence=_evidence())
    assert rec['representative']['file'] == 'runs/icsd_000001/_work/r0/rep.cif'


def test_the_record_is_json_serialisable_with_the_rewritten_paths(tmp_path):
    d, struct = _tree(tmp_path, entry='icsd_000002')
    engine = {'engine': {}, 'distribution': {},
              'ensemble_files': {'representative': str(struct)}}
    rec = build_record(None, engine, {}, evidence=_evidence(), out_dir=str(d))
    assert json.loads(json.dumps(rec, default=str))['representative']['file'] \
        == os.path.join('_work', 'r0', struct.name)
