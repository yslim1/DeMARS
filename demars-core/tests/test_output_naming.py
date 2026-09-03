"""One name, one schema.

`deaverage()` and the stage-6 tools layer both write into the same run directory, and both used to
call their output `record.json` -- two different schemas under one name. Whichever ran last defined
what "the record" was, and a consumer that read the wrong one got nulls where it expected judgement
(the gallery did exactly this: it printed "single config -- no energy distribution to plot" for a
30-config ensemble, because a mar-1.0 record has no `distribution` key at all).

No MLIP and no GPU: `MARRecord.write_ensemble` is called directly on a hand-built record.
"""
import json
import warnings

import numpy as np
import pytest
from ase import Atoms

from demars_core.record import MARRecord


def _one_sample(tmp_path):
    at = Atoms('Si2', positions=[[0, 0, 0], [2.3, 0, 0]], cell=[8.0] * 3, pbc=True)
    rec = MARRecord(source='fixture.cif', status='de-averaged', calculator='stub')
    rec.distribution = {'n_relaxed': 1, 'ground_E': -1.0, 'spread_meV': 0.0}
    # (confs, relaxed, valid, samples, engine_info) -- the 5-tuple write_ensemble consumes
    rec._built = ([at], [at], [True],
                  [{'label': 'rand0', 'mode': 'sqs', 'E_per_atom': -0.5,
                    'charge': 0.0, 'valid': True}], {})
    return rec


def test_the_deterministic_output_does_not_claim_the_stage6_name(tmp_path):
    rec = _one_sample(tmp_path)
    paths = rec.write_ensemble(str(tmp_path))

    assert (tmp_path / 'deaverage_output.json').is_file()
    assert not (tmp_path / 'record.json').exists(), (
        'the engine is writing `record.json` again; stage 6 writes its mar-1.0 record into this '
        'same directory, so one of the two silently overwrites the other')
    assert paths['record'] == str(tmp_path / 'deaverage_output.json'), (
        'ensemble_files["record"] must point at the file that actually exists -- a dangling path '
        'here was live in every CLI run once already')


def test_the_written_path_round_trips(tmp_path):
    rec = _one_sample(tmp_path)
    paths = rec.write_ensemble(str(tmp_path))

    with open(paths['record'], encoding='utf-8') as fh:
        back = json.load(fh)
    assert back['status'] == 'de-averaged' and back['calculator'] == 'stub'


def test_the_gallery_renders_a_mar_1_0_record_with_its_own_builder(tmp_path):
    """This used to assert the OPPOSITE — the gallery could not read a stage-6 record, so it skipped
    it loudly rather than drawing a page of blank fields. The mar-1.0 builder exists now, so the
    requirement flips: the record must be FOUND, and it must be routed to that builder rather than
    to the de-average one, which would produce exactly the blank page the old rule guarded against.
    """
    from demars_core.gallery import _find_results, _is_mar10

    (tmp_path / 'record.json').write_text(json.dumps(
        {'schema_version': 'mar-1.0', 'mechanism': {'class': 'B'},
         'gates': {'charge': {'state': 'derived', 'pass': True}}, 'review': None,
         'representative': {'file': 'x.cif'}}))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        found = list(_find_results(str(tmp_path)))

    assert len(found) == 1, 'a mar-1.0 record must now be rendered, not skipped'
    assert _is_mar10(found[0][2]), 'and routed to the mar-1.0 builder'
    assert not caught, f'nothing to warn about any more: {[str(w.message) for w in caught]}'


def test_a_record_that_is_neither_schema_is_still_skipped_loudly(tmp_path):
    """The mirror the old rule was really protecting: a page of blank fields is worse than an error.
    Recognising two schemas must not turn into recognising anything."""
    from demars_core.gallery import _find_results

    (tmp_path / 'record.json').write_text(json.dumps({'something': 'else', 'unrelated': 1}))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        found = list(_find_results(str(tmp_path)))

    assert found == []
    assert any('neither' in str(w.message) for w in caught), 'skipped, but silently'


def test_a_legacy_run_directory_still_renders(tmp_path):
    """Runs made before the rename keep the old name for the SAME schema -- they must still open."""
    from demars_core.gallery import _find_results

    (tmp_path / 'record.json').write_text(json.dumps(
        {'status': 'de-averaged', 'formula': 'Si2', 'distribution': {'n_relaxed': 3}}))

    found = list(_find_results(str(tmp_path)))
    assert len(found) == 1 and found[0][2]['formula'] == 'Si2'
