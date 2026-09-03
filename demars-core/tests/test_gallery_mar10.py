"""The gallery's mar-1.0 renderer.

`demars gallery` rendered only `MARRecord` (the deterministic engine's own output) and skipped the
stage-⑥ record loudly — so a campaign's run directory produced an empty page, which is the "gallery
is not wired to campaign run directories" gap. This is the missing half: per-entry pages rendered
from the record layer, showing gates, verdict, E_above_hull, class and confidence.

The load-bearing property is honesty: a gate whose `pass` is null is NOT a pass, and a gate the
record does not carry at all is not one either. Both must be visually distinct from a pass, or the
index page launders exactly what the four-state vocabulary exists to expose.

No MLIP, no GPU, no network.
"""
import json
import re

import pytest

from demars_core.gallery import (_gate_chip, _is_mar10, _mar10_card, _mar10_detail, _mar10_flags,
                                build_gallery)


def _rec(**over):
    rec = {
        'schema_version': 'mar-1.0',
        'provenance': {'formula_sum': 'Nb1 Pd0.23 S2', 'chemical_name': 'test'},
        'mechanism': {'class': 'B', 'class_label': 'vacancy solid solution',
                      'interpretation': 'a dilute interlayer site', 'confidence': 'high'},
        'representative': {'source': 'engine-repick', 'n_atoms': 322, 'composition': {'Nb': 100},
                           'config_label': 'rand15'},
        'ensemble': {'n_frames': 32, 'spread_meV': 40.0, 'std_meV': 12.0},
        'gates': {'charge': {'state': 'derived', 'pass': True, 'basis': 'x'},
                  'fidelity': {'pass': True, 'basis': 'x'},
                  'connectivity': {'state': 'derived', 'pass': True, 'basis': 'x'},
                  'sqs': {'state': 'derived', 'pass': True, 'basis': 'x'},
                  'hull': {'state': 'derived', 'pass': None, 'basis': 'no threshold configured'}},
        'review': {'final_verdict': 'confirm', 'n_rounds': 1},
        'disorder_descriptor': {'available': True, 'disorder_set': ['O', 'V'],
                                'no_full_backbone': False},
        'hull': {'state': 'derived', 'E_above_hull_eV_per_atom': 0.012, 'mode': 'self-consistent',
                 'corrections': 'mp2020', 'decomposition': {'AB': 1.0}},
    }
    rec.update(over)
    return rec


def test_a_mar10_record_is_recognised():
    assert _is_mar10(_rec()) is True
    assert _is_mar10({'distribution': {}, 'status': 'de-averaged'}) is False


# ---- the honesty of the gate chips ------------------------------------------

def test_a_pass_and_a_null_pass_do_not_look_alike():
    ok = _gate_chip('charge', {'state': 'derived', 'pass': True})
    null = _gate_chip('connectivity', {'state': 'vacuous', 'pass': None})
    fail = _gate_chip('sqs', {'state': 'derived', 'pass': False})
    assert 'gate ok' in ok and 'pass' in ok
    assert 'gate unchecked' in null and 'vacuous' in null and '>pass<' not in null
    assert 'gate err' in fail and 'FAIL' in fail
    assert len({re.search(r'gate (\w+)', c).group(1) for c in (ok, null, fail)}) == 3


def test_a_reported_hull_says_reported_not_pass():
    """`derived` with `pass: null` is the hull gate carrying a number with no threshold to judge it
    against. It is neither a pass nor unchecked, and calling it either would be wrong."""
    chip = _gate_chip('hull', {'state': 'derived', 'pass': None, 'basis': 'no threshold'})
    assert 'reported' in chip and 'gate unchecked' in chip
    assert 'pass' not in chip.replace('gate', '')


def test_a_gate_the_record_does_not_carry_says_absent():
    """An older record predating a gate must not render as nothing — a reader cannot tell an absent
    gate from a passing one."""
    chip = _gate_chip('hull', None)
    assert 'absent' in chip and 'gate unchecked' in chip


# ---- the index page as an audit view ----------------------------------------

def test_a_clean_record_shows_no_flags():
    flags, worst = _mar10_flags(_rec())
    assert flags == [] and worst == 'ok'


@pytest.mark.parametrize('over,expect,worst', [
    ({'review': None}, 'UNREVIEWED', 'unchecked'),
    ({'escalate': 'needs literature'}, 'ESCALATED', 'err'),
    ({'review': {'final_verdict': 'escalate', 'unresolved_blocking': ['x']}},
     'blocking objection', 'err'),
])
def test_what_the_index_must_surface_without_a_click(over, expect, worst):
    flags, got = _mar10_flags(_rec(**over))
    assert any(expect in f for f in flags), flags
    assert got == worst


def test_a_failing_gate_outranks_an_unchecked_one():
    g = _rec()['gates']; g['charge'] = {'state': 'derived', 'pass': False, 'basis': 'q=1.2'}
    g['connectivity'] = {'state': 'not_run', 'pass': None, 'basis': 'no frame'}
    flags, worst = _mar10_flags(_rec(gates=g))
    assert worst == 'err'
    assert any('charge FAILS' in f for f in flags) and any('connectivity not_run' in f for f in flags)


def test_an_absent_gate_is_flagged_too():
    g = dict(_rec()['gates']); g.pop('hull')
    flags, worst = _mar10_flags(_rec(gates=g))
    assert any('hull absent' in f for f in flags), flags
    assert worst == 'unchecked'


# ---- the rendered pages ------------------------------------------------------

def test_the_card_carries_the_class_verdict_and_gates():
    card = _mar10_card('icsd_1', _rec(), 'r-x.html', True)
    for want in ('badge cls', '>B<', 'confirm', 'gate ok', 'engine-repick', 'O/V'):
        assert want in card, want


def test_a_sentence_in_the_class_field_does_not_break_the_badge():
    long = 'E (bonded/slaved interstitial-D disorder), with a composition-forced residual'
    card = _mar10_card('icsd_1', _rec(mechanism={'class': long}), 'r-x.html', False)
    shown = re.search(r'badge cls" title="([^"]*)">([^<]*)<', card)
    assert len(shown.group(2)) <= 16, shown.group(2)
    assert shown.group(1).startswith('E (bonded'), 'the full string belongs in the tooltip'


def test_the_detail_page_states_an_unreviewed_record_plainly():
    page = _mar10_detail('icsd_1', _rec(review=None), None, 'T')
    assert 'UNREVIEWED' in page and 'not reviewed-and-clean' in page


def test_the_detail_page_shows_a_refused_hull_as_a_state_not_a_number():
    page = _mar10_detail('icsd_1', _rec(hull={'state': 'not_run', 'reason': 'no MP key'}), None, 'T')
    assert 'not_run' in page and 'no MP key' in page


def test_a_missing_final_tier_is_stated(tmp_path):
    rep = dict(_rec()['representative'])
    rep['E_final_unavailable_reason'] = 'repicked from the ensemble'
    page = _mar10_detail('icsd_1', _rec(representative=rep), None, 'T')
    assert 'no final tier' in page and 'repicked' in page


# ---- end to end --------------------------------------------------------------

def test_a_directory_of_mar10_records_builds_a_gallery(tmp_path):
    for i, over in enumerate(({}, {'review': None})):
        d = tmp_path / f'icsd_{i}'
        d.mkdir()
        (d / 'record.json').write_text(json.dumps(_rec(**over)))
    out = build_gallery(str(tmp_path), out_dir=str(tmp_path / '_g'), title='T')
    index = (tmp_path / '_g' / 'index.html').read_text()
    assert index.count('class="card"') == 2, 'both records must render'
    assert 'gate ok' in index and 'UNREVIEWED' in index
    assert len(list((tmp_path / '_g').glob('r-*.html'))) == 2
    assert out == str(tmp_path / '_g')


def test_both_schemas_can_sit_in_one_directory(tmp_path):
    """A tree may hold engine-only runs and judged entries side by side; each gets its own builder."""
    (tmp_path / 'judged').mkdir()
    (tmp_path / 'judged' / 'record.json').write_text(json.dumps(_rec()))
    (tmp_path / 'engine_only').mkdir()
    (tmp_path / 'engine_only' / 'deaverage_output.json').write_text(json.dumps({
        'status': 'de-averaged', 'formula': 'CuAu', 'calculator': 'sevennet',
        'distribution': {'ground_E_per_atom': -3.0, 'spread_meV': 5.0}}))
    build_gallery(str(tmp_path), out_dir=str(tmp_path / '_g'), title='T')
    index = (tmp_path / '_g' / 'index.html').read_text()
    assert index.count('class="card"') == 2
    assert 'badge cls' in index and 'de-averaged' in index
