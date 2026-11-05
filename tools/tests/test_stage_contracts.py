"""Contract tests for the stage CLIs in `tools/`.

These pin the seams that live in `tools/` rather than in the package, and that nothing else
watched: the shape of `engine.json`, the `record.json` -> `deaverage_output.json` rename and the
path fixup that goes with it, the `mode` tag set, and the `built` 5-tuple `--from` hand-assembles.
The full list is the INTERNAL CONTRACT note in `demars_core/api.py`.

Why this file exists: the dangling `ensemble_files['record']` path was present in EVERY run through
`demars_engine.py` and no test looked at it. `demars-core`'s suite covers the package; `tools/` had
no suite at all, and the stage contracts are exactly what lives there.

No MLIP and no GPU -- the relaxer is injected and the SevenNet backend is monkeypatched, so this
runs on CPU in seconds. Run with `pytest tools/tests`.
"""
import json
import os

import pytest

import _icsd_env                                       # noqa: F401  MUST precede demars_core

from demars_core._engine import mar_engine as ME       # noqa: E402
from demars_core._engine import mar_evidence as MEV    # noqa: E402
from demars_core._engine.mar_record import build_record  # noqa: E402
from demars_core.record import MARRecord               # noqa: E402
from demars_core import api as API                     # noqa: E402

from demars_engine import _engine_json, _rename_deterministic_output  # noqa: E402


FIXDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '..', '..', 'demars-core', 'tests', 'fixtures')
FIXTURE = 'cod_1544358.cif'          # half-occupied orbit: enumeration actually has something to do


def _stub_relax(confs):
    """The injected relaxer: (energies, relaxed, valid). Energies are per-config so take-lowest has
    a real winner to pick, but nothing here depends on their values."""
    return [-100.0 - i for i in range(len(confs))], list(confs), [True] * len(confs)


@pytest.fixture(scope='module')
def engine_run():
    """One cheap engine run, reused: structure -> (ev, built, info, trace), as `deaverage()` gets it."""
    path = os.path.join(FIXDIR, FIXTURE)
    txt = open(path, encoding='utf-8').read()
    ev = MEV.evidence_from_text(txt, iid=None, search_siblings=False)
    struct = ME.structure_from_text(txt)
    e_ev, built, info, trace = ME.run_self_driving(
        structure=struct, evidence=ev, relax=_stub_relax, NR=4, MIN_A=8.0, max_rounds=1)
    assert built is not None, 'fixture no longer enumerates -- pick another'
    return path, ev, (e_ev, built, info, trace)


# ------------------------------------------------------------------------------------------------
# engine.json -- the stage 3 -> stage 6 contract
# ------------------------------------------------------------------------------------------------

def test_engine_json_survives_the_round_trip_into_build_record(engine_run):
    """The real contract is not "engine.json has keys" -- it is that stage 6 can CONSUME it.

    So assert it end to end: MARRecord -> _engine_json -> build_record -> a record whose gates were
    actually recomputed. `build_record` recomputing rather than trusting is the "code proves" rule;
    a gate that silently comes back absent would mean the record proves nothing.
    """
    path, ev, (e_ev, built, info, trace) = engine_run
    rec = MARRecord.from_engine(e_ev, built, info, trace, source=path, calculator='stub')

    ej = _engine_json(rec)
    out = build_record(None, ej, {}, evidence=ev)

    gates = out.get('gates') or {}
    assert 'charge' in gates and 'fidelity' in gates, f'stage 6 produced no gates: {sorted(gates)}'
    assert isinstance(gates['fidelity'], dict) and 'pass' in gates['fidelity']
    json.dumps(out, default=str)                 # the CLI writes it -- it must be serialisable


def test_engine_json_keeps_group_orbits_that_MARRecord_drops(engine_run):
    """The load-bearing fixup. `MARRecord.to_dict()['engine']` is a DISPLAY subset and drops
    `group_orbits`; the fidelity gate needs it. `_engine_json` re-merges the unfiltered einfo from
    `rec._built[4]` -- itself a shape dependency on the engine's 5-tuple.

    Without this, stage 6 still runs and still emits a fidelity gate -- just one computed from less
    than it should be. That is the silent-wrong-answer class, so it gets its own test.
    """
    path, _ev, (e_ev, built, info, trace) = engine_run
    rec = MARRecord.from_engine(e_ev, built, info, trace, source=path, calculator='stub')

    displayed = (rec.to_dict().get('engine') or {})
    full = _engine_json(rec)['engine']
    assert 'group_orbits' in built[4], 'engine no longer reports group_orbits -- contract moved'
    assert 'group_orbits' in full
    if 'group_orbits' not in displayed:
        return                                   # the filtering this fixup exists to undo
    pytest.skip('MARRecord now carries group_orbits itself; the _engine_json merge is redundant')


def test_engine_json_renames_representative_to_lowest(engine_run):
    """`build_record` reads `distribution.lowest`; `MARRecord` calls the same thing
    `representative`. The rename is silent if it stops happening -- `lowest` just goes None."""
    path, _ev, (e_ev, built, info, trace) = engine_run
    rec = MARRecord.from_engine(e_ev, built, info, trace, source=path, calculator='stub')

    dist = _engine_json(rec)['distribution']
    assert dist.get('lowest') is not None
    assert 'representative' not in dist


def test_engine_lowest_reports_a_composition_not_a_config_label(engine_run):
    """`representative.composition` must be the composition on BOTH branches.

    The engine-lowest branch used to write `lowest['label']` there, so every engine-path record
    carried `'rand26'` where a composition belongs -- while the charge gate, 40 lines earlier in the
    same function, read `lowest['composition']` correctly. Silent: nothing raises, no gate trips,
    and the field only looks wrong if you read it (the set-B scorer did).
    """
    path, ev, (e_ev, built, info, trace) = engine_run
    rec = MARRecord.from_engine(e_ev, built, info, trace, source=path, calculator='stub')

    out = build_record(None, _engine_json(rec), {}, evidence=ev)
    representative = out['representative']
    assert representative['source'] == 'engine-lowest', 'fixture stopped taking the engine path'

    comp = representative['composition']
    assert isinstance(comp, dict) and comp, f'composition is not a composition: {comp!r}'
    assert all(isinstance(v, int) for v in comp.values()), comp
    assert sum(comp.values()) == representative['n_atoms'], 'composition disagrees with n_atoms'
    # the label is provenance, not composition -- kept, but under its own key
    assert isinstance(representative.get('config_label'), str)


# ------------------------------------------------------------------------------------------------
# the record.json rename + the path it used to leave dangling
# ------------------------------------------------------------------------------------------------

class _FakeRec:
    def __init__(self, files):
        self.ensemble_files = files


def test_rename_moves_the_file_and_repoints_ensemble_files(tmp_path):
    """The regression this whole file exists for: `write_ensemble()` records the PRE-rename name, so
    `ensemble_files['record']` pointed at a file that no longer existed -- in every CLI run."""
    det = tmp_path / 'record.json'
    det.write_text('{"deterministic": true}')
    rec = _FakeRec({'record': str(det), 'ensemble': str(tmp_path / 'ensemble.xyz')})

    new = _rename_deterministic_output(str(tmp_path), rec)

    assert new == str(tmp_path / 'deaverage_output.json')
    assert os.path.exists(new) and not os.path.exists(str(det))
    assert rec.ensemble_files['record'] == new, 'the dangling path is back'
    assert rec.ensemble_files['ensemble'].endswith('ensemble.xyz'), 'other entries were disturbed'


def test_rename_frees_the_name_for_stage_6(tmp_path):
    """The point of the rename: stage 6 must be able to write its own `record.json` here without
    either file overwriting the other."""
    (tmp_path / 'record.json').write_text('{"deterministic": true}')
    rec = _FakeRec({'record': str(tmp_path / 'record.json')})

    _rename_deterministic_output(str(tmp_path), rec)
    (tmp_path / 'record.json').write_text('{"mar-1.0": true}')

    assert json.loads((tmp_path / 'record.json').read_text()) == {'mar-1.0': True}
    assert json.loads((tmp_path / 'deaverage_output.json').read_text()) == {'deterministic': True}


def test_rename_is_a_noop_when_there_is_nothing_to_rename(tmp_path):
    """--from mode and a failed build write no record.json; the CLI must not care."""
    rec = _FakeRec({'record': '/somewhere/else/record.json'})
    assert _rename_deterministic_output(str(tmp_path), rec) is None
    assert rec.ensemble_files['record'] == '/somewhere/else/record.json'


# ------------------------------------------------------------------------------------------------
# the `mode` tag -- a VALUE dependency shared with the package
# ------------------------------------------------------------------------------------------------

def test_make_relax_tags_are_a_closed_set_that_FINAL_TIER_MODES_indexes(monkeypatch):
    """`--from` mode decides whether to run the omni-mpa final tier from this tag.

    Rename a tag without updating `FINAL_TIER_MODES` and nothing raises -- the final tier simply
    stops running, and every affected record quietly reports only the nano energy. So assert the
    vocabulary is closed and that the final-tier set indexes into it.
    """
    monkeypatch.setattr(API.backend, 'ase_calculator', lambda **kw: (object(), None))

    tags = {
        API._make_relax('sevennet', False)[1],
        API._make_relax('sevennet', False, rigid_units=True)[1],
        API._make_relax(object(), False)[1],            # an injected ASE calculator
    }
    assert tags == {'sevennet', 'sevennet-ase-rigid', 'ase-generic'}, (
        f'the mode vocabulary changed: {sorted(tags)} -- update FINAL_TIER_MODES with it')
    assert set(API.FINAL_TIER_MODES) < tags, 'FINAL_TIER_MODES names a tag nothing produces'
    assert 'ase-generic' not in API.FINAL_TIER_MODES, (
        'an injected ASE calculator has no omni-mpa weights; it must not reach the final tier')


# ------------------------------------------------------------------------------------------------
# the `built` 5-tuple -- a SHAPE dependency
# ------------------------------------------------------------------------------------------------

def test_final_omni_accepts_the_tuple_that_from_mode_hand_assembles(monkeypatch, engine_run):
    """`--from` has no enumeration, so `tools/demars_engine.py` builds `built` by hand. That makes
    `_final_omni`'s unpacking part of the contract: change the tuple and --from breaks on a path
    that only runs for custom/repaired builds -- the hardest entries, least likely to be spot-checked.
    """
    _path, _ev, (_e_ev, built, _info, _trace) = engine_run
    atoms = built[1][0]                                  # a relaxed config from the real engine

    monkeypatch.setattr(API.backend, 'use_model', lambda **kw: None)
    monkeypatch.setattr(API.backend, 'batched_fire_relax',
                        lambda confs: ([-123.0] * len(confs), list(confs), [True] * len(confs)))

    # assembled EXACTLY as tools/demars_engine.py:_relax_one does it
    hand = ([atoms], [atoms], [True], [{'E_per_atom': -1.0, 'valid': True}], {})
    fm, final_struct = API._final_omni(hand, d3=False)

    assert fm is not None and final_struct is not None
    assert fm['E_final_per_atom'] == round(-123.0 / len(atoms), 4)
    assert fm['n_atoms'] == len(atoms)
    assert len(hand) == 5, 'the tuple --from assembles must stay 5 wide'


# --------------------------------------------------------------------------------------------
# D11 -- the sibling VALUE and its PROVENANCE must move together
# --------------------------------------------------------------------------------------------

def _ej_from_the_inner_run():
    """engine.json as `deaverage()` leaves it: no iid inside, so the lookup never ran."""
    return {'engine': {'icsd_id': None,
                       'ordered_sibling': 'not searched',
                       'provenance_summary': {
                           'n_unconfident': 2,
                           'unconfident': ['exclusion_merge.Sb', 'ordered_sibling'],
                           'states': {'exclusion_merge.Sb': 'out_of_window',
                                      'ordered_sibling': 'not_run'}}}}


def test_a_sibling_search_that_ran_is_not_still_recorded_as_not_run():
    """The regression: the CLI overwrote the VALUE from an outer lookup and left the inner
    provenance in place, so one file claimed the search both happened and never ran. `not_run`
    means UNCHECKED -- reporting it next to a checked-and-absent string is the confusion
    AGENTS.md exists to prevent."""
    from demars_engine import _stamp_sibling

    ej = _stamp_sibling(_ej_from_the_inner_run(), {
        'ordered_sibling': 'none — no fully-ordered ICSD entry of this composition',
        'ordered_sibling_ids': [],
        'ordered_sibling_provenance': {'state': 'derived', 'confident': True}}, icsd_id=999001)

    s = ej['engine']['provenance_summary']
    assert s['states']['ordered_sibling'] == 'derived'
    assert 'ordered_sibling' not in s['unconfident']
    assert s['n_unconfident'] == len(s['unconfident']) == 1
    assert 'no fully-ordered' in ej['engine']['ordered_sibling']


def test_a_lookup_that_really_did_not_run_stays_unconfident():
    """The guard on the fix: it must not launder every sibling field into 'searched'."""
    from demars_engine import _stamp_sibling

    ej = _stamp_sibling(_ej_from_the_inner_run(), {
        'ordered_sibling': 'none (no sibling DB configured)',
        'ordered_sibling_provenance': {'state': 'not_run', 'confident': False}})

    s = ej['engine']['provenance_summary']
    assert s['states']['ordered_sibling'] == 'not_run'
    assert s['unconfident'].count('ordered_sibling') == 1, 'duplicated instead of replaced'
    assert s['n_unconfident'] == 2


def test_the_icsd_id_actually_reaches_the_artifact():
    """`--icsd-id` self-excludes the entry from its own sibling search. It was never written to
    engine.json, so a run with the flag and a run without were byte-identical -- there was no way
    to tell from the artifact whether self-exclusion had been applied."""
    from demars_engine import _stamp_sibling

    assert _stamp_sibling(_ej_from_the_inner_run(), None, 999001)['engine']['icsd_id'] == 999001
    assert _stamp_sibling(_ej_from_the_inner_run(), None, None)['engine']['icsd_id'] is None


def test_stamping_twice_is_the_same_as_stamping_once():
    """main() stamps the flag before the lookup and again with the result."""
    from demars_engine import _stamp_sibling

    sib = {'ordered_sibling': 'none — no fully-ordered ICSD entry of this composition',
           'ordered_sibling_provenance': {'state': 'derived', 'confident': True}}
    once = _stamp_sibling(_ej_from_the_inner_run(), sib, 1)
    twice = _stamp_sibling(_stamp_sibling(_ej_from_the_inner_run(), None, 1), sib, 1)
    assert once == twice


def test_the_record_shows_the_cell_that_was_built_not_only_the_one_requested(engine_run):
    """D13: `generation_recipe.sampling` is copied from the engine verbatim, so the achieved cell
    reaches the record only if the engine puts it there. Without it a reader has `min_cell_A: 15.0`
    and no way to learn the shipped cell was 14.216 A short of it except by measuring the file."""
    path, ev, (e_ev, built, info, trace) = engine_run
    rec = MARRecord.from_engine(e_ev, built, info, trace, source=path, calculator='stub')
    ej = _engine_json(rec)
    rd = build_record(None, ej, {}, evidence=ev)

    smp = rd['generation_recipe']['sampling']
    assert 'min_cell_A' in smp and 'min_cell_A_achieved' in smp, smp
    assert smp['min_cell_A_achieved'] == ej['engine']['sampling']['min_cell_A_achieved']
    assert ej['engine']['provenance_summary']['states']['supercell'] in ('derived', 'overridden')


def test_a_thinned_ensemble_is_visible_in_the_record(engine_run):
    """D8: one set-A entry ranked a representative out of 4 relaxations of a requested 30 and
    recorded `status: de-averaged`. The record had NO field in which the missing 26 could be
    noticed -- `n_configs` alone reads like the whole ensemble. Same disease as D13: the request
    is recorded, the fulfilment is not."""
    path, ev, (e_ev, built, info, trace) = engine_run
    confs, rel, val, samples, einfo = built

    rec = MARRecord.from_engine(e_ev, built, info, trace, source=path, calculator='stub')
    full = build_record(None, _engine_json(rec), {}, evidence=ev)['ensemble']
    assert full['n_enumerated'] == len(samples)
    assert full['n_configs'] == full['n_enumerated']
    assert full['sampling_shortfall'] is None, 'a complete ensemble must not be flagged'

    # now drop all but one relaxation, exactly as a silent convergence failure would
    thin = ([confs[0]], [rel[0]], [True], [dict(samples[0])] + [
        dict(s, valid=False, E_per_atom=None) for s in samples[1:]], einfo)
    rec2 = MARRecord.from_engine(e_ev, thin, info, trace, source=path, calculator='stub')
    short = build_record(None, _engine_json(rec2), {}, evidence=ev)['ensemble']

    assert short['n_configs'] == 1 and short['n_enumerated'] == len(samples)
    assert short['sampling_shortfall'], 'a representative chosen from 1 of N is reported as whole'
    assert f'1/{len(samples)}' in short['sampling_shortfall']


# ---- D36: the --from path crashed on the marker the final tier returns ------------------------
#
# `api._final_omni` reports a final tier that was ASKED FOR and could not run by returning the
# marker dict `_final_unavailable()` builds, paired with `final_struct = None`. It never returns
# None. `_relax_one` tested `if fm is None`, so the marker fell through to the success branch and
# `fm['E_final_per_atom']` raised KeyError -- mid-run, during a campaign.
# D23 gave `deaverage()` this distinction; the CLI's copy of the same tier was missed.

def _one_atoms_relax(atoms_list, *a, **k):
    """`--from` relaxes exactly one structure, so this is the single-config shape. Deliberately NOT
    named `_stub_relax`: that name is already the module's per-config relaxer for `engine_run`, and
    shadowing it silently broke seven fixtures."""
    return [-10.0 * len(atoms_list[0])], list(atoms_list), [True]


@pytest.fixture
def _from_path(monkeypatch, tmp_path):
    """`--from` wired to a stub relaxer, on an ORDERED structure.

    Ordered on purpose: `--from` realises a custom/repaired build, and `AseAtomsAdaptor` refuses a
    partially-occupied structure outright -- so the bundled disordered fixtures are the wrong input
    for this path.
    """
    from ase import Atoms
    import demars_engine as DE
    monkeypatch.setattr(API, '_make_relax',
                        lambda *a, **k: (_one_atoms_relax, API.FINAL_TIER_MODES[0], {'model': 'stub'}))
    at = Atoms('NaCl', positions=[[0, 0, 0], [2.8, 0, 0]], cell=[5.6] * 3, pbc=True)
    cif = tmp_path / 'ordered.cif'
    at.write(str(cif), format='cif')
    return DE, str(cif), str(tmp_path / 'out')


def test_the_from_path_survives_a_final_tier_that_could_not_run(_from_path, monkeypatch):
    DE, cif, out_dir = _from_path
    marker = API._final_unavailable('CudaOutOfMemory: no room at the inn', 42)
    monkeypatch.setattr(API, '_final_omni', lambda built, d3=False: (marker, None))

    out = DE._relax_one(cif, None, False, out_dir)     # must not raise

    fm = out['single_MAR']['final_MAR']
    assert fm['available'] is False and fm['requested'] is True
    assert 'no room at the inn' in fm['reason'], 'the reason is the point -- it must survive verbatim'
    assert 'E_final_per_atom' not in out['single_MAR'], \
        'nothing may claim a final energy that was never computed'


def test_the_from_path_still_records_a_final_tier_that_did_run(_from_path, monkeypatch):
    """The other half of the discriminator: a real result must still land as flat keys + files."""
    DE, cif, out_dir = _from_path
    from ase.io import read
    ran = {'E_final_per_atom': -3.5, 'n_atoms': 2, 'modal': 'mpa'}
    monkeypatch.setattr(API, '_final_omni', lambda built, d3=False: (ran, read(cif)))

    out = DE._relax_one(cif, None, False, out_dir)

    assert out['single_MAR']['E_final_per_atom'] == -3.5
    assert 'final_MAR' not in out['single_MAR'], 'a tier that ran is not an unavailability marker'
    assert os.path.exists(f'{out_dir}/representative_final.cif')
