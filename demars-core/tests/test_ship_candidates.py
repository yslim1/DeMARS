"""Which sample may become the representative (`mar_engine.ship_candidates`).

The enumeration appends two bracket probes (`dispersed`, `clustered`) built at maximum / minimum
similarity to BOUND the distribution. `strategy.md`'s SQS principle is random-only — a disordered
phase's representative must be a typical random snapshot, not the 0 K minimum, and a designed extreme
is not a member of the random distribution at all. Take-lowest over the whole pool lets a probe win,
usually by a margin at the relaxation noise floor.

The research tree wrote that flaw down next to the principle and deferred the fix; the record layer
then grew `gates.sqs` to DETECT it and `representative_pick` to work AROUND it by hand. This is the
fix, and its load-bearing property is that THREE places choose the lowest and now share one
definition — otherwise the record, the final tier and the written `representative.*` can disagree.

No MLIP, no GPU, no network.
"""
import inspect

import pytest

from demars_core._engine.mar_engine import BRACKET_MODES, bracket_margins, ship_candidates


def _pool(disp=-7.2, clus=-6.5, rand=(-7.0, -7.1), valid_random=True):
    out = [{'label': f'rand{i}', 'mode': 'random', 'E_per_atom': e, 'valid': valid_random}
           for i, e in enumerate(rand)]
    out += [{'label': 'dispersed', 'mode': 'dispersed', 'E_per_atom': disp, 'valid': True},
            {'label': 'clustered', 'mode': 'clustered', 'E_per_atom': clus, 'valid': True}]
    return out


# ---- the selection ----------------------------------------------------------

def test_a_probe_that_wins_take_lowest_is_not_a_candidate():
    """The whole point: `dispersed` at -7.2 is the global minimum and must not be the representative."""
    pool = _pool()
    assert min(p['E_per_atom'] for p in pool) == -7.2, 'fixture check: the probe IS the global minimum'
    cand = ship_candidates(pool)
    assert {c['label'] for c in cand} == {'rand0', 'rand1'}
    assert min(cand, key=lambda c: c['E_per_atom'])['label'] == 'rand1'


def test_invalid_samples_are_excluded_as_before():
    pool = _pool()
    pool[0]['valid'] = False
    assert {c['label'] for c in ship_candidates(pool)} == {'rand1'}


def test_both_extremes_are_excluded_not_just_clustered():
    """`sqs_gate` learned this the hard way: a gate covering `clustered` only still missed two of
    three records that shipped a probe."""
    assert set(BRACKET_MODES) == {'clustered', 'dispersed'}
    labels = {c['label'] for c in ship_candidates(_pool())}
    assert 'dispersed' not in labels and 'clustered' not in labels


def test_it_falls_back_when_no_random_sample_survived():
    """There is nothing else to ship then. `sqs_gate` stays as the watchdog that says a probe shipped."""
    pool = _pool(valid_random=False)
    assert {c['label'] for c in ship_candidates(pool)} == {'dispersed', 'clustered'}


def test_an_empty_pool_is_empty_not_an_error():
    assert ship_candidates([]) == []
    assert ship_candidates([{'label': 'x', 'mode': 'random', 'valid': False}]) == []


# ---- the margin that must survive the exclusion -----------------------------

def test_the_probe_margins_are_reported():
    """Excluding the probes silently would erase the signal `strategy.md` asks for in the same breath:
    a clustered config FAR below the random ones means the material may genuinely order."""
    m = bracket_margins(_pool())
    assert set(m) == {'dispersed', 'clustered'}
    assert m['dispersed']['margin_meV_vs_lowest_ship'] == -100.0, m   # below the random draws
    assert m['clustered']['margin_meV_vs_lowest_ship'] == 600.0, m    # above them


def test_a_clustered_probe_far_below_the_random_draws_is_visible():
    m = bracket_margins(_pool(clus=-7.5))
    assert m['clustered']['margin_meV_vs_lowest_ship'] == -400.0, (
        'a strongly ordering material must be legible from the distribution alone')


def test_margins_are_empty_when_there_is_nothing_to_compare():
    assert bracket_margins([]) == {}


# ---- the three sites agree --------------------------------------------------

def test_every_site_that_picks_the_lowest_uses_this_one_definition():
    """The real risk of this change, and it bit once: FIVE places choose the lowest, and the first
    attempt fixed three of them — none of which the `tools/` CLI actually goes through, so an
    end-to-end run still shipped the probe while the source-level test passed. Enumerated here so a
    new call site cannot be added quietly.

        1. `mar_engine`'s own `distribution.lowest`      (running the engine module directly)
        2. `api._final_omni`                             (which structure gets the final tier)
        3. `mar_engine`'s written `representative.*`
        4. `MARRecord.from_engine`'s `distribution`       (deaverage() / tools/demars_engine.py)
        5. `MARRecord.write_ensemble`'s written files     (same path)
    """
    from demars_core import api, record
    from demars_core._engine import mar_engine

    eng_src, rec_src = inspect.getsource(mar_engine), inspect.getsource(record)
    assert 'lo = min(ship, key=lambda x: x["E_per_atom"])' in eng_src, '(1) distribution.lowest'
    assert 'ship = engine.ship_candidates(samples)' in inspect.getsource(api._final_omni), '(2) final tier'
    assert "_ship_labels = {s.get('label') for s in ship_candidates(samples)}" in eng_src, '(3) written rep'
    assert "lo = min(ship, key=lambda x: x['E_per_atom'])" in \
        inspect.getsource(record.MARRecord.from_engine), '(4) MARRecord distribution'
    assert 'ship_candidates(samples)' in inspect.getsource(record.MARRecord.write_ensemble), \
        '(5) MARRecord written files'

    # and NO site may still take the raw minimum over every valid sample
    for name, src in (('api._final_omni', inspect.getsource(api._final_omni)),
                      ('MARRecord.from_engine', inspect.getsource(record.MARRecord.from_engine))):
        assert "min(ok, key=lambda x: x['E_per_atom'])" not in src, name
    assert 'min(ok, key=lambda x: x["E_per_atom"])' not in eng_src, 'mar_engine'


def test_the_representative_written_to_disk_is_the_one_the_record_names():
    """(3) and (5) pick a FILE while (1) and (4) pick a SAMPLE. They agreed only by both taking the
    global minimum; now they agree by both consulting `ship_candidates`, and this checks the label
    actually lines up rather than trusting that."""
    import numpy as np
    from ase import Atoms

    from demars_core.record import MARRecord

    samples, rel = [], []
    for lab, mode, e in (('rand0', 'random', -7.00), ('rand1', 'random', -7.10),
                         ('dispersed', 'dispersed', -7.20), ('clustered', 'clustered', -6.50)):
        samples.append({'label': lab, 'mode': mode, 'E_per_atom': e, 'valid': True, 'charge': 0.0})
        rel.append(Atoms('Cu', positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True))
    built = ([None] * 4, rel, [True] * 4, samples, {'supercell': [1, 1, 1]})

    rec = MARRecord.from_engine({'oxidation_states': {'Cu': 0}, 'ordered_sibling': None},
                                built, {}, [], 'x.cif', 'stub')
    assert rec.distribution['representative']['label'] == 'rand1', rec.distribution['representative']
    assert rec.distribution['ground_E_per_atom'] == -7.20, 'the distribution still spans the probes'
    assert rec.distribution['n_ship_candidates'] == 2
    assert rec.distribution['bracket_probes']['dispersed']['margin_meV_vs_lowest_ship'] == -100.0

    import tempfile
    from ase.io import read
    with tempfile.TemporaryDirectory() as d:
        rec.write_ensemble(d)
        ens = read(f'{d}/ensemble.xyz', index=':')
        assert [a.info.get('label') for a in ens][0] == 'dispersed', (
            'the ensemble is still energy-sorted and still contains the probes')
        # the written representative is the SHIP candidate, not the ensemble's first frame
        assert len(read(f'{d}/representative.cif')) == 1


def test_the_distribution_still_describes_every_valid_sample():
    """Only the representative is chosen from the candidates. The spread, the std and the histogram
    are the whole distribution -- bounding it is exactly what the probes are for, so narrowing those
    to the random samples would throw away the bounds and shrink every reported spread."""
    from demars_core._engine import mar_engine
    src = inspect.getsource(mar_engine)

    # the distribution's own numbers come from `ok` (every valid sample)
    assert 'Es = sorted(smp["E_per_atom"] for smp in ok)' in src
    assert '"std_meV": round(1000 * float(np.std([s["E_per_atom"] for s in ok])), 2)' in src
    assert 'hi = max(ok, key=lambda x: x["E_per_atom"])' in src
    # while the representative comes from the candidates, and the count is recorded
    assert 'lo = min(ship, key=lambda x: x["E_per_atom"])' in src
    assert '"n_ship_candidates": len(ship)' in src
