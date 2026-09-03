"""The Antypov O/S/V/P orbit descriptor (`demars_core.disorder_class`).

Pure-CIF, no MLIP, no GPU. Two things are pinned: the LABELS on the bundled COD fixtures -- the
descriptor is only useful to the analyst if it discriminates, and a silent regression to "everything
is O" would look like a clean answer -- and the `no_full_backbone` triage flag, which `strategy.md`
reads out of `record.disorder_descriptor` to decide NOT to random-decorate.

The input key is the FILE. There is no id lookup here; the published module took an ICSD id.
"""
import os

import pytest

from demars_core.disorder_class import classify, classify_text

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')

# disorder_set expected per fixture, read off the mechanism the manifest records:
#   S  = mixed elements on a FULL site (substitutional solid solution)
#   V  = single element, partial occupancy (vacancy)
#   SV = both on one orbit;  P = intersecting (split) sites;  SVP = all three
EXPECTED = {
    'cod_9003141.cif': (['O', 'S'],        'substitution on a full cation site'),
    'cod_1006128.cif': (['O', 'S'],        'A-site substitution, full site'),
    'cod_1010497.cif': (['O', 'S'],        'substitution on two full M sites'),
    'cod_1011163.cif': (['O', 'V'],        'cation vacancies, single element'),
    'cod_1521474.cif': (['S', 'V'],        'full substituted cation site + O vacancy'),
    'cod_1520848.cif': (['O', 'SV', 'V'],  'mixed cation site that is also vacancy-bearing'),
    'cod_1544358.cif': (['O', 'P', 'V'],   'split O sites (intersecting) + O vacancy'),
    'cod_9005182.cif': (['O', 'S', 'SV'],  'partial Ca site + mixed tetrahedral sites'),
    'cod_4000330.cif': (['O', 'SVP', 'V'], 'split tunnel sites, mixed and vacancy-bearing'),
}


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_the_orbit_labels_match_the_recorded_mechanism(name):
    want, why = EXPECTED[name]
    got = classify(os.path.join(FIX, name))
    assert got['disorder_set'] == want, f'{name} ({why}): {got["disorder_set"]} != {want}'
    assert got['n_orbits'] == len(got['orbit_labels']) > 0
    assert got['multiset'] == sorted(got['orbit_labels'])


def test_the_descriptor_discriminates():
    """A regression that collapsed every orbit to one label would still pass a
    "returns a dict with the right keys" test. The bundled fixtures must not all read alike."""
    sets = {tuple(classify(os.path.join(FIX, n))['disorder_set']) for n in EXPECTED}
    assert len(sets) >= 5, f'the descriptor is not discriminating: {sets}'


# ---- the triage flag ---------------------------------------------------------

def _cif(sites, a=5.0):
    """Minimal P1 CIF. `sites` = [(label, element, x, y, z, occ)]."""
    head = ['data_synthetic',
            f'_cell_length_a {a}', f'_cell_length_b {a}', f'_cell_length_c {a}',
            '_cell_angle_alpha 90', '_cell_angle_beta 90', '_cell_angle_gamma 90',
            "_space_group_name_H-M_alt 'P 1'", '_space_group_IT_number 1',
            'loop_', '_atom_site_label', '_atom_site_type_symbol',
            '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z',
            '_atom_site_occupancy']
    rows = [f'{lab} {el} {x} {y} {z} {occ}' for lab, el, x, y, z, occ in sites]
    return '\n'.join(head + rows) + '\n'


def test_a_fully_occupied_cell_has_no_disorder_and_no_backbone_flag():
    got = classify_text(_cif([('Na1', 'Na', 0.0, 0.0, 0.0, 1.0),
                              ('Cl1', 'Cl', 0.5, 0.5, 0.5, 1.0)]), oxidation_states={})
    assert set(got['disorder_set']) == {'O'}, got
    assert got['no_full_backbone'] is False, 'a fully occupied cell HAS a backbone'


def test_no_full_backbone_fires_only_when_every_orbit_is_vacancy_bearing():
    """The flag the analyst acts on: no fully occupied scaffold anywhere -> the deposited cell is an
    average of a correlated superstructure or a doubling artifact -> do NOT random-decorate. One
    full orbit is enough to clear it, which is why it is `all`, not `any`."""
    every = classify_text(_cif([('Na1', 'Na', 0.0, 0.0, 0.0, 0.5),
                                ('Cl1', 'Cl', 0.5, 0.5, 0.5, 0.5)]), oxidation_states={})
    assert set(every['disorder_set']) == {'V'}, every
    assert every['no_full_backbone'] is True, every

    one_full = classify_text(_cif([('Na1', 'Na', 0.0, 0.0, 0.0, 0.5),
                                   ('Cl1', 'Cl', 0.5, 0.5, 0.5, 1.0)]), oxidation_states={})
    assert one_full['no_full_backbone'] is False, one_full


def test_an_occupancy_that_sums_just_over_one_still_classifies():
    """Refinements report 1.0003-1.04 all the time. pymatgen's default occupancy_tolerance=1.0
    raises on those, which used to leave the whole descriptor empty -- so the tolerant parse is
    load-bearing, and it must RESCALE rather than drop the partial occupancies."""
    got = classify_text(_cif([('Na1', 'Na', 0.0, 0.0, 0.0, 1.02),
                              ('Cl1', 'Cl', 0.5, 0.5, 0.5, 1.0)]), oxidation_states={})
    assert set(got['disorder_set']) == {'O'}, got


# ---- the seams --------------------------------------------------------------

def test_a_path_and_its_own_text_agree():
    """`classify` normalises through `io.to_cif_text`; reading the file itself must not change the
    answer, or the record and a hand-run of the module would disagree."""
    path = os.path.join(FIX, 'cod_1544358.cif')
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    assert classify(path)['multiset'] == classify_text(text)['multiset']


def test_supplied_oxidation_states_are_used_instead_of_a_second_parse(monkeypatch):
    """The record already holds the stage-1a bundle, so it hands the oxidation states over. If the
    descriptor re-derived them anyway, every record would pay for a second full evidence pass."""
    from demars_core._engine import mar_evidence as MEV

    def _boom(*a, **k):
        raise AssertionError('evidence was re-derived despite oxidation_states being supplied')

    monkeypatch.setattr(MEV, 'evidence_from_text', _boom)
    got = classify(os.path.join(FIX, 'cod_1521474.cif'), oxidation_states={'Y': 3, 'Zr': 4, 'O': -2})
    assert got['disorder_set'] == ['S', 'V'], got


def test_a_mixed_valence_element_averages_instead_of_crashing():
    """`evidence['oxidation_states']` gives a LIST for a mixed-valence element; the radius lookup
    needs one number, and the mean is the one the record's other consumers use. Pinned against the
    scalar mean rather than a literal label set -- the labels are radius-sensitive, so hard-coding
    them here would pin the ionic-radius table, not the averaging."""
    listed = classify(os.path.join(FIX, 'cod_1544358.cif'),
                      oxidation_states={'Sr': 2, 'Fe': [3, 4], 'O': -2})
    mean = classify(os.path.join(FIX, 'cod_1544358.cif'),
                    oxidation_states={'Sr': 2, 'Fe': 3.5, 'O': -2})
    assert listed['multiset'] == mean['multiset'], (listed['multiset'], mean['multiset'])
    assert 'P' in listed['disorder_set'], listed        # the split O sites are still seen
