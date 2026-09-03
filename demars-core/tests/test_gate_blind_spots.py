"""Three ways a gate said "clean" about something it had not looked at.

All three were found by execution during the set-A/set-B re-run, and all three share the shape this
project keeps tripping over: the deterministic layer returns a verdict that is true of what it
measured and false of what a reader takes it to mean.

  * the connectivity gate examines only COORD_FORMERS, a fixed 21-element set. One record
    read "clean apart from 9/48 Mn" while 16 of 62 Fe were under-coordinated -- and the mechanism
    under test lived on the Fe sublattice.
  * the charge gate can only float an element the CIF declares with a NON-INTEGER value, so a
    deposited loop that is wrong about an integer-declared species cannot be corrected. One entry
    declares `La3+ 3.4` (not a real species) and `Mo4+ 4`; the true chemistry is Mo6+, under which
    the shipped cell is exactly neutral -- but the gate reported q = +22.4, FAIL.
  * dist_stats tagged a single energy as `narrow` / `near-degenerate -> solid-solution-like`. Two
    records shipped that positive finding beside their own prose retracting it.
"""
import numpy as np
import pytest
from ase import Atoms

from demars_core.connectivity import audit_frame
from demars_core._engine.mar_record import _ox_override, dist_stats, valence_aware_charge


# ---- connectivity coverage -------------------------------------------------

def _fe_p_o():
    """An Fe centre and a P centre, both octahedral/tetrahedral-ish, in one cell."""
    pos = [[0, 0, 0]]                                   # Fe
    pos += [[2.0, 0, 0], [-2.0, 0, 0], [0, 2.0, 0], [0, -2.0, 0], [0, 0, 2.0], [0, 0, -2.0]]
    pos += [[6.0, 6.0, 6.0]]                            # P
    pos += [[7.5, 6.0, 6.0], [4.5, 6.0, 6.0], [6.0, 7.5, 6.0], [6.0, 6.0, 7.5]]
    return Atoms('FeO6PO4', positions=pos, cell=[16.0] * 3, pbc=True)


def test_the_gate_names_the_cations_it_never_examined():
    summary, _defects = audit_frame(_fe_p_o())
    cov = summary['_coverage']
    assert 'P' in cov['examined_centres'], 'P is a COORD_FORMERS element and must be examined'
    assert 'Fe' in cov['not_examined'], (
        'Fe is absent from COORD_FORMERS, so its coordination was never checked -- the gate must '
        'say so, or "clean" reads as "every cation checked"')
    assert 'O' not in cov['not_examined'], 'a ligand is not an unexamined centre'


def test_coverage_says_so_when_nothing_was_skipped():
    p_only = Atoms('PO4', positions=[[0, 0, 0], [1.5, 0, 0], [-1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]],
                   cell=[14.0] * 3, pbc=True)
    cov = audit_frame(p_only)[0]['_coverage']
    assert cov['not_examined'] == []
    assert 'every non-ligand element present was examined' in cov['note']


# ---- charge gate override --------------------------------------------------

def test_a_cif_declared_integer_species_can_be_corrected_with_provenance():
    """That entry in miniature: La 3.4 (impossible) + Mo 4 declared, real chemistry Mo6+."""
    comp = {'Li': 208, 'La': 96, 'Zr': 56, 'Mo': 8, 'O': 384}
    cif_states = {'Li': 1, 'La': 3.4, 'Zr': 4, 'Mo': 4, 'O': -2}

    as_deposited = valence_aware_charge(comp, cif_states)
    assert as_deposited['pass'] is False, 'the CIF loop should not neutralize this composition'

    corrected = valence_aware_charge(comp, cif_states,
                                     ox_override={'La': 3, 'Mo': 6})
    assert corrected['pass'] is True and abs(corrected['q_per_cell']) < 1e-6


def test_the_override_is_never_silent():
    comp = {'Na': 1, 'Cl': 1}
    g = valence_aware_charge(comp, {'Na': 1, 'Cl': -1}, ox_override={'Na': 1})
    assert g['oxidation_override'] == {'Na': 1}
    assert 'override' in g['mode'], 'the verdict must say the states were not the CIF\'s'


def test_no_override_leaves_the_verdict_untouched():
    comp = {'Na': 1, 'Cl': 1}
    g = valence_aware_charge(comp, {'Na': 1, 'Cl': -1})
    assert g['pass'] is True and 'oxidation_override' not in g
    assert g['mode'] == 'neutral(mean)'


@pytest.mark.parametrize('judg,expect', [
    ({'oxidation_override': {'states': {'Mo': 6}, 'reason': 'r', 'evidence': 'e'}}, {'Mo': 6}),
    ({'oxidation_override': {'Mo': 6}}, {'Mo': 6}),        # bare mapping still accepted
    ({}, None),
    (None, None),
])
def test_the_override_is_read_off_the_judgment(judg, expect):
    assert _ox_override(judg) == expect


# ---- single-point distribution ---------------------------------------------

def test_one_energy_is_not_a_narrow_distribution():
    d = dist_stats([-6.4617])
    assert d['n_configs'] == 1
    assert d['shape'] is None and d['sro_read_indicative'] is None, (
        'a single point was tagged as a distribution shape; a downstream consumer reads '
        'sro_read_indicative as a positive finding')
    assert d['sro_read_available'] is False
    assert 'one valid configuration' in d['sro_read_unavailable_reason']


def test_a_real_distribution_still_gets_its_read():
    d = dist_stats([-6.50, -6.49, -6.46, -6.40])
    assert d['n_configs'] == 4
    assert d['shape'] is not None and d['sro_read_indicative']
    assert 'sro_read_available' not in d or d.get('sro_read_available') is not False


def test_no_energies_is_still_empty():
    assert dist_stats([]) == {} and dist_stats(None) == {}


# ---- and one way the gate said DEFECT about something that was fine ---------
#
# The mirror image of the three above. `former_coordination` derives ONE modal CN per ELEMENT, so an
# element that centres two different unit types in the same structure gets one expectation for both.
# A cyanide entry with methylated cations produced 160 false defects on a chemically fine
# structure, in two distinct ways: 64 methyl C judged against the cyanide C's mode, and 96 cyanide N
# re-attached as 0-coordination "fully stripped centres" for being ligands. Those false positives are
# what blocked making this gate automatic -- automation would have failed the record.
#
# The structures here are synthetic on purpose: the frozen run lives under `tmp/`, which is
# gitignored, so a test that read it would pass here and vanish in a clone.

def _cyanide_and_methylamine():
    """4x cyanide (C bonded to N only) + 4x H3C-NH2 (C bonded to N and 3 H; N bonded to 2 H).

    Reproduces that shape: C centres two unit types ({N} at CN 1, {N,H} at CN 4) and half the N
    are LIGANDS of a C rather than centres of their own.
    """
    sym, pos = [], []
    for k in range(4):                                    # cyanide, C=N 1.16 A
        o = np.array([5.0 * k, 0.0, 0.0])
        sym += ['C', 'N']; pos += [o, o + [1.16, 0, 0]]
    for k in range(4):                                    # methylamine
        o = np.array([5.0 * k, 0.0, 10.0])
        sym += ['C', 'N']; pos += [o, o + [1.47, 0, 0]]
        sym += ['H'] * 3                                  # on C, 1.09 A
        pos += [o + [-0.51, 0.96, 0], o + [-0.51, -0.48, 0.83], o + [-0.51, -0.48, -0.83]]
        sym += ['H'] * 2                                  # on N, 1.02 A
        pos += [o + [1.81, 0.83, 0.44], o + [1.81, -0.83, 0.44]]
    return Atoms(''.join(sym), positions=pos, cell=[25.0, 12.0, 22.0], pbc=True)


def _sulfates(n_full=3, extra='none'):
    """n_full intact SO4, plus one S that is either stripped bare or left as SO3."""
    sym, pos = [], []
    def so(o, n_o):
        out_s, out_p = ['S'], [o]
        d = [[1.47, 0, 0], [-1.47, 0, 0], [0, 1.47, 0], [0, 0, 1.47]][:n_o]
        out_s += ['O'] * n_o; out_p += [o + np.array(v) for v in d]
        return out_s, out_p
    for k in range(n_full):
        s, p = so(np.array([6.0 * k, 0.0, 0.0]), 4); sym += s; pos += p
    if extra == 'stripped':
        sym += ['S']; pos += [np.array([0.0, 0.0, 12.0])]
    elif extra == 'so3':
        s, p = so(np.array([0.0, 0.0, 12.0]), 3); sym += s; pos += p
    return Atoms(''.join(sym), positions=pos, cell=[24.0, 12.0, 24.0], pbc=True)


def test_one_element_centring_two_unit_types_is_not_one_expectation():
    """The 64-defect half of it: methyl C measured against the cyanide C's mode."""
    summary, defects = audit_frame(_cyanide_and_methylamine())
    roles = {k for k in summary if not k.startswith('_')}
    assert roles == {'C[H+N]', 'C[N]', 'N'}, roles
    assert summary['C[N]']['expected'] == 1 and summary['C[H+N]']['expected'] == 4
    assert defects == [], f'a chemically fine structure must not report defects: {defects[:3]}'


def test_a_ligand_is_not_a_stripped_centre():
    """The 96-defect half: N is in COORD_FORMERS *and* COORD_ANIONS, so a cyanide N -- a ligand doing
    its job -- was re-attached at coordination 0, the worst verdict the gate has."""
    summary, defects = audit_frame(_cyanide_and_methylamine())
    assert summary['_coverage']['ligand_role'] == {'N': 4}, summary['_coverage']
    assert not any(d['centre'] == 'N' and d['coordination'] == 0 for d in defects)
    assert summary['N']['expected'] == 2 and summary['N']['n_centres'] == 4, \
        'the amine N are still centres; only the cyanide N moved to the ligand role'


def test_a_centre_that_lost_every_ligand_is_still_a_defect():
    """The hazard the role split creates: if an empty ligand signature were allowed to form its own
    role, a stripped centre would define its own expectation of zero and PASS. It is judged against
    its element's dominant role instead."""
    summary, defects = audit_frame(_sulfates(extra='stripped'))
    assert summary['S']['expected'] == 4
    assert [d['coordination'] for d in defects] == [0], defects


def test_losing_one_ligand_does_not_buy_a_new_role():
    """An SO3 keeps the ligand signature {O} of its SO4 siblings, so it stays in their group and
    stays a defect. That is why splitting on the signature cannot hide what this gate exists to find."""
    summary, defects = audit_frame(_sulfates(extra='so3'))
    assert set(summary) - {'_coverage'} == {'S'}, 'signature unchanged -> one role, not two'
    assert summary['S']['expected'] == 4
    assert [d['coordination'] for d in defects] == [3], defects


def test_the_role_split_adds_no_tunable():
    """Role sizes on the frozen run's 60 structures are NOT bimodal (8 of 16 fractions inside
    (0.05, 0.5)), so a role-size cutoff would be load-bearing and corpus-tuned. The only prevalence
    knob stays the element-level `MIN_UNIT_PREVALENCE`, whose own distribution IS bimodal."""
    import inspect
    from demars_core._engine.mar_engine import centre_expectations
    params = set(inspect.signature(centre_expectations).parameters)
    assert params == {'atoms', 'formers', 'ligands', 'tol', 'min_prevalence', 'expect'}, params


def test_a_single_role_element_reads_exactly_as_before():
    """14 of the frozen run's 15 entries have one unit type per element. Those must not churn: the
    label stays the bare element and the verdict is unchanged."""
    summary, defects = audit_frame(_fe_p_o())
    assert 'P' in summary and 'P[O]' not in summary
    assert summary['P']['expected'] == 4 and defects == []
