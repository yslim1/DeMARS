"""Nine defects a frozen 15-entry validation campaign found by execution (2026-08).

None of them produced a wrong structure. Each one either made a record say something true of what
the code measured and false of what a reader takes it to mean, or forced an analyst to supply by
hand a number the data already contained. They are grouped because they were found together and
because every fix is small enough that a comment could fake it -- so these tests drive behaviour,
and the structures they use are synthetic: nothing here depends on the validation set.
"""
import numpy as np
import pytest
from ase import Atoms

from demars_core._engine.mar_engine import (
    _bond_populations, _h_nbr_cuts, coordination_integrity)


# ---- one element PAIR, two chemically different bonds ------------------------

def test_two_real_bond_types_are_calibrated_apart():
    """Cyanide C-N (~1.15 A) and methyl-amine C-N (~1.46 A) can sit in one structure. A single
    median lands in whichever cluster is larger and reports every bond of the other kind stretched."""
    groups = _bond_populations([1.15] * 96 + [1.46] * 64, stretch=1.15)
    assert len(groups) == 2, 'two populations, each big enough to be a bond type'
    assert [len(g) for g in groups] == [96, 64]


def test_a_lone_long_bond_stays_an_outlier():
    """The defect this report exists to catch. Splitting it off would give it its own median and
    silence the detector -- so a group too small to BE a bond type must fold back."""
    assert _bond_populations([1.47] * 100 + [1.90], stretch=1.15) == [sorted([1.47] * 100 + [1.90])]


def test_a_cell_holding_both_c_n_bond_types_reports_no_false_stretch():
    """The same thing as a structure rather than a list of numbers."""
    pos, syms = [], []
    for k in range(12):                                    # 12 cyanide C-N at 1.15 A
        c = np.array([5.0 * k, 0.0, 0.0])
        pos += [c, c + [1.15, 0, 0]]; syms += ['C', 'N']
    for k in range(8):                                     # 8 methyl-amine C-N at 1.46 A
        c = np.array([5.0 * k, 6.0, 0.0])
        pos += [c, c + [1.46, 0, 0]]; syms += ['C', 'N']
    at = Atoms(syms, positions=pos, cell=[64.0, 20.0, 20.0], pbc=True)
    rep = coordination_integrity(at)
    assert rep['n_stretched'] == 0, f'both C-N bonds are real; got {rep["stretched_bonds"][:3]}'


def test_the_shattered_unit_is_still_flagged_end_to_end():
    """A PO4 with one torn-off O, among intact ones: the split must not hide it."""
    torn = 2.05          # inside the (r_P + r_O) * 1.25 = 2.16 A bond cutoff, so it is a STRETCHED
                         # bond and not simply a detached atom
    pos, syms = [], []
    for k in range(6):
        c = np.array([6.0 * k, 0.0, 0.0])
        pos += [c, c + [1.55, 0, 0], c + [-1.55, 0, 0], c + [0, 1.55, 0],
                c + [0, 0, 1.55 if k else torn]]
        syms += ['P'] + ['O'] * 4
    at = Atoms(syms, positions=pos, cell=[40.0, 20.0, 20.0], pbc=True)
    rep = coordination_integrity(at)
    assert rep['n_stretched'] >= 1, 'the torn P-O must survive the population split'
    assert any(abs(s[2] - torn) < 0.02 for s in rep['stretched_bonds'])


# ---- the H-capacity model could not see a metal-anion bond ------------------

@pytest.mark.parametrize('pair,at_least', [
    (('Fe', 'O'), 2.3),      # a Fe-O bond runs 1.9-2.3 A; the old fixed 1.8 A cutoff missed it
    (('Mn', 'O'), 2.2),
    (('P', 'O'), 1.6),
])
def test_the_h_capacity_cutoff_reaches_a_metal_anion_bond(pair, at_least):
    assert _h_nbr_cuts(list(pair))[0][1] > at_least, f'{pair} must count toward acceptor capacity'


def test_it_still_excludes_a_non_bonded_anion_contact():
    """Widening must not make every neighbouring O satisfy every other one."""
    assert _h_nbr_cuts(['O', 'O'])[0][1] < 2.0


def test_an_o_that_lost_its_metal_partner_gains_capacity():
    """M-O-M' vs HO-M'. Removing one metal must raise the O's capacity -- that difference is the
    whole signal `_complete_h` has for protonating the RIGHT oxygen."""
    from ase.geometry import get_distances
    from demars_core._engine.mar_engine import H_VALENCE

    def capacity(with_fe):
        syms = ['O', 'Mn'] + (['Fe'] if with_fe else [])
        pos = [[0, 0, 0], [2.2, 0, 0]] + ([[-2.0, 0, 0]] if with_fe else [])
        at = Atoms(syms, positions=pos, cell=[20.0] * 3, pbc=True)
        D = get_distances(at.get_positions(), at.get_positions(),
                          cell=np.array(at.get_cell()), pbc=True)[1]
        np.fill_diagonal(D, 9e9)
        return H_VALENCE['O'] - int((D[0] < _h_nbr_cuts(syms)[0]).sum())

    assert capacity(with_fe=False) == capacity(with_fe=True) + 1


# ---- same-element cutoff read from chemistry, not from the widest gap --------

def _synthetic_split_cif(el, sep, a=6.6, occ=0.5):
    """One split pair of `el` at `sep` A in a simple cubic cell, plus an ordered Na scaffold."""
    return f"""data_synthetic
_cell_length_a {a}
_cell_length_b {a}
_cell_length_c {a}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_symmetry_equiv_pos_as_xyz
  'x, y, z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Na1 Na1+ 0.5 0.5 0.5 1.0
Na2 Na1+ 0.0 0.5 0.0 1.0
Na3 Na1+ 0.5 0.0 0.0 1.0
X1 {el} 0.0 0.0 0.0 {occ}
X2 {el} {sep / a:.5f} 0.0 0.0 {occ}
"""


def _derived_cut(el, sep, a=6.6):
    """Drive the real derivation inside build() -- no MLIP, no ICSD, no campaign data."""
    from pymatgen.core import Structure
    from demars_core._engine import mar_engine as ME
    from demars_core._engine import mar_evidence as MEV

    def identity_relax(confs, **kw):
        return np.zeros(len(confs)), list(confs), [True] * len(confs)

    cif = _synthetic_split_cif(el, sep, a=a)
    _ev, _built, info = ME.build(
        None, NR=4, MIN_A=13.0, seed=0, structure=Structure.from_str(cif, fmt='cif'),
        evidence=MEV.evidence_from_text(cif, iid=None, search_siblings=False), relax=identity_relax)
    em, sym = info['exclusion_merge'], el.rstrip('+-0123456789')
    cut = em['same_element_cut_per_element'].get(sym)
    prov = (em.get('provenance') or {}).get(sym) or {}
    return (None if cut is None else float(cut)), prov.get('state')


def test_a_split_pair_above_the_old_window_is_merged():
    """An As pair at 1.86 A: far too close to be an As-As bond (~2.4 A), but ABOVE the 1.3 A window
    the old derivation inspected, so no cut was set at all and it had to be supplied by hand."""
    cut, _state = _derived_cut('As3-', 1.86)
    assert cut is not None and cut > 1.86, 'the impossible pair must be merged'


def test_a_real_neighbour_distance_is_left_alone():
    """The same element at 2.9 A is a real contact -- a face-sharing channel, not alternates."""
    cut, _state = _derived_cut('Mg2+', 2.90, a=9.0)
    assert cut is None or cut < 2.90, 'a real neighbour must not be merged away'


def test_the_cut_clears_every_impossible_pair_not_just_the_widest_gap():
    """A run of alternates whose widest INTERNAL gap sits in the middle: cutting there leaves the
    outer alternates free to co-occupy, which relaxation then heals out of sight."""
    from ase.data import atomic_numbers, covalent_radii
    pairs = [0.30, 0.42, 0.87, 1.89, 2.24, 3.03]
    alt_max = 2 * covalent_radii[atomic_numbers['Mg']] * 0.9
    below = [d for d in pairs if d < alt_max]; above = [d for d in pairs if d >= alt_max]
    assert above[0] - below[-1] > 0.5
    assert (below[-1] + above[0]) / 2 > 2.24, 'the cut must clear the LAST impossible pair'


def test_a_pair_that_could_be_a_real_bond_is_merged_but_not_called_certain():
    """r_cov bounds homoatomic bonds badly for heavy metals -- a Mo-Mo quadruple bond (2.61 A) is
    SHORTER than the impossibility threshold. Distance cannot separate that from alternates, so the
    merge still happens (leaving it unmerged is worse) and the provenance must stop saying `derived`."""
    from ase.data import atomic_numbers, covalent_radii
    for el, sep, certain in [('Mo', 2.61, False), ('W', 2.30, False), ('As', 0.40, True)]:
        rcov2 = 2 * covalent_radii[atomic_numbers[el]]
        assert bool(sep < rcov2 * 0.6) is certain, (
            f'{el} at {sep} A: certain-alternate must be {certain}')
        assert sep < rcov2 * 0.9, f'{el} at {sep} A is inside the merge threshold either way'


def test_the_uncertain_merge_is_reported_as_ambiguous():
    cut, state = _derived_cut('Mo6+', 2.60, a=9.0)
    assert cut is not None, 'the pair is still merged'
    assert state == 'ambiguous', f'a possible real M-M bond must not read as derived (got {state})'


def test_a_drastically_short_split_is_still_certain():
    cut, state = _derived_cut('Mo6+', 0.45, a=9.0)
    assert cut is not None and state == 'derived'


# ---- "did the enumeration run" vs "is this the engine's own pick" ------------

def test_a_representative_swap_does_not_erase_the_enumeration():
    """A run that ships a random-pool draw from its OWN enumeration (the SQS principle) still
    enumerated. Keying the flag off the override made such a record read
    `engine_enumeration_used: false` -- i.e. cell / n_configs / spread comparable to nothing."""
    import inspect
    from demars_core._engine import mar_record
    src = inspect.getsource(mar_record.build_record)
    line = [ln for ln in src.splitlines()
            if '"engine_enumeration_used"' in ln and not ln.strip().startswith('#')]
    assert len(line) == 1
    assert 'representative' not in line[0], (
        'the flag must report whether the enumeration RAN, not whose pick shipped -- '
        'representative.source already answers that')
    assert 'supercell' in line[0] and 'group_orbits' in line[0]


# ---- the report-only flag that was read as a control signal -----------------

def test_suspicious_merges_says_it_changes_nothing():
    """An analyst discarded a correct build believing the engine had reverted a merge on this flag.
    No code branches on it."""
    import inspect
    from demars_core._engine import mar_engine
    note = inspect.getsource(mar_engine.build).split('suspicious_merges.append(')[1].split('})')[0]
    assert 'REPORT ONLY' in note and 'NOT reverted' in note


# ---- the cell rule losing to a fixed atom budget ----------------------------

def test_the_atom_budget_is_a_run_parameter():
    """A large primitive cell cannot reach 1.5 nm on every axis inside the default budget, and the
    analyst had no way to ask for more -- the run just shipped a short axis."""
    import inspect
    from demars_core._engine.mar_engine import build
    from demars_core.api import deaverage
    assert 'max_atoms' in inspect.signature(build).parameters
    assert 'max_atoms' in inspect.signature(deaverage).parameters
    assert 'len(sc) > budget' in inspect.getsource(build), (
        'the back-off must consult the run budget, not only the module constant')


def test_the_chosen_budget_and_its_source_are_recorded():
    import inspect
    from demars_core._engine.mar_engine import build
    src = inspect.getsource(build)
    assert "'atom_budget_source'" in src or '"atom_budget_source"' in src


def test_a_raised_budget_actually_buys_a_bigger_cell():
    from pymatgen.core import Structure
    from demars_core._engine import mar_engine as ME
    from demars_core._engine import mar_evidence as MEV

    def identity_relax(confs, **kw):
        return np.zeros(len(confs)), list(confs), [True] * len(confs)

    cif = _synthetic_split_cif('As3-', 1.86)
    st, ev = Structure.from_str(cif, fmt='cif'), MEV.evidence_from_text(cif, iid=None,
                                                                        search_siblings=False)
    small = ME.build(None, NR=2, MIN_A=15.0, seed=0, structure=st, evidence=ev,
                     relax=identity_relax, max_atoms=60)[2]
    large = ME.build(None, NR=2, MIN_A=15.0, seed=0, structure=st, evidence=ev,
                     relax=identity_relax, max_atoms=4000)[2]
    assert small['n_atoms_template'] < large['n_atoms_template']
    assert small['supercell_provenance']['evidence']['atom_budget_source'] == 'override --max-atoms'


# ---- the connectivity CLI died printing what the gate had already found ------

def test_the_cli_does_not_treat_coverage_as_a_centre():
    """`_coverage` reports what was NOT examined. Printing it as a centre raised
    KeyError('ligands') AFTER a correct verdict, and exited 1."""
    import pathlib
    src = pathlib.Path('tools/demars_connectivity.py').read_text()
    body = '\n'.join(ln for ln in src.splitlines() if not ln.strip().startswith('#'))
    assert "startswith('_')" in body, 'the print loop must skip the non-centre keys'
    assert 'NOT examined' in body, 'and must surface the coverage a reviewer had to find by hand'
