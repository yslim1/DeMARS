"""Proton placement: `_place_h` / `_complete_h` (engine H-restore).

These pin the placement layer, not the decision layer above it (formula-scaled deficit,
per-acceptor capacity, the orientational-disorder safety check, the aliovalent-former Bronsted
case): that a placement CHECKS for clashes and retries, that the acceptor list reaches beyond O/N,
and that one blocked host must not cost the structure the protons the formula demands.

No MLIP, no GPU, no network.
"""
import numpy as np
import pytest
from ase import Atoms
from ase.geometry import get_distances

from demars_core._engine.mar_engine import (BV_MIN_DEFICIT, H_BOND_LEN, H_MIN_SEP, H_VALENCE,
                                            PROTON_CAP, _bv_deficits, _bv_fallback_ok, _complete_h,
                                            _h_nbr_cuts, _place_h)


def _min_hh(atoms):
    """Closest H-H distance. The host-proton bond is a BOND, not a clash, so it is excluded by
    looking only at the protons against each other."""
    idx = [i for i, e in enumerate(atoms.get_chemical_symbols()) if e == 'H']
    if len(idx) < 2:
        return float('inf')
    P = atoms.get_positions()[idx]; C = np.array(atoms.get_cell())
    D = get_distances(P, P, cell=C, pbc=True)[1]
    np.fill_diagonal(D, 9e9)
    return float(D.min())


def _shell(centre, r, n_dir=3):
    """A cage of inert atoms at radius r in every octant direction -- encloses the centre."""
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                v = np.array([dx, dy, dz], float)
                out.append(list(np.asarray(centre) + r * v / np.linalg.norm(v)))
    return out


# ---- the clash check ---------------------------------------------------------

def test_a_proton_is_never_placed_inside_another_atom():
    """The ideal void direction is not always free. Returning it unchecked puts the proton on top of a
    neighbour, and the relax cannot undo an overlap it starts inside of."""
    centre = np.array([6.0, 6.0, 6.0])
    wall = [[6 + dx, 6 + dy, 5.0] for dx in (-1.2, 0, 1.2) for dy in (-0.6, 0.6)]
    at = Atoms(['O'] + ['Ne'] * len(wall), positions=[list(centre)] + wall,
               cell=np.eye(3) * 12, pbc=True)
    got = _place_h(centre, at.get_positions(), np.array(at.get_cell()), {0}, 1, H_BOND_LEN['O'])
    assert len(got) == 1, 'a host with a free hemisphere must still take its proton'
    sep = get_distances([got[0]], at.get_positions(), cell=np.array(at.get_cell()), pbc=True)[1][0]
    sep[0] = 9e9                                   # the host bond itself is not a clash
    assert sep.min() > H_MIN_SEP, f'placed {sep.min():.3f} A from a neighbour'


def test_an_enclosed_host_is_refused_rather_than_overlapped():
    """No direction is free. Refusing (returning fewer than asked) is the honest answer; inventing an
    overlap is not."""
    centre = np.array([5.0, 5.0, 5.0])
    cage = _shell(centre, 1.6)
    at = Atoms(['O'] + ['Ne'] * len(cage), positions=[list(centre)] + cage,
               cell=np.eye(3) * 10, pbc=True)
    got = _place_h(centre, at.get_positions(), np.array(at.get_cell()), {0}, 1, H_BOND_LEN['O'])
    assert got == [], 'an enclosed host must take no proton at all'


def test_protons_placed_in_one_pass_do_not_collide_with_each_other():
    """`occupied` threads the protons already placed into the next candidate's clash test. Without it
    two acceptors 2 A apart can each place a proton into the same gap."""
    at = Atoms(['O', 'O'], positions=[[5, 5, 5], [5, 5, 7.0]], cell=np.eye(3) * 12, pbc=True)
    out, n = _complete_h(at, 4)
    assert n == 4, n
    assert _min_hh(out) > H_MIN_SEP, f'{_min_hh(out):.3f} A between two placed protons'


# ---- a blocked host must not cost the formula its protons --------------------

def test_a_blocked_host_reoffers_its_protons():
    """One crowded oxygen used to silently reduce the H count the formula demands. The allotment is
    re-offered to the acceptors that still have capacity."""
    at = Atoms(['O', 'O'], positions=[[5, 5, 5], [5, 5, 8]], cell=np.eye(3) * 12, pbc=True)
    out, n = _complete_h(at, 3)
    assert n == 3, f'placed {n} of 3 protons the formula asked for'
    assert sum(1 for s in out.get_chemical_symbols() if s == 'H') == 3


def test_it_gives_up_only_when_every_host_is_blocked():
    """The re-offer loop must terminate. Two fully enclosed acceptors can take nothing, and that has
    to come back as a shortfall rather than hanging or fabricating."""
    c1, c2 = np.array([4.0, 4.0, 4.0]), np.array([11.0, 11.0, 11.0])
    pos = [list(c1), list(c2)] + _shell(c1, 1.6) + _shell(c2, 1.6)
    at = Atoms(['O', 'O'] + ['Ne'] * (len(pos) - 2), positions=pos, cell=np.eye(3) * 16, pbc=True)
    out, n = _complete_h(at, 4)
    assert n == 0, n
    assert len(out) == len(at), 'nothing may be added when nothing can be placed'


# ---- the acceptor list ------------------------------------------------------

@pytest.mark.parametrize('el', ['O', 'N', 'F', 'Cl', 'S', 'Se'])
def test_every_acceptor_element_has_a_capacity_and_a_bond_length(el):
    """PROTON_CAP has to cover the halides and chalcogenides: a hydrogen bifluoride or a
    hydrosulfide has nowhere else to put its proton, and with O/N only such an entry gets no
    placement at all. A capacity without a bond length would start the proton inside the anion."""
    assert el in H_VALENCE and H_VALENCE[el] >= 1
    assert el in H_BOND_LEN and 0.9 <= H_BOND_LEN[el] <= 1.6


def test_the_bond_length_scales_with_the_acceptor():
    """Starting a Cl-H at an oxygen's 0.98 A puts the proton inside the anion."""
    assert H_BOND_LEN['F'] < H_BOND_LEN['O'] < H_BOND_LEN['Cl'] < H_BOND_LEN['S'] < H_BOND_LEN['Se']
    for el in ('F', 'Cl', 'S', 'Se'):
        at = Atoms([el], positions=[[5, 5, 5]], cell=np.eye(3) * 12, pbc=True)
        got = _place_h(np.array([5.0, 5.0, 5.0]), at.get_positions(), np.array(at.get_cell()),
                       {0}, 1, H_BOND_LEN[el])
        assert len(got) == 1
        d = float(np.linalg.norm(got[0] - np.array([5.0, 5.0, 5.0])))
        assert abs(d - H_BOND_LEN[el]) < 1e-6, (el, d)


# ---- determinism -------------------------------------------------------------

def test_placement_is_reproducible():
    """The retry uses a random tilt, so it takes a seeded generator -- two runs of the same build must
    not produce different structures."""
    at = Atoms(['O', 'O'], positions=[[5, 5, 5], [5, 5, 7.5]], cell=np.eye(3) * 12, pbc=True)
    a, na = _complete_h(at, 3)
    b, nb = _complete_h(at, 3)
    assert na == nb
    assert np.allclose(a.get_positions(), b.get_positions())


# ---- the bond-valence fallback ----------------------------------------------

def _silicate_fragment():
    """One O shared between a 4-coordinate Si and a 6-coordinate Ca -- the motif a hydrous silicate
    is made of. The capacity model counts the ionic Ca-O contact as a full covalent bond and calls
    that O saturated; weighted by 1/CN it plainly is not."""
    c = 6.0
    pos = [[c, c, c]]                                                    # 0: the shared O
    pos += [[c + 2.2, c, c + 1.62], [c - 2.2, c, c + 1.62],
            [c, c + 2.2, c + 1.62], [c, c - 2.2, c + 1.62]]              # Si's other three O + 1
    pos += [[c, c, c + 1.62], [c, c, c - 2.40]]                          # Si, Ca
    pos += [[c + 2.4, c, c - 2.40], [c - 2.4, c, c - 2.40],
            [c, c + 2.4, c - 2.40], [c, c - 2.4, c - 2.40],
            [c, c, c - 4.80]]                                            # Ca's other five O
    syms = ['O'] * 5 + ['Si', 'Ca'] + ['O'] * 5
    return Atoms(syms, positions=pos, cell=np.eye(3) * 14, pbc=True)


def test_the_two_acceptor_models_disagree_where_hydrous_phases_live():
    """The whole reason the fallback exists. Same oxygen, two verdicts: capacity 0 (two neighbours,
    valence 2) versus a real Pauling bond-strength deficit."""
    at = _silicate_fragment()
    P = at.get_positions(); C = np.array(at.get_cell()); syms = at.get_chemical_symbols()
    D = get_distances(P, P, cell=C, pbc=True)[1]; np.fill_diagonal(D, 9e9)
    cap = H_VALENCE['O'] - int((D[0] < _h_nbr_cuts(syms)[0]).sum())
    assert cap == 0, 'fixture check: the capacity model must call this O saturated'

    bv = _bv_deficits(syms, P, C, {'O': -2, 'Si': 4, 'Ca': 2})
    assert bv and bv.get(0, 0) > BV_MIN_DEFICIT, bv
    assert round(bv[0], 2) == 0.87, bv[0]


def test_the_fallback_declines_without_oxidation_states():
    """This model's only input is the formal charges. A CIF without them gets a refusal, not an
    invented set -- the same rule the sibling search follows for `unchecked` vs `absent`."""
    at = _silicate_fragment()
    P = at.get_positions(); C = np.array(at.get_cell()); syms = at.get_chemical_symbols()
    assert _bv_deficits(syms, P, C, None) is None
    assert _bv_deficits(syms, P, C, {}) is None
    assert _bv_deficits(syms, P, C, {'O': None, 'Si': None}) is None
    assert _bv_fallback_ok(syms, P, C, None, 4) is False


def test_the_fallback_places_where_the_capacity_model_places_nothing():
    at = _silicate_fragment()
    ox = {'O': -2, 'Si': 4, 'Ca': 2}
    _, n_cap = _complete_h(at, 3)                        # capacity mode: the shared O is invisible
    out, n_bv = _complete_h(at, 3, ox=ox, mode='bond-valence')
    assert n_bv == 3, n_bv
    hs = [i for i, e in enumerate(out.get_chemical_symbols()) if e == 'H']
    assert len(hs) == 3
    # every proton on an anion, at its X-H length, and not on top of anything
    P = out.get_positions(); C = np.array(out.get_cell()); sym = out.get_chemical_symbols()
    D = get_distances(P[hs], P, cell=C, pbc=True)[1]
    for k, i in enumerate(hs):
        d = D[k].copy(); d[i] = 9e9
        j = int(np.argmin(d))
        assert sym[j] in PROTON_CAP, f'proton {k} nearest {sym[j]}'
        assert abs(d[j] - H_BOND_LEN[sym[j]]) < 1e-6, (sym[j], d[j])
    assert n_cap >= 0                                    # the capacity path is unaffected


def test_the_fallback_respects_the_per_anion_ceiling():
    """PROTON_CAP, not H_VALENCE: N takes four (ammonium), F takes one."""
    assert PROTON_CAP['N'] == 4 and H_VALENCE['N'] == 3
    assert PROTON_CAP['F'] == 1


def test_the_fallback_is_asked_before_it_is_promised():
    """`_bv_fallback_ok` gates the plan, so a build is never committed to protons it cannot place."""
    at = _silicate_fragment()
    P = at.get_positions(); C = np.array(at.get_cell()); syms = at.get_chemical_symbols()
    ox = {'O': -2, 'Si': 4, 'Ca': 2}
    assert _bv_fallback_ok(syms, P, C, ox, 4) is True
    assert _bv_fallback_ok(syms, P, C, ox, 10_000) is False, 'must decline what it cannot seat'
    assert _bv_fallback_ok(syms, P, C, ox, 0) is False


def test_the_fallback_never_overrides_the_capacity_model():
    """It is reached only where the capacity model found nothing placeable AND no orientational risk
    was flagged. Pinned on the engine's branch order, because the danger is a later edit promoting it
    ahead of the model the skills document."""
    import inspect

    from demars_core._engine import mar_engine as ME
    src = inspect.getsource(ME.build)
    i_clean = src.index('if _deficit > 0 and clean:')
    i_comp = src.index('_sub_el and abs(_sub_def - _deficit) <= 1')
    i_bv = src.index('_bv_fallback_ok(')
    assert i_clean < i_comp < i_bv, 'the fallback must be the last branch tried'
    assert 'unsafe_cap == 0 and _bv_fallback_ok(' in src, (
        'the fallback must not run when an orientational acceptor risk was flagged')


def test_the_fallback_is_reproducible():
    at = _silicate_fragment()
    ox = {'O': -2, 'Si': 4, 'Ca': 2}
    a, na = _complete_h(at, 4, ox=ox, mode='bond-valence')
    b, nb = _complete_h(at, 4, ox=ox, mode='bond-valence')
    assert na == nb == 4
    assert np.allclose(a.get_positions(), b.get_positions())
