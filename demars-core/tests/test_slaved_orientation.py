"""Slaved-pair and orientation-unit detection (`mar_evidence.slaved_units` / `orientation_units`).

Why they exist: independent per-site decoration breaks a bonded or an orientational unit, so a
non-zero count is a reason to refuse a plain enumeration. `taxonomy.md` defines class E — Slaved,
and these two are what compute it.

Both emit neutral facts. No MLIP, no GPU, no network.
"""
import os

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from demars_core._engine.mar_evidence import (ORIENT_OCC, SLAVE_BOND, SLAVE_SCORE, evidence_from_text,
                                              orientation_units, rigid_units, slaved_units)

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')


def _ev(name):
    with open(os.path.join(FIX, name), encoding='utf-8') as fh:
        return evidence_from_text(fh.read(), iid=None, search_siblings=False)


# ---- slaved pairs on a real structure ---------------------------------------

def test_a_partial_anion_orbit_that_only_exists_where_its_parent_does():
    """The oxygen-deficient ferrite fixture: half-occupied Fe with the O sites that complete its
    coordination. Cross-checked against `cross_orbit_contacts`, computed independently — the Fe-O
    separation it reports must be inside the window this detector paired them through."""
    d = _ev('cod_1544358.cif')['disorder']
    su = d['slaved_units']
    assert su, 'the coupled Fe/O orbits of this fixture must be seen'
    for u in su:
        assert u['trigger_element'] == 'Fe' and u['follower_element'] == 'O', u
        assert u['n_parent_positions'] > 0 and u['followers_per_parent'] >= 1
    assert {u['followers_per_parent'] for u in su} == {1, 2}, su

    lo, hi = SLAVE_BOND
    fe_o = [c['d_A'] for c in d['cross_orbit_contacts_below_2.6A']
            if set(c['orbit_a']) | set(c['orbit_b']) == {'Fe', 'O'}]
    assert any(lo <= dd <= hi for dd in fe_o), (
        f'no independently measured Fe-O contact inside {SLAVE_BOND}: {fe_o}')


def test_the_follower_never_outranks_its_parent():
    """H follows O follows a metal, never the reverse. Without the ordering the same pair is reported
    twice, once in each direction, and neither says which atom is placed by the other."""
    assert SLAVE_SCORE['H'] > SLAVE_SCORE['O'] > 0
    for u in _ev('cod_1544358.cif')['disorder']['slaved_units']:
        f = max(SLAVE_SCORE.get(e, 0) for e, _o in u['follower_signature'])
        p = max(SLAVE_SCORE.get(e, 0) for e, _o in u['parent_signature'])
        assert f > p, u


def test_a_full_parent_is_not_a_parent():
    """The false-positive guard: a FULL orbit encodes no choice, so an orbit near it is not slaved to
    it. Without this a 0.88-occupied oxygen orbit reads as slaved to the full oxygen orbit."""
    lat = Lattice.cubic(9.0)
    s = Structure(lat, [{'Fe': 1.0}, {'O': 0.5}],
                  [[0, 0, 0], [0.2, 0, 0]])              # Fe-O 1.8 A, inside SLAVE_BOND
    assert slaved_units(s) == [], 'a full parent must never be paired'

    s2 = Structure(lat, [{'Fe': 0.5}, {'O': 0.5}], [[0, 0, 0], [0.2, 0, 0]])
    assert slaved_units(s2), 'the same geometry with a PARTIAL parent is the real case'


def test_an_occupancy_that_does_not_match_is_not_a_pair():
    """A follower orbit is slaved when its occupancy tracks the parent's. 0.5 against 0.1 does not."""
    lat = Lattice.cubic(9.0)
    s = Structure(lat, [{'Fe': 0.1}, {'O': 0.5}], [[0, 0, 0], [0.2, 0, 0]])
    assert slaved_units(s) == []


def test_a_parent_without_a_follower_breaks_the_pairing():
    """EVERY parent position must find a follower; one that does not means the bond is not what holds
    the orbit. Two Fe, one O in reach."""
    lat = Lattice.cubic(12.0)
    s = Structure(lat, [{'Fe': 0.5}, {'Fe': 0.5}, {'O': 0.5}],
                  [[0, 0, 0], [0.5, 0.5, 0.5], [0.15, 0, 0]])
    assert slaved_units(s) == []


# ---- orientation units -------------------------------------------------------

def _rotor():
    """A full O anchor with four half-occupied O alternatives around it, mutually in reach: the
    choice is which pair is occupied, not which atom."""
    lat = Lattice.cubic(10.0)
    return Structure(lat, [{'O': 1.0}, {'O': 0.5}, {'O': 0.5}, {'O': 0.5}, {'O': 0.5}],
                     [[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [-0.2, 0, 0], [0, -0.2, 0]])


def test_an_anchored_partial_orbit_is_an_orientation_unit():
    ou = orientation_units(_rotor())
    assert len(ou) == 1, ou
    u = ou[0]
    assert u['element'] == 'O' and u['occupancy'] == 0.5
    assert u['n_anchors'] == 1
    assert u['positions_per_anchor'] == 2, u          # k = round(0.5 * 4)
    assert u['alternatives_per_anchor'] >= 2, 'one alternative is no choice at all'


@pytest.mark.parametrize('occ', [0.1, 0.9])
def test_an_occupancy_outside_the_window_is_a_vacancy_sublattice_not_a_rotor(occ):
    """A 0.1 or a 0.9 orbit is dilution or a near-full site, not a set of discrete orientations."""
    lat = Lattice.cubic(10.0)
    s = Structure(lat, [{'O': 1.0}] + [{'O': occ}] * 4,
                  [[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [-0.2, 0, 0], [0, -0.2, 0]])
    assert orientation_units(s) == []
    assert not (ORIENT_OCC[0] <= occ <= ORIENT_OCC[1])


def test_no_anchor_means_no_orientation_unit():
    """The anchor is what the alternatives are alternatives AROUND. Without a full orbit of the same
    element there is nothing to orient about, and the orbit is a plain partial sublattice."""
    lat = Lattice.cubic(10.0)
    s = Structure(lat, [{'O': 0.5}] * 4,
                  [[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [0.2, 0.2, 0]])
    assert orientation_units(s) == []


def test_the_two_unit_detectors_see_different_things():
    """`rigid_units` keys on a FORMER centre and its partial light ligand shell; this one on a partial
    orbit anchored by a FULL orbit of the SAME element. Neither subsumes the other, which is why both
    are reported."""
    rotor = _rotor()
    assert orientation_units(rotor) and not rigid_units(rotor), (
        'no distinct centre element here, so the former-centred detector cannot see it')

    d = _ev('cod_4000330.cif')['disorder']            # split tunnel sites around former centres
    assert d['rigid_unit_signal'] and not d['orientation_units']


# ---- what the bundle carries -------------------------------------------------

def test_both_counts_reach_the_evidence_bundle():
    """The analyst reads stage 1a to pick the mechanism class, and class E is decided there."""
    d = _ev('cod_1544358.cif')['disorder']
    for key in ('slaved_units', 'orientation_units', 'rigid_unit_signal'):
        assert key in d and isinstance(d[key], list)


def test_the_detection_runs_once_per_structure():
    """`orientation_units` needs the slaved result to skip those orbits. Recomputing it per consumer
    ran the whole pairing search three times for every evidence bundle."""
    import inspect

    from demars_core._engine import mar_evidence as MEV
    src = inspect.getsource(MEV.evidence_from_text)
    assert src.count('slaved_units(s)') == 1, 'the pairing search must run once'
    assert 'orientation_units(s, _slaved)' in src
    assert 'slaved' in inspect.signature(MEV.orientation_units).parameters
