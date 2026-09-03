"""The rigid-unit pre-stage must be cheap, and cheap WITHOUT changing the constraint.

The pre-stage exists to stop a freshly decorated cell from tearing a discrete unit apart while
the sublattice relaxes around it. On a molecular crystal it was outweighing the free relaxation
it protects: on a methylammonium-caged cyanide (576 atoms, 32 cages -> 320 constrained pairs) three
stack samples of the live run were all inside FixBondLengths, GPU at 0%.

The cause was redundancy, not tolerance: upstream recomputes a per-pair `find_mic()` on every one
of up to `maxiter` RATTLE sweeps, from inputs the sweep never modifies. So the test that matters
is not "is it faster" alone -- it is "is it faster AND numerically the same as upstream". A
speedup bought by loosening the constraint would pass a timing test and silently change chemistry.
"""
import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixBondLengths

from demars_core.calculators import _fix_bond_lengths


def _cell():
    """A periodic cell of diatomic units, far enough apart to be independent."""
    rng = np.random.default_rng(0)
    pos, pairs = [], []
    for i in range(6):
        c = np.array([2.0 + 3.0 * (i % 3), 2.0 + 3.0 * (i // 3), 2.0])
        pos += [c, c + np.array([0.0, 0.0, 1.1])]
        pairs.append((2 * i, 2 * i + 1))
    at = Atoms('CO' * 6, positions=np.array(pos), cell=[12.0, 12.0, 9.0], pbc=True)
    at.positions += 0.05 * rng.standard_normal(at.positions.shape)
    return at, pairs


def test_the_hoisted_constraint_is_arithmetically_the_same_as_upstream():
    at, pairs = _cell()
    rng = np.random.default_rng(1)
    p = rng.standard_normal((len(at), 3))
    new = at.positions + 0.02 * rng.standard_normal(at.positions.shape)

    up, ours = FixBondLengths(pairs, tolerance=1e-6), _fix_bond_lengths(pairs, tolerance=1e-6)
    up.maxiter = 5000

    p_up, p_ours = p.copy(), p.copy()
    up.adjust_momenta(at, p_up)
    ours.adjust_momenta(at, p_ours)
    assert np.allclose(p_up, p_ours, rtol=0, atol=1e-12)

    n_up, n_ours = new.copy(), new.copy()
    up.adjust_positions(at, n_up)
    ours.adjust_positions(at, n_ours)
    assert np.allclose(n_up, n_ours, rtol=0, atol=1e-12)


def test_the_minimum_image_work_no_longer_scales_with_the_sweep_count():
    """The defect itself: find_mic called maxiter x npairs times where npairs would do."""
    at, pairs = _cell()
    p = np.random.default_rng(2).standard_normal((len(at), 3))

    import ase.constraints.fix_bond_lengths as up_mod
    import demars_core.calculators as our_mod

    calls = {'up': 0, 'ours': 0}

    def _counted(which, real):
        def f(*a, **k):
            calls[which] += 1
            return real(*a, **k)
        return f

    real_up = up_mod.find_mic
    up_mod.find_mic = _counted('up', real_up)
    try:
        c = FixBondLengths(pairs, tolerance=1e-6)
        c.maxiter = 5000
        c.adjust_momenta(at, p.copy())
    finally:
        up_mod.find_mic = real_up

    import ase.geometry as geo
    real_ours = geo.find_mic
    geo.find_mic = _counted('ours', real_ours)
    try:
        our_mod._fix_bond_lengths(pairs, tolerance=1e-6).adjust_momenta(at, p.copy())
    finally:
        geo.find_mic = real_ours

    # one vectorised call covers every pair, whatever the sweep count
    assert calls['ours'] == 1
    # and upstream needed strictly more -- it pays per pair, per sweep
    assert calls['up'] > calls['ours']


def test_the_pre_stage_still_holds_the_unit_together():
    """A speedup that stopped constraining would pass the timing test and break the chemistry."""
    at, pairs = _cell()
    d0 = [at.get_distance(i, j, mic=True) for i, j in pairs]

    c = _fix_bond_lengths(pairs, tolerance=1e-6)
    c.bondlengths = np.array(d0)
    new = at.positions + np.array([[0.3, -0.2, 0.1]] * len(at))
    new[1] += 0.4                      # try to stretch the first unit
    c.adjust_positions(at, new)

    at2 = at.copy()
    at2.positions = new
    d1 = [at2.get_distance(i, j, mic=True) for i, j in pairs]
    assert np.allclose(d0, d1, atol=1e-4)


@pytest.mark.parametrize('meth', ['adjust_momenta', 'adjust_positions'])
def test_non_convergence_is_still_an_error_not_a_silent_pass(meth):
    at, pairs = _cell()
    c = _fix_bond_lengths(pairs, tolerance=1e-30, maxiter=2)
    arg = (np.random.default_rng(3).standard_normal((len(at), 3)) if meth == 'adjust_momenta'
           else at.positions + 0.3)
    with pytest.raises(RuntimeError, match='Did not converge'):
        getattr(c, meth)(at, arg)
