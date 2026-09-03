"""Pluggable MLIP backend -- the abstraction that lets DeMARS run on ANY
universal potential, not just our SevenNet/OEQ install.

Two backends:

  * "sevennet" (or "7net-nano"/"7net-omni"/a checkpoint path): the native, fast
    path. The engine's own batched GPU relaxer (torch-sim, via
    demars_core._torchsim.batched_fire_relax) is used unchanged -- this is DeMARS's canonical nano-sampling -> omni-mpa-final
    tiering. Nothing is patched.

  * ANY ase.calculators Calculator instance (MACE, CHGNet, ORB, a bespoke
    SevenNetCalculator with a user checkpoint, ...): the generic path. We route
    the engine's relaxer through a serial ASE FIRE relaxer bound to that
    calculator, so a third party with a different potential is a first-class
    citizen. Slower (no batching), but universal.

The generic path is what makes the interface `deaverage(struct, calculator=...)`
honest: the engine never learns which potential it is talking to.
"""
import warnings

import numpy as np

__all__ = ['is_sevennet_spec', 'parse_sevennet_spec', 'ase_relax_batch']

_SEVENNET_KEYWORDS = ('sevennet', '7net', '7net-nano', 'nano',
                      '7net-omni', 'omni')
# SevenNet-Nano's cutoff variants: 7net-nano-4.5 / -5.0 / -5.5 / -6.0
_SEVENNET_PREFIXES = ('7net-nano-',)


def is_sevennet_spec(calculator):
    """True if `calculator` names the native SevenNet path (a keyword or a
    checkpoint path string), False if it is an ASE Calculator instance to route
    through the generic path."""
    if isinstance(calculator, str):
        low = calculator.lower()
        if low in _SEVENNET_KEYWORDS or low.startswith(_SEVENNET_PREFIXES):
            return True
        # a checkpoint path (…​.pth/.pt) also goes native via use_model
        return low.endswith(('.pth', '.pt'))
    return False


def parse_sevennet_spec(calculator):
    """-> (model, modal) for the backend's use_model. 'sevennet' defaults to nano.

    Nano is single-task (distilled from Omni's mpa), so it takes no modal; Omni is
    multi-modal and is pinned to 'mpa' -- DeMARS's final tier.
    """
    low = str(calculator).lower()
    if low in ('sevennet', '7net-nano', 'nano', ''):
        return '7net-nano', None
    if low.startswith(_SEVENNET_PREFIXES):
        return low, None               # explicit nano cutoff variant, no modal
    if low in ('7net-omni', 'omni'):
        return '7net-omni', 'mpa'
    return calculator, None            # explicit checkpoint path or keyword


def _cell_filter(atoms):
    """Wrap atoms so FIRE relaxes the cell too (variable-cell, like the batched backend).
    Falls back across ASE versions; positions-only if no filter is available."""
    try:
        from ase.filters import FrechetCellFilter
        return FrechetCellFilter(atoms)
    except Exception:
        try:
            from ase.constraints import ExpCellFilter
            return ExpCellFilter(atoms)
        except Exception:
            return atoms


def _rigid_pairs(atoms, constrain):
    """-> [(i, j), ...] bond pairs to hold rigid in the pre-stage, or [] for none.

    `constrain` is None (no pre-stage), 'auto' (detect discrete former-ligand units from the
    structure itself -- no per-compound table; see mar_engine.rigid_unit_bonds), or a
    callable(atoms) -> pairs for a caller that knows the units better than the detector.
    """
    if constrain is None:
        return []
    if callable(constrain):
        return list(constrain(atoms) or [])
    if isinstance(constrain, str) and constrain.lower() == 'auto':
        from ._engine.mar_engine import rigid_unit_bonds
        return rigid_unit_bonds(atoms)[0]
    raise ValueError(f"constrain must be None, 'auto', or a callable -- got {constrain!r}")


def _fix_bond_lengths(pairs, tolerance=1e-6, maxiter=5000):
    """ASE's FixBondLengths with the per-iteration minimum-image work hoisted out of the loop.

    Same constraint, same arithmetic, same numbers -- only the redundancy is gone. Upstream's
    RATTLE loops recompute, on EVERY one of up to `maxiter` sweeps, a `find_mic()` per pair that
    depends solely on `atoms.positions`, `atoms.cell` and `atoms.pbc` -- none of which that loop
    touches (it writes `p`, or `new`, and reads `old`). So the identical call is made maxiter x
    npairs times where npairs would do, and `find_mic` is the expensive part.

    It matters because the cost lands on the rigid-unit pre-stage, which is charged per force
    evaluation via adjust_forces -> adjust_momenta. On a molecular crystal of a few hundred atoms
    with a few hundred constrained pairs, three independent stack samples of the running relaxation
    were ALL inside these two loops, with the GPU at 0% -- a 200-step pre-stage outweighing the
    5000-step free relaxation it exists to protect.

    Hoisting is exact, not an approximation: per pair, the invariants are the mic vector, the
    reduced mass and the reference bond length, and each sweep's update reads only those plus the
    array it is correcting. Vectorising the mic call over all pairs at once is a second, larger
    win, since `find_mic` is written for arrays.
    """
    from ase.constraints import FixBondLengths
    from ase.geometry import find_mic

    class _Hoisted(FixBondLengths):
        def _invariants(self, atoms):
            """-> (mic vectors, reduced masses, bond lengths) -- everything the sweep re-reads."""
            if self.bondlengths is None:
                self.bondlengths = self.initialize_bond_lengths(atoms)
            old, masses = atoms.positions, atoms.get_masses()
            a, b = self.pairs[:, 0], self.pairs[:, 1]
            d, _ = find_mic(old[a] - old[b], atoms.cell, atoms.pbc)   # one call, all pairs
            m = 1 / (1 / masses[a] + 1 / masses[b])
            return d, m, self.bondlengths

        def adjust_momenta(self, atoms, p):
            D, M, CD = self._invariants(atoms)
            masses = atoms.get_masses()
            for _ in range(self.maxiter):
                converged = True
                for j, (a, b) in enumerate(self.pairs):
                    d = D[j]
                    dv = p[a] / masses[a] - p[b] / masses[b]
                    x = -np.dot(dv, d) / CD[j] ** 2
                    if abs(x) > self.tolerance:
                        p[a] += x * M[j] * d
                        p[b] -= x * M[j] * d
                        converged = False
                if converged:
                    return
            raise RuntimeError('Did not converge')

        def adjust_positions(self, atoms, new):
            D, M, CD = self._invariants(atoms)
            old, masses = atoms.positions, atoms.get_masses()
            R0 = old[self.pairs[:, 0]] - old[self.pairs[:, 1]]
            for _ in range(self.maxiter):
                converged = True
                for j, (a, b) in enumerate(self.pairs):
                    d0 = D[j]
                    d1 = new[a] - new[b] - R0[j] + d0
                    x = 0.5 * (CD[j] ** 2 - np.dot(d1, d1)) / np.dot(d0, d1)
                    if abs(x) > self.tolerance:
                        new[a] += x * M[j] / masses[a] * d0
                        new[b] -= x * M[j] / masses[b] * d0
                        converged = False
                if converged:
                    return
            raise RuntimeError('Did not converge')

    c = _Hoisted(pairs, tolerance=tolerance)
    c.maxiter = maxiter
    return c


def ase_relax_batch(atoms_list, calc, fmax=0.05, steps=None, *,
                    constrain=None, prestage_steps=200):
    """Serial FIRE relaxation with an arbitrary ASE calculator, returning the
    SAME (E, rel, val) contract as _torchsim.batched_fire_relax:
        E   : np.ndarray of TOTAL potential energies (eV), one per input
        rel : list of relaxed ase.Atoms
        val : list of bool (False on a relaxation failure)

    Parameters
    ----------
    constrain : None | 'auto' | callable(atoms) -> [(i, j), ...]
        Optional RIGID-UNIT pre-stage. When it yields bond pairs, each structure is relaxed in two
        stages: (1) those bond LENGTHS held by ase.constraints.FixBondLengths at FIXED CELL, so a
        discrete unit (SO4, PO4, BO3, MoO6, NH4, ...) can rotate and translate as a rigid body while
        the surrounding sublattice relaxes around it, but cannot be torn apart; (2) constraints
        released and the normal variable-cell relax run to convergence.

        This exists because a freshly decorated cell places a unit's ligands at their built offsets
        while neighbouring cations are still at averaged CIF positions -- sometimes inside bonding
        range. Relaxing freely from there lets a cation pull a ligand off its centre; the unit breaks
        and only partly heals, which charge and fidelity (composition-only) both wave through.

        'auto' derives the units from the structure -- covalent-radii bond detection plus the
        coordination number actually observed -- so it generalises across oxo-anions, molecular ions
        and octahedral frameworks alike, and returns nothing (hence a plain free relax, unchanged
        behaviour) for structures that hold no discrete unit.
    prestage_steps : int
        Optimizer steps for the constrained stage. 0 disables the pre-stage even if pairs are found.

    NOTE: this is the ASE path. The batched torch-sim backend has no holonomic bond constraint, so a
    build that needs the pre-stage must be relaxed through here (pass an ASE calculator instance).
    """
    from ase.optimize import FIRE
    # one budget rule for both relax paths; one definition of "not a relaxation failure"
    from ._torchsim import _auto_steps, _is_infra_error
    E, rel, val = [], [], []
    for at in atoms_list:
        a = at.copy()
        a.calc = calc
        nsteps = _auto_steps(len(a)) if steps is None else steps
        try:
            pairs = _rigid_pairs(a, constrain)
            if pairs and prestage_steps > 0:
                a.set_constraint(_fix_bond_lengths(pairs, tolerance=1e-6, maxiter=5000))
                # positions only -- no cell filter, so the unit reorients inside a frozen lattice
                FIRE(a, logfile=None).run(fmax=fmax, steps=prestage_steps)
                a.set_constraint()          # release; the free stage relaxes cell + every DOF
            filt = _cell_filter(a)
            opt = FIRE(filt, logfile=None)
            opt.run(fmax=fmax, steps=nsteps)
            # Reaching the step ceiling is NOT convergence. This path used to append val=True
            # unconditionally, so a structure that merely ran out of steps entered the ensemble
            # as if it were a minimum -- the mirror image of the batched path, which drops such
            # a config as if the STRUCTURE were at fault. Both were silent. Match the batched
            # contract (val=False) and say which of the two it was.
            #
            # Decide from the residual force, NOT from the optimizer's own convergence
            # predicate: its signature is not stable across ASE releases (on ase 3.26 it wants a
            # `gradient` argument, having moved onto `Optimizable` in later releases). Calling it
            # bare raised TypeError there, the except below swallowed it, and EVERY config came
            # back val=False with no warning -- an environment fault reported as chemistry. The
            # same criterion FIRE itself uses, computed here, cannot drift with the dependency.
            gmax = float(np.sqrt((np.asarray(filt.get_forces()) ** 2).sum(axis=1)).max())
            if gmax > fmax:
                mf = float(np.sqrt((a.get_forces() ** 2).sum(axis=1)).max())
                a.info['relax_converged'] = False
                a.info['relax_max_force'] = mf
                a.info['relax_steps_budget'] = int(nsteps)
                warnings.warn(f'FIRE did not reach fmax={fmax} within steps={nsteps} '
                              f'(residual max force {mf:.3f} eV/A); reported as invalid. A '
                              f'step-ceiling hit is NOT evidence that the structure cannot relax.',
                              stacklevel=2)
                E.append(float('nan'))
                rel.append(a)
                val.append(False)
                continue
            E.append(float(a.get_potential_energy()))
            rel.append(a)
            val.append(True)
        except Exception as exc:
            # Same contract as the batched path: an unusable environment (missing dependency,
            # dead GPU, a dependency whose API moved) is NOT a statement about this structure,
            # so it propagates instead of being written into the record as val=False. A genuine
            # per-config failure still maps to val=False -- but never silently.
            if _is_infra_error(exc):
                raise
            warnings.warn(f'relaxation failed for one config ({type(exc).__name__}: {exc}); '
                          f'reported as invalid', stacklevel=2)
            E.append(float('nan'))
            rel.append(at)
            val.append(False)
    return np.asarray(E), rel, val
