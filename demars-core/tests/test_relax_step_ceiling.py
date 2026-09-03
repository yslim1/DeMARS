"""A relaxation that merely ran out of optimizer steps must SAY SO (no MLIP / no GPU).

Both relax paths bound the work with `steps` (400 by default), and both used to treat hitting that
ceiling silently -- in OPPOSITE directions:

  * `calculators.ase_relax_batch` appended `val=True` unconditionally, so an unconverged geometry
    entered the ensemble as if it were a minimum, and its energy was quoted as a relaxed energy.
  * `_torchsim.batched_fire_relax` set `val=False` with no explanation, which reads downstream as
    "this structure does not relax" -- a claim about the chemistry -- when the truth was "the budget
    ran out", a claim about the run.

Neither is recoverable from the artifacts afterwards, which is the silent-wrong-value class this
repo keeps tripping over. The fix is not a better number; it is that the distinction survives into
`atoms.info` (a warning dies with the terminal, the run dir does not).

These tests use a hand-written ASE calculator with a constant non-zero force, so FIRE provably
cannot converge and no MLIP is needed.
"""
import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from demars_core.calculators import ase_relax_batch


class ConstantPull(Calculator):
    """Every atom feels the same constant force -- a potential with no minimum to find."""

    implemented_properties = ['energy', 'forces', 'stress']

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        n = len(atoms)
        f = np.zeros((n, 3))
        f[:, 0] = 3.0                      # 3 eV/A along x, far above any sane fmax
        self.results = {'energy': -1.0 * n, 'forces': f, 'stress': np.zeros(6)}


def _two_atoms():
    return Atoms('Si2', positions=[[0, 0, 0], [2.3, 0, 0]], cell=[12.0] * 3, pbc=True)


def test_step_ceiling_is_reported_not_passed_off_as_a_relaxed_minimum():
    """The regression: this path used to return val=True for a structure it never converged."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        E, rel, val = ase_relax_batch([_two_atoms()], ConstantPull(), fmax=0.05, steps=2)

    assert val[0] is False or val[0] == False, (            # noqa: E712 -- list-of-bool contract
        'an unconverged relaxation is being reported as valid; its energy would enter the '
        'ensemble as if it were a minimum')
    assert np.isnan(E[0]), 'a non-minimum energy must not be handed back as a number'
    assert any('step' in str(w.message).lower() for w in caught), \
        'hitting the step ceiling produced no warning'


def test_the_reason_survives_in_atoms_info_not_only_in_a_warning():
    """A warning is gone once the terminal scrolls; the run dir is what an auditor reads."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _E, rel, _val = ase_relax_batch([_two_atoms()], ConstantPull(), fmax=0.05, steps=2)

    info = rel[0].info
    assert info.get('relax_converged') is False
    assert info['relax_steps_budget'] == 2, 'the budget that was hit must be recorded'
    assert info['relax_max_force'] > 0.05, info            # the residual force, so "how far off?"


def test_a_converged_relaxation_is_not_annotated():
    """The stamp must mean something: it may not appear on structures that DID converge."""
    class AlreadyThere(ConstantPull):
        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            self.results = {'energy': -2.0 * len(atoms),
                            'forces': np.zeros((len(atoms), 3)), 'stress': np.zeros(6)}

    E, rel, val = ase_relax_batch([_two_atoms()], AlreadyThere(), fmax=0.05, steps=50)
    assert val[0] is True or val[0] == True                # noqa: E712
    assert np.isfinite(E[0])
    assert 'relax_converged' not in rel[0].info
    assert 'relax_max_force' not in rel[0].info


def test_auto_step_budget_scales_with_size_but_is_bounded_at_both_ends():
    """The cap only ever binds on structures that DON'T converge, so its job is deciding how long
    to hold a stuck relaxation. Floor = the historical 400 (never regress); ceiling bounds the
    stuck tail so one pathological large cell cannot eat 20k steps in a 1000-entry campaign.

    The ceiling is a tuning knob and this test pins it deliberately: changing it changes how many
    configs come back val=False, so it should not move without someone noticing here."""
    from demars_core._torchsim import _auto_steps, _STEPS_MIN, _STEPS_MAX

    assert _auto_steps(1) == _STEPS_MIN, 'a tiny cell must not get a budget below the old default'
    assert _auto_steps(30) == 600 and _auto_steps(45) == 900           # 20 * N in the linear band
    assert _auto_steps(50) == _STEPS_MAX and _auto_steps(5000) == _STEPS_MAX
    assert _STEPS_MIN == 400 and _STEPS_MAX == 1000
    assert all(_auto_steps(n) <= _auto_steps(n + 1) for n in (1, 20, 63, 249, 250, 999))


def test_both_relax_paths_share_one_budget_rule():
    """A per-path budget would make the same structure converge on one backend and fail on the
    other -- a difference in the ANSWER produced by a difference in plumbing."""
    import inspect

    from demars_core import calculators as C
    from demars_core._torchsim import batched_fire_relax

    assert inspect.signature(batched_fire_relax).parameters['steps'].default is None
    assert inspect.signature(C.ase_relax_batch).parameters['steps'].default is None
    assert '_auto_steps' in inspect.getsource(C.ase_relax_batch), \
        'the ASE path must use the same budget rule, not a second copy of the numbers'


def test_an_explicit_steps_argument_still_wins():
    """Callers that know better (a screening pass, a test) must keep control."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _E, rel, _val = ase_relax_batch([_two_atoms()], ConstantPull(), fmax=0.05, steps=3)
    assert rel[0].info['relax_steps_budget'] == 3


def test_a_hard_failure_is_still_distinguishable_from_a_step_ceiling():
    """val=False has two causes now; only one of them carries the ceiling stamp."""
    class Explodes(Calculator):
        implemented_properties = ['energy', 'forces', 'stress']

        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            raise RuntimeError('calculator is broken')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _E, rel, val = ase_relax_batch([_two_atoms()], Explodes(), fmax=0.05, steps=2)

    assert val[0] is False or val[0] == False              # noqa: E712
    assert 'relax_converged' not in rel[0].info, (
        'a crashed calculator must not be labelled as a step-ceiling hit -- they need '
        'different fixes (fix the environment vs. raise `steps`)')


def test_the_ceiling_verdict_does_not_go_through_the_optimizers_own_predicate():
    """The verdict may not depend on `Optimizer.converged()`.

    Measured on this machine: on ase 3.26 that method requires a `gradient` argument (convergence
    moved onto `Optimizable` in later releases), so `opt.converged()` raised TypeError, the
    except-clause swallowed it, and EVERY config came back val=False with no warning and no stamp
    -- a dependency version silently deciding a chemical result. ase 3.29 answers the same call
    fine, which is exactly why a suite that runs in one environment could not see it.

    That also rules out testing this behaviourally from inside one environment, so pin the wiring:
    the verdict is computed from the residual force, the same criterion FIRE itself applies.
    """
    import inspect

    from demars_core import calculators as C

    src = inspect.getsource(C.ase_relax_batch)
    assert 'opt.converged()' not in src, (
        'the step-ceiling verdict is back on the optimizer predicate, whose signature is not '
        'stable across ASE releases; an upgrade would turn every config invalid without saying so')
    decision = src.split('opt.run(', 1)[1]
    assert 'get_forces()' in decision.split('relax_converged', 1)[0], \
        'the convergence decision must be read off the forces the optimizer left behind'


def test_an_environment_fault_is_not_recorded_as_an_invalid_structure():
    """`except Exception -> val=False` laundered missing dependencies and dead GPUs into a
    chemical claim ("these structures do not relax"). Infra faults propagate; the batched path
    already had this contract and the serial path did not."""
    class NoBackend(Calculator):
        implemented_properties = ['energy', 'forces', 'stress']

        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            raise ImportError('No module named sevenn')

    with pytest.raises(ImportError):
        ase_relax_batch([_two_atoms()], NoBackend(), fmax=0.05, steps=2)


def test_a_per_config_failure_is_still_val_false_but_says_so():
    """The other half of the same contract: a broken structure stays a per-config failure, and
    silence is what made it indistinguishable from an environment fault."""
    class Explodes(Calculator):
        implemented_properties = ['energy', 'forces', 'stress']

        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            raise ValueError('this geometry is degenerate')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _E, _rel, val = ase_relax_batch([_two_atoms()], Explodes(), fmax=0.05, steps=2)

    assert val[0] is False or val[0] == False               # noqa: E712
    assert any('reported as invalid' in str(w.message) for w in caught), \
        'a config dropped from the ensemble must leave a trace'


def test_the_step_budget_is_a_parameter_not_only_a_constant():
    """`--max-steps` exists so the ceiling can be tuned per run instead of by editing a constant.

    Every other engine knob (--nr, --min-nm, --excl, --same-excl, --couple-cut, --rounds) is a
    flag; the step budget was the one setting that could only be changed in source, which means
    changing it mid-campaign is a code change rather than a recorded run parameter.
    """
    import inspect

    from demars_core.api import _make_relax, deaverage

    assert 'max_steps' in inspect.signature(deaverage).parameters
    assert 'max_steps' in inspect.signature(_make_relax).parameters

    src = inspect.getsource(_make_relax)
    body = '\n'.join(ln.split('#', 1)[0] for ln in src.splitlines())
    assert body.count('steps=max_steps') == 3, \
        'all three relax paths (batched, rigid-ASE, generic-ASE) must honour the override'


def test_the_chosen_budget_is_recorded_in_the_engine_contract():
    """A run whose budget differs from the default must say so in engine.json, not only in prose.

    The `--same-excl` override is not carried in generation_recipe, and a reviewer showed the
    consequence: re-deriving that build from its machine-readable fields reproduces the default
    (defective) run. This is the same failure, so the flag lands in sampling.max_steps with the
    source of the value next to it.
    """
    import pathlib

    src = pathlib.Path('tools/demars_engine.py').read_text()
    body = '\n'.join(ln.split('#', 1)[0] for ln in src.splitlines())
    assert "'--max-steps'" in body, 'the flag is not exposed on the CLI'
    assert "sampling'\]\['max_steps'\]" in body.replace('\\', '') or "['max_steps']" in body, \
        'the chosen budget is not written into engine.json sampling'
    assert 'override --max-steps' in body and 'auto: clip(' in body, \
        'the record must say whether the budget was chosen or defaulted'
