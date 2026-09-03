"""A round that dies at config 29/30 must not throw away 29 (defect D14).

Measured twice in the set-B campaign: one entry relaxed for 1 h 08 m and left nothing but a stale
`engine.json`; another was retried under a timeout it could not meet and made ZERO net progress
across several restarts, because every attempt started from an empty directory.

The property under test is not "a file appears" but **partial work survives a kill and is reused**
-- and, just as important, that reuse can never be silent theft from a different run: a different
potential, a different decoration or a failed relaxation must all MISS the cache.

No MLIP and no GPU: the relaxer is a stub, so these run in milliseconds.
"""
import os

import numpy as np
import pytest
from ase import Atoms

from demars_core import _checkpoint as CK


def _conf(n=3, shift=0.0):
    return Atoms('Si%d' % n, positions=[[i + shift, 0, 0] for i in range(n)],
                 cell=[12.0] * 3, pbc=True)


class StubRelax:
    """Deterministic 'relaxation': moves atom 0 and returns an energy fixed by the geometry.

    `die_after` makes it raise once it has relaxed that many configs -- the kill we are defending
    against. `seen` records every config it was actually asked to relax, which is how the tests
    tell reuse from recomputation.
    """

    def __init__(self, die_after=None, fail_indices=()):
        self.die_after = die_after
        self.fail_indices = set(fail_indices)
        self.seen = []

    def __call__(self, confs):
        E, rel, val = [], [], []
        for a in confs:
            if self.die_after is not None and len(self.seen) >= self.die_after:
                raise RuntimeError('killed mid-round')
            n = len(self.seen)
            self.seen.append(a.copy())
            r = a.copy()
            r.positions[0][1] += 0.25                    # a geometry only the relaxer produces
            if n in self.fail_indices:
                E.append(float('nan')); rel.append(a.copy()); val.append(False)
            else:
                E.append(-1.0 * len(a) - 0.001 * float(a.positions.sum()))
                rel.append(r); val.append(True)
        return np.asarray(E, dtype=float), rel, val


def test_a_kill_mid_round_keeps_the_configs_that_finished(tmp_path):
    """The regression: before this, a kill at config 3 of 4 left nothing on disk at all."""
    p = str(tmp_path / 'ck.xyz')
    confs = [_conf(shift=s) for s in (0.0, 0.1, 0.2, 0.3)]
    with pytest.raises(RuntimeError):
        CK.checkpointed(StubRelax(die_after=2), p, chunk=1)(confs)

    assert os.path.exists(p), 'the round died and took every finished config with it'
    assert len(CK.read_checkpoint(p)) == 2


def test_resume_relaxes_only_what_is_missing(tmp_path):
    p = str(tmp_path / 'ck.xyz')
    confs = [_conf(shift=s) for s in (0.0, 0.1, 0.2, 0.3)]
    with pytest.raises(RuntimeError):
        CK.checkpointed(StubRelax(die_after=2), p, chunk=1)(confs)

    second = StubRelax()
    E, rel, val = CK.checkpointed(second, p, chunk=1)(confs)

    assert len(second.seen) == 2, f'resume recomputed work already on disk ({len(second.seen)}/4)'
    assert all(val) and len(E) == 4


def test_what_comes_back_from_disk_is_the_relaxed_result_not_the_input(tmp_path):
    """A resume that silently returned the UNRELAXED geometry would still 'pass' a count check."""
    p = str(tmp_path / 'ck.xyz')
    confs = [_conf(shift=s) for s in (0.0, 0.1)]
    ref_E, ref_rel, _ = CK.checkpointed(StubRelax(), p, chunk=1)(confs)

    got_E, got_rel, got_val = CK.checkpointed(StubRelax(die_after=0), p, chunk=1)(confs)

    assert all(got_val)
    assert np.allclose(got_E, ref_E, atol=1e-6)
    for a, b in zip(got_rel, ref_rel):
        assert np.allclose(a.get_positions(), b.get_positions(), atol=1e-6)
        assert np.allclose(a.get_cell(), b.get_cell(), atol=1e-6)
        assert a.get_chemical_symbols() == b.get_chemical_symbols()


def test_a_torn_final_frame_costs_that_frame_and_nothing_else(tmp_path):
    """The file is appended to while the process can be killed at any instant."""
    p = str(tmp_path / 'ck.xyz')
    confs = [_conf(shift=s) for s in (0.0, 0.1, 0.2)]
    CK.checkpointed(StubRelax(), p, chunk=1)(confs)

    # A kill lands mid-write, so the danger is not a MISSING line -- it is a last line that still
    # parses as coordinates while being short of what was written. Cut inside it, newline and all.
    text = open(p).read()
    with open(p, 'w') as fh:
        fh.write(text[:-3])

    assert len(CK.read_checkpoint(p)) == 2, 'a torn tail cost more than the frame in flight'
    again = StubRelax()
    _E, _rel, val = CK.checkpointed(again, p, chunk=1)(confs)
    assert len(again.seen) == 1 and all(val)


def test_a_failed_relaxation_is_never_cached(tmp_path):
    """val=False can be the environment (OOM, no CUDA kernels), not the structure. Caching it
    would freeze a transient fault into a permanent 'this config failed' -- D10's laundering."""
    p = str(tmp_path / 'ck.xyz')
    confs = [_conf(shift=s) for s in (0.0, 0.1)]
    E, _rel, val = CK.checkpointed(StubRelax(fail_indices=[1]), p, chunk=1)(confs)
    assert val == [True, False] and np.isnan(E[1])
    assert len(CK.read_checkpoint(p)) == 1

    retry = StubRelax()
    _E2, _rel2, val2 = CK.checkpointed(retry, p, chunk=1)(confs)
    assert len(retry.seen) == 1 and all(val2), 'the failure was cached instead of being retried'


def test_a_different_potential_is_a_different_key(tmp_path):
    """Same geometry, different weights -> different energy. Reuse across tags would be silent."""
    p = str(tmp_path / 'ck.xyz')
    confs = [_conf(shift=0.0)]
    CK.checkpointed(StubRelax(), p, tag='7net-nano|d3=False')(confs)

    other = StubRelax()
    CK.checkpointed(other, p, tag='7net-omni|d3=False')(confs)
    assert len(other.seen) == 1, 'an energy from one potential was reused for another'


def test_a_changed_decoration_misses_the_cache(tmp_path):
    """The key is the config itself, so a changed seed/cut/supercell needs no version field."""
    p = str(tmp_path / 'ck.xyz')
    CK.checkpointed(StubRelax(), p)([_conf(shift=0.0)])

    moved = StubRelax()
    CK.checkpointed(moved, p)([_conf(shift=0.5)])
    assert len(moved.seen) == 1

    same = StubRelax()
    CK.checkpointed(same, p)([_conf(shift=0.0)])
    assert len(same.seen) == 0, 'an identical decoration failed to match its own cache entry'


def test_the_potential_tag_is_stable_across_processes():
    """Caught by running the real wiring, not by the unit tests above: the first version built the
    tag from `str(calculator)`, which for an ASE Calculator instance contains a memory address --
    so every run named a different potential, every key missed, and the resume silently never
    happened. A cache miss looks exactly like work still to do, which is why this needs a test."""
    class Stub:
        pass

    t = CK.relaxer_tag('ase-generic', Stub(), False)
    assert t == CK.relaxer_tag('ase-generic', Stub(), False), 'the tag is not reproducible'
    assert '0x' not in t and 'object at' not in t, t
    assert CK.relaxer_tag('sevennet', '7net-nano', False) != \
        CK.relaxer_tag('sevennet', '7net-omni', False), 'two potentials share a tag'
    assert CK.relaxer_tag('sevennet', '7net-nano', False) != \
        CK.relaxer_tag('sevennet', '7net-nano', True), 'D3 changes the energies but not the tag'


def test_the_key_ignores_signed_zero(tmp_path):
    """-0.0 == 0.0 but has different bits, and the key hashes bytes."""
    a, b = _conf(), _conf()
    a.positions[0][1] = 0.0
    b.positions[0][1] = -0.0
    assert CK.config_key(a) == CK.config_key(b)


# --------------------------------------------------------------------------------------------
# through the real engine -- the wrapper has to survive `build()`, not just a list of configs
# --------------------------------------------------------------------------------------------

FIXDIR = os.path.join(os.path.dirname(__file__), 'fixtures')


def _build(relax, **kw):
    from demars_core._engine import mar_engine as ME
    from demars_core._engine import mar_evidence as MEV
    txt = open(os.path.join(FIXDIR, 'cod_1544358.cif')).read()
    ev = MEV.evidence_from_text(txt, iid=None, search_siblings=False)
    s = ME.structure_from_text(txt)
    return ME.build(structure=s, evidence=ev, relax=relax, NR=6, MIN_A=8.0, seed=0, **kw)


def test_an_interrupted_engine_round_resumes_where_it_stopped(tmp_path):
    """End-to-end: the same decorations must come back out of the cache, with the same energies
    as an uninterrupted run. If `build()` re-decorated differently, the keys would all miss."""
    ref = StubRelax()
    _ev, ref_built, _info = _build(ref)
    assert ref_built is not None, 'the fixture stopped building -- this test measures nothing'
    ref_E = [s.get('E_per_atom') for s in ref_built[3]]

    p = str(tmp_path / 'ck.xyz')
    with pytest.raises(RuntimeError):
        _build(CK.checkpointed(StubRelax(die_after=3), p, chunk=1))
    assert len(CK.read_checkpoint(p)) == 3

    rest = StubRelax()
    _ev2, built2, _info2 = _build(CK.checkpointed(rest, p, chunk=1))
    assert built2 is not None
    assert len(rest.seen) == len(ref.seen) - 3, 'resume recomputed configs already on disk'
    assert [s.get('E_per_atom') for s in built2[3]] == ref_E
