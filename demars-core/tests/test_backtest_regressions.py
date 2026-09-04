"""Regression tests for the defects the 2026-08-07 set-A backtest exposed.

Covered here: (1)(2)(3) in the engine, plus the provenance convention. Each defect carries its
own narrative in the `# ---- D<n>:` section header that introduces its tests -- this file is the
record, not a pointer to one.

They share one property: **they produce a plausible wrong answer silently.** None of them
raised, none of them tripped a gate, and most were caught only because a human/reviewer
looked at the structure. That is exactly the class of bug a test suite has to pin down, because at
1,000-entry scale nobody re-reads the output.

No MLIP and no GPU: the relaxer is injected, so `build()` runs end-to-end on CPU in milliseconds.
"""
import json
import os
import sys
import types

import numpy as np
import pytest
from ase import Atoms

from demars_core._engine import mar_engine as ME
from demars_core._engine import mar_evidence as MEV


FIXDIR = os.path.join(os.path.dirname(__file__), 'fixtures')

# Half-occupied Fe orbit -- the same geometry as the backtest entry that found defect ①.
HALF_OCC_FIXTURE = 'cod_1544358.cif'
# Wide enough that some decorations cannot reach the target as an independent set, narrow enough
# that most still can. A cut where EVERY sample fails is a different test (see the error case).
MIXED_CUT = 2.2


def _constant_total_energy(confs):
    """Relaxer stub returning the SAME TOTAL energy for every config.

    This is the pathology in its purest form: E/atom = E_total/n_atoms, so with the total held
    fixed the config with the FEWEST atoms has the lowest per-atom energy and wins take-lowest.
    A cation-deficient decoration is therefore not merely tolerated by the old ranking -- it is
    actively preferred.
    """
    return [-1000.0] * len(confs), list(confs), [True] * len(confs)


def _build(fixture, **kw):
    txt = open(os.path.join(FIXDIR, fixture)).read()
    ev = MEV.evidence_from_text(txt, iid=None, search_siblings=False)
    s = ME.structure_from_text(txt)
    return ME.build(structure=s, evidence=ev, relax=_constant_total_energy,
                    NR=12, MIN_A=10.0, seed=0, **kw)


# --------------------------------------------------------------------------------------------
# ① short-filled decorations must never be ranked
# --------------------------------------------------------------------------------------------

def test_short_filled_decorations_are_reported_and_invalidated():
    """The independent-set fill is a single greedy pass and can miss the occupancy target.

    It must SAY so (`short_fill`) and the sample must be invalid. Before the fix the shortfall was
    invisible and the sample went on to compete for take-lowest.
    """
    _ev, built, _info = _build(HALF_OCC_FIXTURE, same_excl=MIXED_CUT)
    assert built is not None, 'fixture/cut no longer produces a usable mix -- retune MIXED_CUT'
    _confs, _rel, _val, samples, _i = built

    short = [s for s in samples if s.get('short_fill')]
    assert short, 'no decoration fell short -- this fixture/cut no longer exercises the defect'
    assert len(short) < len(samples), 'every decoration fell short -- see the error-path test'

    for s in short:
        assert s['valid'] is False
        assert 'short-filled' in s['invalid_reason']
        assert s['E_per_atom'] is None
    for s in samples:
        if s.get('valid'):
            assert not s.get('short_fill')


def test_short_filled_configs_really_are_a_different_composition():
    """Pins the REASON the shortfall matters, not just that it is flagged.

    A short-filled config has genuinely fewer atoms, so per-atom energy is not comparable to a
    complete decoration's. If this assertion ever fails, `short_fill` has stopped tracking the
    thing that makes the ranking unsound and the flag has become decorative.
    """
    _ev, built, _info = _build(HALF_OCC_FIXTURE, same_excl=MIXED_CUT)
    confs, _rel, _val, samples, _i = built

    n_short = [len(confs[i]) for i, s in enumerate(samples) if s.get('short_fill')]
    n_full = [len(confs[i]) for i, s in enumerate(samples) if not s.get('short_fill')]
    assert n_short and n_full
    assert max(n_short) < min(n_full), 'short-filled configs are supposed to be atom-deficient'


def test_valid_samples_all_share_one_composition():
    """Take-lowest ranks by E/atom, which is only meaningful within a single composition."""
    _ev, built, _info = _build(HALF_OCC_FIXTURE, same_excl=MIXED_CUT)
    _confs, _rel, _val, samples, _i = built
    comps = {tuple(sorted(s['composition'].items())) for s in samples if s.get('valid')}
    assert len(comps) == 1, f'per-atom energies would be compared across {len(comps)} compositions'


def test_no_valid_decoration_reports_an_error_instead_of_crashing():
    """An exclusion cut too wide to satisfy must fail loudly.

    Invalidating short-filled samples introduced a reachable empty-`ok` case; downstream does
    `min(ok, ...)`, which would raise a bare ValueError with no indication of the cause.
    """
    _ev, built, info = _build(HALF_OCC_FIXTURE, same_excl=3.0)
    assert built is None
    assert info['error'].startswith('no valid decoration')
    assert 'short-filled' in info['error']
    assert info['short_fill_per_sample']


def test_unconstrained_build_is_unaffected():
    """Guard against the fix firing on ordinary structures: with no exclusion, nothing is short."""
    _ev, built, _info = _build(HALF_OCC_FIXTURE)
    _confs, _rel, _val, samples, _i = built
    assert all(not s.get('short_fill') for s in samples)
    assert all(s.get('valid') for s in samples)


# --------------------------------------------------------------------------------------------
# ② a handful of accidental contacts must not define a "unit"
# --------------------------------------------------------------------------------------------

def _scattered(el, n, spacing, cell):
    """n atoms of `el` on a cubic grid at `spacing` -- far enough apart to share no bonds."""
    pts = [(x * spacing, y * spacing, z * spacing)
           for x in range(10) for y in range(10) for z in range(10)][:n]
    return Atoms(f'{el}{n}', positions=pts, cell=cell, pbc=True)


def test_accidental_same_element_contacts_do_not_create_a_unit():
    """Ga2Te3, the entry that found defect ②.

    Te is in BOTH `COORD_FORMERS` and `COORD_ANIONS`, so a Te-Te contact reads as a former-ligand
    bond. With the modal CN taken over bonded centres only, three chance contacts among ~100 Te set
    the target CN to 1 and every unbonded Te was then reported as a stripped centre.
    """
    at = _scattered('Te', 60, 6.0, [60.0, 60.0, 60.0])
    # two close pairs -- inside the covalent-radius cutoff, but 4 atoms out of 60
    at.positions[0] = [30.0, 30.0, 30.0]
    at.positions[1] = [32.7, 30.0, 30.0]
    at.positions[2] = [30.0, 40.0, 30.0]
    at.positions[3] = [32.7, 40.0, 30.0]

    bonds, modal, _med = ME.former_coordination(at)
    assert 'Te' not in modal, 'a 4-in-60 contact rate is not a structural motif'
    assert not bonds, 'the rejected element must not leave bonds behind for the other consumers'


def test_prevalent_units_are_still_detected():
    """Positive control for the prevalence gate -- it must not suppress real units.

    If the threshold is ever raised carelessly this fails before the gate goes silent in production.
    """
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    v /= np.linalg.norm(v[0])
    pos, sym = [], []
    for k in range(8):                       # 8 isolated SO4 tetrahedra
        c = np.array([6.0 * (k % 4) + 5.0, 6.0 * (k // 4) + 5.0, 5.0])
        pos.append(c); sym.append('S')
        for d in v:
            pos.append(c + 1.47 * d); sym.append('O')
    at = Atoms(sym, positions=pos, cell=[40.0, 40.0, 40.0], pbc=True)

    _bonds, modal, _med = ME.former_coordination(at)
    assert modal.get('S') == 4


# --------------------------------------------------------------------------------------------
# the provenance convention -- defects (4) and (5) are two of its states
# --------------------------------------------------------------------------------------------

def _prov_build(fixture, **kw):
    txt = open(os.path.join(FIXDIR, fixture)).read()
    ev = MEV.evidence_from_text(txt, iid=None, search_siblings=False)
    _e, _b, info = ME.build(structure=ME.structure_from_text(txt), evidence=ev,
                            relax=_constant_total_energy, NR=2, MIN_A=8.0, seed=0, **kw)
    return info


PROV_FIXTURES = ['cod_1544358.cif', 'cod_1010497.cif', 'cod_9003141.cif']


def test_provenance_state_stays_a_closed_set():
    """A free-text state is the bug this convention replaces -- keep it unable to come back."""
    for fx in PROV_FIXTURES:
        info = _prov_build(fx)
        for name, blk in info['provenance_summary']['states'].items():
            assert blk in MEV.PROV_STATES, f'{fx}:{name} = {blk!r}'
    with pytest.raises(ValueError):
        MEV.provenance('probably_fine', 'not a member of the closed set')


def test_confident_is_derived_from_the_state_not_set_by_hand():
    for state in MEV.PROV_STATES:
        assert MEV.provenance(state, 'x')['confident'] == (state in MEV.PROV_CONFIDENT)
    assert set(MEV.PROV_CONFIDENT) == {'derived', 'overridden', 'vacuous'}


def test_vacuous_is_separated_from_unmatched():
    """The load-bearing distinction. Collapse these two and `couple_cut` reports unconfident on
    4 of 4 set-A entries, which is indistinguishable from no signal at all.

    `vacuous`   -- no cross-orbit contact exists; the default is right by construction.
    `unmatched` -- contacts exist but matched no former/anion rule; only chemistry can say whether
                   that is correct (Li is not a former) or a table gap (Co in a Co-As intermetallic).
    """
    vac = _prov_build('cod_1010497.cif')['couple_cut']['provenance']
    unm = _prov_build('cod_1544358.cif')['couple_cut']['provenance']
    assert vac['state'] == 'vacuous' and vac['confident'] is True
    assert vac['evidence']['n_contacts'] == 0
    assert unm['state'] == 'unmatched' and unm['confident'] is False
    assert unm['evidence']['n_contacts'] > 0 and unm['evidence']['n_former_anion'] == 0
    assert unm['evidence']['elements'], 'the analyst needs the elements to judge the non-match'


def test_out_of_window_is_reported_where_defect_4_bites():
    """Pairs exist but sit above the 1.3 A window the derivation inspects, so no cut is set.

    Before this, that case was worded exactly like "nothing within 3 A" and read as a non-event --
    which is how one entry's face-sharing Mg-Mg at 2.87 A needed a hand-passed --same-excl.
    """
    prov = _prov_build('cod_1010497.cif')['exclusion_merge']['provenance']
    oow = [p for p in prov.values() if p['state'] == 'out_of_window']
    assert oow, 'this fixture no longer exercises defect (4)'
    for p in oow:
        assert p['confident'] is False
        assert p['evidence']['closest_A'] is not None       # data EXISTS -- that is the whole point
        assert p['evidence']['closest_A'] >= p['evidence']['window_A']


def test_an_override_is_never_reported_as_derived():
    prov = _prov_build('cod_1544358.cif', same_excl=2.2)['exclusion_merge']['provenance']
    assert prov, 'no per-element provenance emitted'
    for el, p in prov.items():
        assert p['state'] == 'overridden', f'{el} = {p["state"]}'
        assert p['confident'] is True and p['source'] == '--same-excl'


def test_summary_agrees_with_the_individual_blocks():
    """The rollup is what stage ② will gate on, so it must not drift from the fields it summarises."""
    for fx in PROV_FIXTURES:
        info = _prov_build(fx)
        expect = {f'exclusion_merge.{el}': p
                  for el, p in info['exclusion_merge']['provenance'].items()}
        expect['couple_cut'] = info['couple_cut']['provenance']
        expect['supercell'] = info['supercell_provenance']
        expect['charge_balancing'] = info['charge_balancing']['provenance']
        expect['ordered_sibling'] = MEV.provenance('not_run', 'search disabled in this fixture build')

        s = info['provenance_summary']
        assert s['states'] == {k: p['state'] for k, p in expect.items()}
        assert s['unconfident'] == sorted(k for k, p in expect.items() if not p['confident'])
        assert s['n_unconfident'] == len(s['unconfident'])


def test_basis_survives_so_downstream_does_not_break():
    """`basis` is still parsed in 7 places; the convention is additive, not a replacement."""
    for fx in PROV_FIXTURES:
        info = _prov_build(fx)
        assert info['couple_cut']['basis']
        em = info['exclusion_merge']
        assert set(em['basis']) == set(em['provenance']), 'basis and provenance cover the same elements'
        for el, p in em['provenance'].items():
            assert p['detail'] == em['basis'][el]


def test_sibling_reports_not_run_rather_than_absent():
    """AGENTS.md: "no sibling DB configured" means UNCHECKED and must never read as "none exists"."""
    txt = open(os.path.join(FIXDIR, 'cod_1544358.cif')).read()
    ev = MEV.evidence_from_text(txt, iid=None, search_siblings=False)
    p = ev['ordered_sibling_provenance']
    assert p['state'] == 'not_run' and p['confident'] is False


@pytest.fixture(scope='module')
def corpus():
    """Every shipped fixture, built into an ordered config. No MLIP -- the relaxer is injected."""
    import glob
    out = []
    for path in sorted(glob.glob(os.path.join(FIXDIR, 'cod_*.cif'))):
        txt = open(path).read()
        ev = MEV.evidence_from_text(txt, iid=None, search_siblings=False)
        _e, built, _i = ME.build(structure=ME.structure_from_text(txt), evidence=ev,
                                 relax=_constant_total_energy, NR=2, MIN_A=8.0, seed=0)
        if built:
            out.append((os.path.basename(path), built[0][0]))
    assert len(out) >= 5, 'the fixture corpus stopped building -- this test measures nothing'
    return out


def test_the_prevalence_threshold_sits_in_an_empty_gap(corpus):
    """MIN_UNIT_PREVALENCE must not be a value tuned to the four backtest entries.

    The defence is that the observed distribution is BIMODAL: an element either centres a unit
    essentially always, or essentially never. Nothing real lands near the threshold, so any cutoff
    in a wide band gives identical answers. Pin that by varying the threshold across an order of
    magnitude and demanding the SAME verdict every time -- if a future fixture (or a change to the
    covalent-radius cutoff) puts an element in the ambiguous middle, this fails and the number has
    to be re-argued rather than silently becoming load-bearing.

    Measured 2026-08-11 over this corpus + the four validation representatives: every prevalence was
    0.000, 0.037 or 1.000. The 0.037 point (Ga2Te3 Te, 4/108) comes from a licensed ICSD entry that
    cannot ship as a fixture, so only the upper mode is reproducible here; `_scattered` above covers
    the lower one synthetically.
    """
    for name, at in corpus:
        verdicts = {thr: ME.former_coordination(at, min_prevalence=thr)[1]
                    for thr in (0.05, 0.2, 0.5)}
        assert len(set(map(str, verdicts.values()))) == 1, (
            f'{name}: the unit verdict depends on the threshold -- {verdicts}. '
            'An element now sits in the ambiguous band, so 0.2 is doing real work and needs a '
            'stated justification -- an unmeasured cutoff is a cutoff tuned to one corpus.')


def test_no_fixture_element_lands_near_the_threshold(corpus):
    """The same invariant stated directly, so a failure names the offending element."""
    for name, at in corpus:
        syms = at.get_chemical_symbols()
        bonds, _m, _p = ME.former_coordination(at, min_prevalence=0.0)   # gate off -> raw rates
        for el in sorted(set(syms) & set(ME.COORD_FORMERS)):
            n = syms.count(el)
            prev = sum(1 for i in bonds if syms[i] == el) / n
            assert prev <= 0.05 or prev >= 0.5, f'{name} {el}: prevalence {prev:.3f} is ambiguous'


def test_prevalence_threshold_is_overridable():
    """The trade-off is documented as tunable, so keep the knob wired."""
    at = _scattered('Te', 60, 6.0, [60.0, 60.0, 60.0])
    at.positions[0] = [30.0, 30.0, 30.0]
    at.positions[1] = [32.7, 30.0, 30.0]
    _b, modal_default, _m = ME.former_coordination(at)
    _b, modal_loose, _m = ME.former_coordination(at, min_prevalence=0.0)
    assert 'Te' not in modal_default
    assert 'Te' in modal_loose


# --------------------------------------------------------------------------------------------
# ③ the sibling search must record the key it searched on
# --------------------------------------------------------------------------------------------

def test_sibling_key_field_always_present():
    """`None` when the search did not run -- so a missing key is never confused with an empty result."""
    txt = open(os.path.join(FIXDIR, HALF_OCC_FIXTURE)).read()
    ev = MEV.evidence_from_text(txt, iid=None, search_siblings=False)
    assert 'ordered_sibling_key' in ev
    assert ev['ordered_sibling_key'] is None
    assert ev['ordered_sibling'] == 'not searched'


@pytest.mark.skipif(MEV._sib is None, reason='no local ICSD sibling DB configured')
def test_sibling_key_follows_the_cif_formula():
    """A model repair that rewrites `_chemical_formula_sum` changes the search key.

    That is legitimate behaviour -- but the empty result it can produce reads as
    "checked and absent". Recording the key is what makes the two distinguishable afterwards.
    Ga2Te3 repaired to Ga2.667Te4 searched for Ga3Te4 and found nothing.
    """
    txt = open(os.path.join(FIXDIR, HALF_OCC_FIXTURE)).read()
    ev = MEV.evidence_from_text(txt, iid=None, search_siblings=True)
    key = ev['ordered_sibling_key']
    assert key is not None
    assert key['formula_sum'] == ev['provenance']['formula_sum']
    assert key['nominal_reduced_formula']
    assert key['chemsys']


# --------------------------------------------------------------------------------------------
# D13 the atom budget shrinks the cell silently, and the record keeps quoting the REQUEST
# --------------------------------------------------------------------------------------------

def _sc_build(min_a, fixture=HALF_OCC_FIXTURE):
    txt = open(os.path.join(FIXDIR, fixture)).read()
    ev = MEV.evidence_from_text(txt, iid=None, search_siblings=False)
    s = ME.structure_from_text(txt)
    return ME.build(structure=s, evidence=ev, relax=_constant_total_energy,
                    NR=3, MIN_A=min_a, seed=0)


def test_a_cell_the_atom_budget_cut_no_longer_reads_as_the_requested_cell():
    """The regression: `mar_engine.py:719-723` walks the multiplier down when the template exceeds
    MAXAT*1.6, and `min_cell_A` went on reporting the REQUEST. One set-B entry shipped b = 14.216 A
    under `min_cell_A: 15.0` and the record read as a contract kept -- the reviewer caught it only
    by measuring the shipped lattice vectors by hand.
    """
    _ev, built, info = _sc_build(20.0)
    assert built is not None

    smp = info['sampling']
    assert smp['min_cell_A'] == 20.0, 'the request must still be recorded verbatim'
    assert smp['min_cell_A_achieved'] < 20.0, 'this fixture no longer exercises the back-off'

    p = info['supercell_provenance']
    assert p['state'] == 'overridden' and p['confident'] is True
    assert 'budget' in (p['source'] or ''), p
    assert p['evidence']['supercell_requested'] != p['evidence']['supercell'], p['evidence']
    assert p['evidence']['n_atoms_template'] <= ME.MAXAT * 1.6


def test_a_cell_that_met_the_request_is_plain_derived():
    """The guard: the stamp must mean something, so it may not appear on cells that kept the
    contract. A cut multiplier that still clears MIN_A is not a violation either."""
    _ev, built, info = _sc_build(10.0)
    assert built is not None

    assert info['sampling']['min_cell_A_achieved'] >= 10.0
    p = info['supercell_provenance']
    assert p['state'] == 'derived' and p['source'] is None


def test_the_achieved_value_is_measured_on_the_cell_that_was_actually_built():
    """`min_cell_A_achieved` has to come from the shipped geometry, not from the multiplier it was
    supposed to produce -- re-deriving it from the request is how the field would go stale."""
    _ev, built, info = _sc_build(20.0)
    conf = built[0][0]                                   # a decorated config, the thing that ships
    shortest = float(np.linalg.norm(np.array(conf.get_cell()), axis=1).min())
    assert info['sampling']['min_cell_A_achieved'] == pytest.approx(shortest, abs=0.01)


def test_the_couple_cut_basis_does_not_claim_there_were_no_contacts():
    """D5: `basis` is free text, but it is what the run summary prints and what a reader believes.
    It said "no contact in evidence" whenever the FILTERED list was empty -- false for 3 of the 4
    set-A entries, which had contacts that simply matched no former/anion rule. It reads as
    "nothing to see", and the self-driving loop duly stopped on an 82.8 meV/at spread.
    """
    vac = _prov_build('cod_1010497.cif')['couple_cut']
    unm = _prov_build('cod_1544358.cif')['couple_cut']
    assert vac['provenance']['state'] == 'vacuous', 'fixture no longer exercises the no-contact case'
    assert unm['provenance']['state'] == 'unmatched', 'fixture no longer exercises the defect'

    assert 'no cross-orbit contact in evidence' in vac['basis']
    assert 'no cross-orbit contact in evidence' not in unm['basis'], unm['basis']
    assert 'none matched' in unm['basis'] and 'COORD_FORMERS' in unm['basis'], unm['basis']
    assert vac['basis'] != unm['basis'], 'the two situations still share one sentence'


def test_every_state_gets_its_own_basis_sentence():
    """The rule the fix encodes: `basis` and `provenance.state` must never disagree. A reader who
    only sees the printed summary has to reach the same conclusion as one who parses the record."""
    seen = {}
    for fx in PROV_FIXTURES:
        cc = _prov_build(fx)['couple_cut']
        seen.setdefault(cc['provenance']['state'], set()).add(cc['basis'])
    for state, sentences in seen.items():
        assert len(sentences) == 1 or state in ('derived', 'overridden'), \
            f'{state} produced {len(sentences)} different basis strings'
    bases = {s for ss in seen.values() for s in ss}
    assert len(bases) == len(seen), f'two states share a basis sentence: {seen}'


# --------------------------------------------------------------------------------------------
# D12 an anion whose centre is missing from COORD_FORMERS is a table gap, not a defect
# --------------------------------------------------------------------------------------------

def _ge_sn_se():
    """Three Se in one cell: one on a LISTED former (Ge), one on an UNLISTED one (Sn), one alone.

    That is the Li4(Ge,Sn)Se4 situation in miniature -- the former site is shared by an element the
    table knows and one it does not, and only the third Se is a real uncoordinated anion.
    """
    return Atoms('GeSeSnSeSe',
                 positions=[[0, 0, 0], [2.30, 0, 0],          # Ge-Se: a listed former
                            [10, 0, 0], [12.55, 0, 0],        # Sn-Se: unlisted, same job
                            [20, 10, 10]],                    # Se with nothing in reach
                 cell=[30.0] * 3, pbc=True)


def test_an_anion_on_an_unlisted_former_is_not_counted_as_a_removable_defect():
    """The regression: `dangling_anions` counted every anion whose centre was absent from
    COORD_FORMERS. On the entry that found this it read 100/192 and the self-driving loop spent all
    three rounds trying to re-decorate away a defect that did not exist -- Ge and Sn share the
    former site, and `--expect` cannot add an element, so there was no way out inside the tool."""
    d = ME.config_descriptors(_ge_sn_se())

    assert d['dangling_anions'] == 2, d          # raw count unchanged: still comparable
    assert d['dangling_unexplained'] == 1, 'the Sn-bonded Se is still counted as a real defect'
    expl = d['dangling_explained_by_unlisted_former']
    assert set(expl) == {'Sn'} and expl['Sn']['n_anions'] == 1, expl
    assert expl['Sn']['median_d_A'] == pytest.approx(2.55, abs=0.05)


def test_a_genuinely_uncoordinated_anion_stays_unexplained():
    """The guard. If the fix excused every dangling anion it would hide the failure mode the
    descriptor exists to catch, and the self-driving loop would stop on a broken structure."""
    at = _ge_sn_se()
    del at[3]                                    # drop the Sn-bonded Se; only the lone one is left
    d = ME.config_descriptors(at)

    assert d['dangling_anions'] == 1 and d['dangling_unexplained'] == 1
    assert not d['dangling_explained_by_unlisted_former']


def test_the_split_is_silent_on_structures_with_nothing_to_dispute():
    """Measured before it was written: an earlier version keyed on "elements that BEHAVE like
    formers" (prevalence + uniform CN) and fired on La, Mg, Fe, Ti, Zr, Ca, Cu, Ba ... in ordinary
    ordered crystals. A warning that fires on nearly everything is not a warning."""
    import glob

    fired = []
    for path in sorted(glob.glob(os.path.join(FIXDIR, 'cod_*.cif'))):
        from ase.io import read as aseread
        d = ME.config_descriptors(aseread(path))
        if d['dangling_explained_by_unlisted_former']:
            fired.append(os.path.basename(path))
    assert not fired, f'the ask fires where no defect is being claimed: {fired}'


def test_self_driving_chases_only_the_dangling_it_could_remove():
    """The trigger has to read the SPLIT count, or the fix stops at reporting and the loop still
    burns its three rounds on an unreachable target -- D6's disease, one field over."""
    import inspect

    src = inspect.getsource(ME.run_self_driving)
    assert 'dangling_unexplained' in src, 'the loop is still triggering on the raw dangling count'
    assert "lod.get('dangling_anions', 0)" in src, 'no fallback for descriptors from older runs'


# --------------------------------------------------------------------------------------------
# D6 the self-driving loop must not chase a target it cannot reach
# --------------------------------------------------------------------------------------------

class _FakeDescriptors:
    """`config_descriptors` replaced by a scripted sequence, so a test can set the correlation
    between a defect count and energy without needing an MLIP or a pathological structure.

    `f(call_index)` -> the count. Calls arrive in sample order and the stub relaxer makes energy
    fall with that same index, so `f` alone fixes the sign of the descriptor-energy correlation.
    """

    KEYS = ('dangling_anions', 'over_coord_formers', 'bridging_anions', 'clashes',
            'stretched_bonds', 'dangling_unexplained')

    def __init__(self, f):
        self.f, self.i = f, 0

    def __call__(self, atoms, **kw):
        v = self.f(self.i)
        self.i += 1
        return dict({k: 0 for k in self.KEYS}, under_coord_formers=v,
                    dangling_explained_by_unlisted_former={})


def _drive(monkeypatch, f, n=8):
    """Run the loop on a real fixture with scripted descriptors, energies falling with the index."""
    monkeypatch.setattr(ME, 'config_descriptors', _FakeDescriptors(f))
    txt = open(os.path.join(FIXDIR, HALF_OCC_FIXTURE)).read()
    ev = MEV.evidence_from_text(txt, iid=None, search_siblings=False)
    s = ME.structure_from_text(txt)

    def relax(confs):                       # spread must clear BROAD_SPREAD_MEV to reach the loop
        return ([-1000.0 - 400.0 * i for i in range(len(confs))], list(confs), [True] * len(confs))

    _e, _built, _info, trace = ME.run_self_driving(
        structure=s, evidence=ev, relax=relax, NR=n, MIN_A=8.0, max_rounds=3)
    return trace


def test_a_defect_count_that_falls_as_energy_rises_is_not_chased(monkeypatch):
    """The regression: nothing correlated with energy, but a representative carrying "defects"
    still forced a round. On the entry that found this the descriptor had **rho = -0.682** -- FEWER
    of them went with HIGHER energy -- so removing them was the wrong direction, and the loop spent
    its whole budget going that way. (There it was stoichiometric: 1/3 cation vacancies make the
    mean anion coordination 8/3, so a CN-4 target is unreachable by any decoration.)"""
    trace = _drive(monkeypatch, lambda i: i)          # defects RISE as energy falls

    assert len(trace) == 1, f'the loop kept going: {[e["action"] for e in trace]}'
    act = trace[0]['action']
    assert act.startswith('stop:') and 'NEGATIVELY' in act, act
    assert trace[0]['rep_coupling_defects'] > 0, 'this setup no longer triggers the fallback path'


def test_a_second_identical_round_is_not_spent_when_the_first_changed_nothing(monkeypatch):
    """Uncorrelated, unmoving defects: the constructive round deserves ONE attempt, not three.
    Without this the budget is spent re-running an enforcement that demonstrably did not work."""
    trace = _drive(monkeypatch, lambda i: 3)          # constant: no correlation, no improvement

    assert len(trace) == 2, f'expected enforce then stop, got {[e["action"] for e in trace]}'
    assert trace[0]['action'].startswith('enforce:')
    assert trace[0]['enforced'] == 'under_coord_formers'
    assert trace[1]['action'].startswith('stop:') and 'did not improve' in trace[1]['action']


def test_a_defect_that_genuinely_tracks_energy_is_still_chased(monkeypatch):
    """The guard: it may only stop the loop where chasing is pointless. A descriptor that rises
    with energy is exactly what the self-driving loop is for, and must still be enforced."""
    trace = _drive(monkeypatch, lambda i: 30 - i)     # defects FALL as energy falls

    assert trace[0]['action'].startswith('enforce:'), trace[0]['action']
    assert trace[0]['diagnosis']['top']['descriptor'] == 'under_coord_formers'
    assert trace[0]['diagnosis']['top']['rho'] > 0


def test_a_non_numeric_descriptor_does_not_take_out_the_whole_diagnosis():
    """Found by the D6 tests, introduced by D12: `config_descriptors` gained a dict-valued
    explanatory payload and `diagnose_spectrum` float()s every descriptor it sees. That raises
    inside the ranking loop, so the self-driving diagnosis dies on **every** structure that reaches
    it -- and nothing reached it in the suite, because the other tests stop before the spread test.
    """
    descs = [{'under_coord_formers': 1, 'note': {'Sn': 4}},      # fewest defects at the LOWEST energy
             {'under_coord_formers': 3, 'note': {}},
             {'under_coord_formers': 2, 'note': {'Sn': 1}}]

    out = ME.diagnose_spectrum([-3.0, -1.0, -2.0], descs)

    assert [k for k, *_ in out['ranked']] == ['under_coord_formers'], out['ranked']
    assert out['top'] is not None and out['top']['rho'] > 0


# --------------------------------------------------------------------------------------------
# D10 an autobatcher budget below one system's own metric was reported as a relaxation failure
# --------------------------------------------------------------------------------------------
# The measured path: the memory probe climbs past the real capacity, comes back, and
# `raw * 0.8` lands below a single system's metric. torch-sim then refuses the batch with a plain
# `ValueError`, which is neither an OOM (so no retry) nor an infra marker (so the whole batch is
# laundered into val=False) -> `final_MAR: {}`. The same structure relaxes cleanly with
# `autobatcher=False`. So a *configuration* fault was being recorded as a *chemical* one.

_TS_REFUSAL = ('Max metric of system with index 0 in states: 15232.0 is greater than max_metric '
               '9500.0, please set a larger max_metric or run smaller systems metric.')


def test_the_torchsim_budget_refusal_is_recognised_as_an_environment_fault():
    """Verbatim torch-sim wording (autobatching.py). If this stops matching, every structure in
    the batch silently becomes val=False again."""
    from demars_core._torchsim import _is_infra_error, _is_oom_error

    exc = ValueError(_TS_REFUSAL)
    assert _is_infra_error(exc), 'the budget refusal is back to being laundered into val=False'
    assert not _is_oom_error(exc), (
        'it must not be treated as an OOM either -- shrinking the budget makes this error '
        'strictly worse, and the retries would burn the whole budget before giving up')


def test_a_budget_refusal_propagates_instead_of_invalidating_every_config(monkeypatch):
    """End-to-end through `batched_fire_relax` with a stub model, no GPU: raising the refusal must
    reach the caller, not come back as `val=[False]` with a nan energy."""
    ts = pytest.importorskip('torch_sim')       # a hard dependency of the package, not of the test

    from demars_core import _torchsim as T

    class _StubModel:
        memory_scales_with = 'n_atoms'
        dtype = None

        def __init__(self):
            import torch
            self.device = torch.device('cpu')   # CPU path: the contract needs no GPU

    def _refuse(*a, **kw):
        raise ValueError(_TS_REFUSAL)

    monkeypatch.setattr(T, '_ACTIVE', (_StubModel(), 'stub', 'stub-key'))
    monkeypatch.setattr(ts, 'optimize', _refuse)

    at = Atoms('Si2', positions=[[0, 0, 0], [2.3, 0, 0]], cell=[10.0] * 3, pbc=True)
    with pytest.raises(ValueError, match='max_metric'):
        T.batched_fire_relax([at], steps=2)


def test_an_ordinary_batch_failure_is_still_val_false(monkeypatch):
    """The other side of the marker: a real per-batch failure must keep mapping onto the invalid
    contract, or the fix would turn every chemistry problem into a crashed campaign."""
    ts = pytest.importorskip('torch_sim')       # a hard dependency of the package, not of the test

    from demars_core import _torchsim as T

    class _StubModel:
        memory_scales_with = 'n_atoms'
        dtype = None

        def __init__(self):
            import torch
            self.device = torch.device('cpu')

    def _blow_up(*a, **kw):
        raise RuntimeError('some genuine batch failure')

    monkeypatch.setattr(T, '_ACTIVE', (_StubModel(), 'stub', 'stub-key'))
    monkeypatch.setattr(ts, 'optimize', _blow_up)

    at = Atoms('Si2', positions=[[0, 0, 0], [2.3, 0, 0]], cell=[10.0] * 3, pbc=True)
    E, rel, val = T.batched_fire_relax([at], steps=2)

    assert list(val) == [False] and np.isnan(E[0]) and len(rel) == 1


def test_the_probe_is_compared_against_the_batch_before_it_is_used():
    """The budget check must sit in the autobatcher branch itself. Only a CUDA device reaches that
    branch, so this pins the wiring the way `test_both_relax_paths_share_one_budget_rule` does --
    a probe that undershoots must degrade to autobatcher=False, not raise."""
    import inspect

    from demars_core import _torchsim as T

    src = inspect.getsource(T.batched_fire_relax)
    assert '_largest_system_metric' in src, \
        'the probed budget is being used without checking it against the batch it must fit'
    assert 'autobatcher=False' in src.split('_largest_system_metric', 1)[1], \
        'nothing degrades to the unbatched path when the probe comes back too small'


def test_a_probe_that_ooms_building_its_own_trial_batch_does_not_kill_the_run():
    """torch-sim's probe allocates each trial batch OUTSIDE its own try/except.

    `determine_max_batch_size` grows the batch by 1.6x until a forward pass OOMs, but it builds the
    trial batch with `ts.concatenate_states([state] * n)` on the line BEFORE the try. When it is the
    concatenation that runs out of memory, the error escapes the probe and takes down a relaxation
    whose structures are fine one at a time. One entry hit this five times, on three GPUs of a
    96 GB node including an idle one, which is what makes it a library bug and not a busy card.

    The fallback must be a budget of ONE LARGEST SYSTEM -- not smaller. A budget below a single
    system's own metric is the refusal D10 is about.
    """
    import demars_core._torchsim as T

    calls = {'n': 0}

    def _oom_probe(*a, **k):
        calls['n'] += 1
        raise RuntimeError('CUDA out of memory. Tried to allocate 25.50 GiB')

    scalers = [1000.0, 2500.0, 1800.0]
    mod = types.SimpleNamespace(calculate_memory_scalers=lambda *a, **k: scalers,
                                estimate_max_memory_scaler=_oom_probe)
    saved = sys.modules.get('torch_sim.autobatching')
    sys.modules['torch_sim.autobatching'] = mod
    T._SCALER_CACHE.pop(('k', None, False), None)
    try:
        model = types.SimpleNamespace(memory_scales_with='n_atoms_x_density')
        with pytest.warns(UserWarning, match='building its own trial batch'):
            budget = T._max_memory_scaler(model, ('k', None, False), object())
    finally:
        if saved is None:
            sys.modules.pop('torch_sim.autobatching', None)
        else:
            sys.modules['torch_sim.autobatching'] = saved
        T._SCALER_CACHE.pop(('k', None, False), None)

    assert calls['n'] == 1
    assert budget == pytest.approx(max(scalers)), \
        'the fallback budget must fit the largest system exactly, or torch-sim refuses the batch'


# ---- D21: recording the review deleted what the review asked for -----------------------------
#
# `gates` was assembled from a literal holding charge / fidelity / sibling_comparison, so the
# connectivity result -- the only gate that catches STRUCTURAL error -- had no place in mar-1.0.
# It went into `decision_trace` prose, where no program reads it, and the review stamp (re-running
# stage ⑥ with `--review`) rebuilt `gates` and dropped anything an analyst had put there by hand.
# One entry lived that loop: round 2 objected that connectivity was missing from `gates`, round 3 added
# it, and the stamp re-run removed it again.
#
# The second half of the defect is quieter: at the 08-17 scoring, 6 of the 15 frozen-run "clean"
# connectivity verdicts had `examined_centres == []`. "Clean" and "looked at nothing" were the same
# value. So these tests pin the STATE vocabulary too, not just the field's presence.

def _minimal_engine():
    """Just enough engine.json for build_record; the gates under test do not read the rest."""
    return {'engine': {}, 'distribution': {}, 'ensemble_files': {}}


def _evidence():
    with open(os.path.join(FIXDIR, HALF_OCC_FIXTURE), encoding='utf-8') as fh:
        return MEV.evidence_from_text(fh.read(), iid=None, search_siblings=False)


def _audit_json(**over):
    """A stage-⑤ (`demars_connectivity.py --json`) payload, single frame, examined and clean."""
    out = {'file': '/x/representative.cif', 'n_frames': 1, 'clean': True, 'tol': 1.25,
           'expect': None, 'n_defects': 0, 'defects': [],
           'per_frame_summary': {'P': {'ligands': ['O'], 'expected': 4},
                                 '_coverage': {'examined_centres': ['P'], 'not_examined': ['Fe']}}}
    out.update(over)
    return out


def _gate(conn):
    from demars_core._engine.mar_record import build_record
    return build_record(None, _minimal_engine(), {}, evidence=_evidence(),
                        connectivity=conn)['gates']['connectivity']


def test_connectivity_has_a_place_in_the_gates_schema():
    """The field must exist whether or not an audit ran -- a gate outside the schema is a gate that
    lives in prose and dies on the next re-run."""
    from demars_core._engine.mar_record import build_record
    gates = build_record(None, _minimal_engine(), {}, evidence=_evidence())['gates']
    assert 'connectivity' in gates, 'gates must carry the connectivity result, not decision_trace'


def test_a_missing_audit_is_not_run_and_never_clean():
    g = _gate(None)
    assert g['state'] == 'not_run' and g['confident'] is False
    assert g['pass'] is not True, 'an audit that never ran must not read as a pass'


def test_a_supplied_audit_survives_a_rebuild_of_the_gates():
    """The actual D21 failure: stamping the review re-runs stage ⑥, which rebuilds `gates`. Passing
    the stage-⑤ artifact -- rather than hand-editing the record -- is what makes it survive."""
    from demars_core._engine.mar_record import build_record
    review = [{'round': 1, 'verdict': 'confirm', 'objections': []}]
    rec = build_record(None, _minimal_engine(), {}, evidence=_evidence(),
                       connectivity=_audit_json(), review=review)
    g = rec['gates']['connectivity']
    assert rec['review'] is not None, 'the stamp under test'
    assert g['state'] == 'derived' and g['pass'] is True, g
    assert g['examined_centres'] == ['P']


def test_an_examined_nothing_audit_is_vacuous_not_a_pass():
    """6 of 15 frozen-run "clean" verdicts were this. Counting them as passes is the laundering the
    scoring caught: `pass is True` has to mean a centre was actually examined."""
    g = _gate(_audit_json(per_frame_summary={'_coverage': {'examined_centres': [],
                                                           'not_examined': ['K', 'Na']}}))
    assert g['state'] == 'vacuous'
    assert g['pass'] is not True, 'nothing was examined, so nothing passed'
    assert g['confident'] is True, 'we are sure there was nothing to check -- that part is honest'
    assert g['not_examined'] == ['K', 'Na'], 'the record must carry what was skipped (D25)'


def test_a_real_defect_fails_whatever_the_coverage_says():
    g = _gate(_audit_json(clean=False, n_defects=3,
                          defects=[{'centre': 'P', 'atom_index': i, 'coordination': 3,
                                    'expected': 4} for i in range(3)]))
    assert g['state'] == 'derived' and g['pass'] is False and g['n_defects'] == 3


def test_a_multiframe_audit_cannot_claim_clean_without_coverage():
    """The CLI reports `per_frame_summary` only for a single frame, so an ensemble audit with zero
    defects cannot tell "checked and clean" from "examined nothing"."""
    g = _gate(_audit_json(n_frames=14, per_frame_summary=None))
    assert g['state'] == 'ambiguous' and g['pass'] is not True and g['confident'] is False


def test_the_gate_states_come_from_the_one_closed_vocabulary():
    """A second state vocabulary is the bug PROV_STATES exists to prevent."""
    seen = {_gate(None)['state'],
            _gate(_audit_json())['state'],
            _gate(_audit_json(per_frame_summary={'_coverage': {'examined_centres': []}}))['state'],
            _gate(_audit_json(n_frames=2, per_frame_summary=None))['state']}
    assert seen <= set(MEV.PROV_STATES), seen


# ---- Phase 5 「게이트 자동화」: the two gates that used to wait to be called --------------------
#
# charge and fidelity are recomputed by stage ⑥ automatically and see COMPOSITION only. The two
# checks that catch STRUCTURAL error -- connectivity, and "did we ship the clustered draw" -- depended
# on the analyst calling them. The 08-17 scoring measured the cost of that inversion at the best
# possible conditions (15/15 run, 15/15 reviewed): `gates.connectivity` present on 0 entries, and two
# analysts declaring the gate inapplicable where it in fact applied and was clean. Automation was
# blocked until the gate stopped over-flagging (D28); it no longer does.

def _structure_file(tmp_path):
    """Three intact SO4 written to a file, plus the Atoms object so a caller can break a copy."""
    from ase import Atoms
    sym, pos = [], []
    for k in range(3):
        o = np.array([6.0 * k, 0.0, 0.0])
        sym += ['S'] + ['O'] * 4
        pos += [o, o + [1.47, 0, 0], o + [-1.47, 0, 0], o + [0, 1.47, 0], o + [0, 0, 1.47]]
    at = Atoms(''.join(sym), positions=pos, cell=[24.0, 12.0, 12.0], pbc=True)
    p = tmp_path / 'sulfate.vasp'
    at.write(str(p), format='vasp')
    return str(p), at


def _engine_with(rep_file=None, lowest=None):
    return {'engine': {}, 'ensemble_files': {'representative': rep_file} if rep_file else {},
            'distribution': ({'lowest': lowest} if lowest else {})}


def _rec(engine, judg=None, connectivity=None):
    from demars_core._engine.mar_record import build_record
    return build_record(None, engine, judg or {}, evidence=_evidence(), connectivity=connectivity)


def test_connectivity_runs_itself_when_no_artifact_is_handed_in(tmp_path):
    """The inversion this closes: the gate that catches what charge/fidelity cannot no longer waits
    to be called."""
    f, _at = _structure_file(tmp_path)
    g = _rec(_engine_with(rep_file=f))['gates']['connectivity']
    assert g['state'] == 'derived' and g['pass'] is True, g
    assert g['source'].startswith('auto'), g['source']
    assert g['examined_centres'] == ['S']


def test_the_auto_run_finds_a_real_defect(tmp_path):
    """Automation is worth nothing if it only ever agrees."""
    f, at = _structure_file(tmp_path)
    del at[len(at) - 1]                              # strip one O -> an SO3 among SO4
    at.write(str(tmp_path / 'broken.vasp'), format='vasp')
    g = _rec(_engine_with(rep_file=str(tmp_path / 'broken.vasp')))['gates']['connectivity']
    assert g['pass'] is False and g['n_defects'] == 1, g


def test_a_supplied_artifact_still_wins_over_the_auto_run(tmp_path):
    """An artifact can carry what the auto-run cannot: --expect, and a multi-frame ensemble audit."""
    f, _at = _structure_file(tmp_path)
    g = _rec(_engine_with(rep_file=f), connectivity=_audit_json())['gates']['connectivity']
    assert 'artifact' in g['source'] and g['examined_centres'] == ['P'], g


def test_an_unreadable_representative_degrades_to_not_run_instead_of_raising(tmp_path):
    """9 of the 15 frozen-run records already carry a representative path that no longer resolves --
    an archived or moved tree is normal. A probe that kills the run is not a probe (D24)."""
    g = _rec(_engine_with(rep_file=str(tmp_path / 'gone.vasp')))['gates']['connectivity']
    assert g['state'] == 'not_run' and g['pass'] is None
    assert 'does not resolve' in g['basis'], g['basis']


def test_shipping_the_clustered_lowest_fails_the_sqs_gate():
    """`strategy.md` says not to ship a clustered lowest as-is. Nobody enforced it."""
    g = _rec(_engine_with(lowest={'mode': 'clustered', 'label': 'cpl1'}))['gates']['sqs']
    assert g['state'] == 'derived' and g['pass'] is False
    assert g['lowest_mode'] == 'clustered' and 'strategy.md' in g['basis']


def test_setting_the_clustered_draw_aside_passes():
    """A custom build, or a repick inside the same ensemble, is the correct response -- not a failure."""
    judg = {'representative_override': {'file': None, 'note': 'built by hand'}}
    g = _rec(_engine_with(lowest={'mode': 'clustered', 'label': 'cpl1'}), judg=judg)['gates']['sqs']
    assert g['pass'] is True and 'set aside' in g['basis'], g


def test_a_random_lowest_passes_and_no_enumeration_is_vacuous():
    assert _rec(_engine_with(lowest={'mode': 'random', 'label': 'rand3'}))['gates']['sqs']['pass'] is True
    g = _rec(_engine_with())['gates']['sqs']
    assert g['state'] in ('vacuous', 'not_run') and g['pass'] is None, g


# ---- D19: the only way to ship a different frame was to call it a custom build ------------------
#
# `strategy.md` asks the analyst NOT to ship a `lowest.mode == 'clustered'` draw as-is, and choosing
# another frame of the SAME ensemble is the normal response. The only field that could express it was
# `representative_override`, which means "the analyst built this by hand", so (a) a reselection was
# recorded as a custom build -- `source: custom-build`, gate basis `custom-build structure ...` -- for
# a frame the engine itself enumerated, with the SELECTION reason sitting in `custom_reason`; and
# (b) the ensemble block and the representative came from different places, which drove an analyst to
# copy `ensemble.xyz` next to the override and leave two byte-identical ensembles in one run dir.

def _labelled_ensemble(tmp_path, labels=('rand0', 'rand2', 'clustered')):
    """An extxyz ensemble carrying the per-frame `label`/`mode`/`E_per_atom` the engine writes."""
    from ase import Atoms
    from ase.io import write
    frames = []
    for k, lab in enumerate(labels):
        at = Atoms('SO4', positions=[[0, 0, 0], [1.47, 0, 0], [-1.47, 0, 0],
                                     [0, 1.47, 0], [0, 0, 1.47]],
                   cell=[12.0] * 3, pbc=True)
        at.info.update(label=lab, mode=('clustered' if lab == 'clustered' else 'random'),
                       E_per_atom=-3.0 - 0.001 * k, charge=0.0)
        frames.append(at)
    p = tmp_path / 'ensemble.xyz'
    write(str(p), frames, format='extxyz')
    return str(p)


def _pick_engine(ens, lowest_mode='clustered', lowest_label='clustered'):
    return {'engine': {'supercell': [1, 1, 1], 'group_orbits': {}},
            'ensemble_files': {'ensemble': ens},
            'distribution': {'n_relaxed': 3, 'n_total': 3,
                             'energies_sorted': [-3.002, -3.001, -3.0],
                             'lowest': {'mode': lowest_mode, 'label': lowest_label,
                                        'composition': {'S': 1, 'O': 4}, 'E_per_atom': -3.002}}}


def test_a_repick_is_recorded_as_a_repick_not_a_custom_build(tmp_path):
    ens = _labelled_ensemble(tmp_path)
    judg = {'representative_pick': {'config_label': 'rand2', 'reason': 'the lowest is the clustered draw'}}
    rec = _rec(_pick_engine(ens), judg=judg)
    rp = rec['representative']
    assert rp['source'] == 'engine-repick', rp['source']
    assert rp['config_label'] == 'rand2' and rp['frame_index'] == 1
    assert rp['pick_reason'] == 'the lowest is the clustered draw'
    assert 'custom_reason' not in rp, 'a selection reason must not sit in a build reason slot'
    assert rec['gates']['charge']['basis'].startswith('engine-enumeration'), \
        'a frame the engine enumerated keeps the enumeration gate basis'


def test_the_ensemble_block_still_comes_from_the_same_engine_json(tmp_path):
    """(b): the copy-the-ensemble workaround left two byte-identical ensembles in one run dir. A
    repick reads the frame out of THE ensemble, so there is nothing to copy."""
    ens = _labelled_ensemble(tmp_path)
    judg = {'representative_pick': {'config_label': 'rand2', 'reason': 'r'}}
    rec = _rec(_pick_engine(ens), judg=judg)
    assert rec['ensemble']['file'] == ens
    assert rec['representative']['file'] == ens, 'the ensemble IS the file; the frame index picks it out'
    assert rec['ensemble']['n_configs'] == 3


def test_the_connectivity_gate_audits_the_repicked_frame(tmp_path):
    ens = _labelled_ensemble(tmp_path)
    judg = {'representative_pick': {'config_label': 'clustered', 'reason': 'r'}}
    g = _rec(_pick_engine(ens), judg=judg)['gates']['connectivity']
    assert g['frame_index'] == 2 and g['n_frames'] == 1, g
    assert g['state'] == 'derived' and g['examined_centres'] == ['S']


def test_a_repick_says_why_its_final_energy_is_missing(tmp_path):
    """The final tier ran on the enumeration's lowest, not on this frame. A bare null would read as
    'never requested' (D23)."""
    ens = _labelled_ensemble(tmp_path)
    judg = {'representative_pick': {'config_label': 'rand2', 'reason': 'r'}}
    rp = _rec(_pick_engine(ens), judg=judg)['representative']
    assert rp['E_final_eV_per_atom'] is None
    assert rp['E_final_unavailable_reason'] and 'final tier' in rp['E_final_unavailable_reason']
    assert rp['E_nano_eV_per_atom'] is not None, 'the frame carries its own engine-tier energy'


def test_an_unresolved_pick_does_not_quietly_ship_the_rejected_frame(tmp_path):
    ens = _labelled_ensemble(tmp_path)
    judg = {'representative_pick': {'config_label': 'rand99', 'reason': 'typo'}}
    rec = _rec(_pick_engine(ens), judg=judg)
    rp = rec['representative']
    assert rp.get('pick_unresolved'), 'silence here ships the frame the analyst rejected'
    assert 'rand99' in rp['pick_unresolved']
    assert rec['gates']['sqs']['pass'] is False, \
        'and the clustered lowest it fell back to fails the SQS gate on top'


def test_an_override_and_a_pick_together_are_recorded_as_a_conflict(tmp_path):
    ens = _labelled_ensemble(tmp_path)
    judg = {'representative_pick': {'config_label': 'rand2', 'reason': 'r'},
            'representative_override': {'file': str(tmp_path / 'nope.vasp'), 'note': 'built'}}
    rp = _rec(_pick_engine(ens), judg=judg)['representative']
    assert rp['source'] == 'custom-build'
    assert 'both representative_override and representative_pick' in rp['pick_unresolved']


def test_setting_the_clustered_draw_aside_by_repick_passes_sqs(tmp_path):
    ens = _labelled_ensemble(tmp_path)
    judg = {'representative_pick': {'config_label': 'rand2', 'reason': 'r'}}
    g = _rec(_pick_engine(ens), judg=judg)['gates']['sqs']
    assert g['pass'] is True and 'set aside' in g['basis'], g


def test_a_dispersed_probe_fails_the_sqs_gate_too(tmp_path):
    """N1 (08-18 scoring): three records shipped a bracket probe with `source: engine-lowest` and no
    disclosure -- one clustered and TWO dispersed. A gate covering `clustered` alone would
    still miss two of the three, so both extremes count."""
    ens = _labelled_ensemble(tmp_path, labels=('rand0', 'dispersed'))
    for mode in ('clustered', 'dispersed'):
        g = _rec(_pick_engine(ens, lowest_mode=mode, lowest_label=mode))['gates']['sqs']
        assert g['pass'] is False, (mode, g)
        assert 'BRACKET PROBE' in g['basis'], g['basis']
    g = _rec(_pick_engine(ens, lowest_mode='random', lowest_label='rand0'))['gates']['sqs']
    assert g['pass'] is True, g


# ---- D32: charge balancing erased the disorder and the one place a gate looks did not see it -----
#
# A layered alkali oxide: the alkali 3b site is refined at occ 0.95(1) -- a ~5 sigma sub-stoichiometry, i.e.
# the disorder the entry exists to represent. With one lever available the balancer filled it back to
# 100% to satisfy the absolute |q| < 0.3 gate, and `provenance_summary` still reported
# `n_unconfident = 0`. `charge_balancing` was in the engine output but never as PROVENANCE, so the
# rollup a gate is supposed to check could not count it; the analyst found it by reading engine.json.

def _li_orbit(n_sites=20, n_li=19, rh=20, o=40):
    """That shape: one alkali orbit, refined short of full, in an otherwise ordered oxide frame."""
    go = {'Li': [[i] for i in range(n_sites)]}
    go_tgt = {'Li': {('sub', 'Li'): n_li}}
    return go, go_tgt, {'Rh': rh, 'O': o}, {'Li': 1, 'Rh': 3, 'O': -2}


def test_filling_a_refined_vacancy_to_reach_neutrality_is_not_confident():
    go, go_tgt, fixed, ox = _li_orbit()
    tgt, st = ME.balance_charge(go, go_tgt, fixed, ox)

    assert st['adjusted'] and tgt['Li'][('sub', 'Li')] == 20, 'the pathology itself: 19/20 -> 20/20'
    prov = st['provenance']
    assert prov['state'] == 'ambiguous' and prov['confident'] is False, prov
    assert prov['evidence']['orbits_filled_to_full'] == ['Li'], prov['evidence']
    assert 'Li' in prov['detail'] or 'FULL' in prov['detail']


def test_the_rollup_counts_it_so_a_gate_can_see_it():
    """The whole point: `n_unconfident` is what stage ② gates on, and it read 0 while this happened."""
    _go, _t, _f, _ox = _li_orbit()
    _tgt, st = ME.balance_charge(_go, _t, _f, _ox)
    roll = MEV.provenance_summary({'charge_balancing': st['provenance']})
    assert roll['n_unconfident'] == 1 and roll['unconfident'] == ['charge_balancing']


def test_the_engine_output_carries_the_block_at_all():
    """Wiring: the balancer's provenance has to reach `provenance_summary`, not just
    `charge_balancing`. It was absent from the rollup's inputs entirely."""
    for fx in PROV_FIXTURES:
        info = _prov_build(fx)
        assert 'provenance' in info['charge_balancing'], fx
        assert 'charge_balancing' in info['provenance_summary']['states'], fx


def test_a_nudge_that_leaves_the_orbit_partial_stays_derived():
    """Moving counts toward neutrality is the DESIGNED behaviour -- a refined occupancy can itself be
    charged. Only removing the disorder outright is the unconfident case, so this must not overreport."""
    go = {'M': [[i] for i in range(20)]}
    go_tgt = {'M': {('sub', 'Na'): 9, ('sub', 'Ca'): 9}}       # 18/20 occupied, q = 9 + 18 - 26 = +1
    tgt, st = ME.balance_charge(go, go_tgt, {'O': 13}, {'Na': 1, 'Ca': 2, 'O': -2})
    prov = st['provenance']
    assert st['adjusted'] and st['q_final'] == 0
    assert sum(tgt['M'].values()) < 20, 'the orbit has to still be partial for this to be the case'
    assert prov['state'] == 'derived' and prov['confident'] is True, prov
    assert prov['evidence']['orbits_filled_to_full'] == []


def test_an_already_neutral_cell_is_vacuous_not_derived():
    """Nothing moved, so nothing was decided -- and `vacuous` is confident, which keeps the rollup
    from filling up with noise on the common case."""
    _go, _t, _f, _ox = _li_orbit(n_sites=20, n_li=20)          # q = 20 + 60 - 80 = 0
    _tgt, st = ME.balance_charge(_go, _t, _f, _ox)
    assert st['adjusted'] is False
    assert st['provenance']['state'] == 'vacuous' and st['provenance']['confident'] is True


def test_nothing_moved_on_a_still_charged_cell_is_not_vacuous():
    """`adjusted: False` has two causes and only one of them is 'nothing to decide'.

    When the balancer never moved anything AND the cell is still charged -- it had no lever, or a
    driver disabled it -- the old branch reported `vacuous`, i.e. 'the realized occupancies were
    already charge-neutral', in the same object that carried q_per_cell = -13. `vacuous` is
    CONFIDENT, so `n_unconfident` read 0 and the rollup a gate checks saw nothing: the same blind
    spot D32 closed for the adjusted branch, one branch over.
    """
    _tgt, st = ME.balance_charge({}, {}, {'Li': 19, 'Rh': 20, 'O': 40},
                                 {'Li': 1, 'Rh': 3, 'O': -2})       # q = 19 + 60 - 80 = -1, no lever
    assert st['adjusted'] is False and st['q_initial'] != 0
    prov = st['provenance']
    assert prov['state'] == 'ambiguous' and prov['confident'] is False, prov
    assert prov['evidence']['q_per_cell'] == st['q_initial'], prov['evidence']
    assert 'already charge-neutral' not in prov['detail'], prov['detail']
    roll = MEV.provenance_summary({'charge_balancing': prov})
    assert roll['n_unconfident'] == 1 and roll['unconfident'] == ['charge_balancing']


# ---- D37: the COORD_FORMERS gap bit the BUILD, and no cutoff value could reach it ----------------
#
# D5 already REPORTS this: an `unmatched` couple_cut says "contacts exist, none matched the tables"
# and names the elements -- Ga in one, Co in another. What was missing was a way to act on it.
# `--couple-cut` retargets a cutoff; it cannot ADD a role, so the chemistry stayed unreachable by any
# value. Measured on the real Ga chalcogenide: default -> `unmatched`, n_former_anion 0, cutoff falls back to
# the 2.0 A module default while the Ga-Te bond sits at 2.55 A. With the role named: `overridden`,
# n_former_anion 1, cutoff derived as 2.55 x 1.15 = 2.93 A.
#
# `cod_1544358` is the bundled stand-in (11 cross-orbit contacts, 0 matched, Fe + O -- Fe is absent
# from COORD_FORMERS exactly as Ga is). The frozen run lives under gitignored `tmp/`, so a test that
# read that entry would pass here and vanish in a clone.

COUPLE_GAP_FIXTURE = 'cod_1544358.cif'


def test_the_table_gap_is_reported_as_unmatched_and_says_what_to_do():
    info = _prov_build(COUPLE_GAP_FIXTURE)
    p = info['couple_cut']['provenance']
    assert p['state'] == 'unmatched' and p['confident'] is False, p
    assert p['evidence']['n_contacts'] > 0 and p['evidence']['n_former_anion'] == 0
    assert '--couple-formers' in p['detail'], \
        'a report the analyst cannot act on is what D37 was -- the detail must name the flag'


def test_naming_the_role_reaches_the_build():
    """The fix: the cutoff now DERIVES from the measured contact instead of falling back."""
    base = _prov_build(COUPLE_GAP_FIXTURE)['couple_cut']
    over = _prov_build(COUPLE_GAP_FIXTURE, couple_formers={'Fe'})['couple_cut']
    p = over['provenance']
    assert p['state'] == 'overridden' and p['source'] == '--couple-formers/--couple-anions'
    assert p['evidence']['n_former_anion'] > 0, 'the contact has to become visible to the derivation'
    assert p['evidence']['roles']['formers'] == ['Fe']
    assert over['value_A'] != base['value_A'], (base['value_A'], over['value_A'])
    # the value now comes from a MEASUREMENT rather than the module fallback. Do not recompute the
    # number here: the derivation uses the shortest FORMER-ANION contact, while `closest_A` is the
    # shortest of ALL cross-orbit contacts -- on this fixture a 1.26 A split-O pair that is not one.
    assert 'shortest former-anion contact' in over['basis'], over['basis']
    assert 'default' in base['basis'], base['basis']


def test_the_engine_still_promotes_nothing_by_itself():
    """Measured, not caution: asking which elements BEHAVE like formers (prevalence + uniform CN)
    fired on La, Mg, Fe, Ti, Zr, Ca, Cs, Er, Ni, Zn, Gd, Cu, Ba and a K with CN 27 across 15
    structures -- see `unexplained_dangling`'s "WHAT NOT TO DO". Whether Fe centres a unit here is
    chemistry, so the default must stay the table and the analyst must have to say so."""
    p = _prov_build(COUPLE_GAP_FIXTURE)['couple_cut']['provenance']
    assert p['evidence']['roles'] == {'formers': 'COORD_FORMERS', 'anions': 'COORD_ANIONS'}


def test_one_role_set_governs_the_cutoff_and_the_co_placement():
    """The cutoff and the coupling precompute must not disagree about who is a former -- a fixed table
    in one and an overridable set in the other would make `--couple-formers` half-work."""
    import inspect
    src = inspect.getsource(ME.build)
    sites = [ln for ln in src.splitlines() if 'orbit_els(sg) &' in ln]
    assert sites, 'the coupling precompute moved -- re-point this test'
    for ln in sites:
        assert 'cplF' in ln or 'cplA' in ln, ln
        assert 'COORD_FORMERS' not in ln and 'COORD_ANIONS' not in ln, ln


# ---- D29 + D35: the record could not prove its own construction ---------------------------------
#
# Two halves of one problem: scoring 100 entries by `generation_recipe` was impossible.
#   D29 -- `--same-excl` corrects the SAME-element cut, but the recipe copied only the CROSS-element
#          one (`exclusion_A`). Measured on one entry: round 2 fixed the structure with
#          `--same-excl 3.0` and `generation_recipe` came out BYTE-IDENTICAL to round 1.
#   D35 -- `merged: false` means "not clique-merged", NOT "no exclusion": a non-clique component is
#          split into singletons and its conflict edges are enforced through `excl_adj`, which never
#          left the engine. The field name reads as the opposite and produced three consecutive
#          misreads on one entry; the scorer re-derived exclusion from CIF operators by hand on three.

MERGE_FIXTURE = 'cod_1544358.cif'          # derives a per-element same-element cut
CHAIN_FIXTURE = 'cod_4000330.cif'          # 20 non-clique components -> pairwise exclusion


def _recipe(fixture, **kw):
    from demars_core._engine.mar_record import build_record
    txt = open(os.path.join(FIXDIR, fixture)).read()
    ev = MEV.evidence_from_text(txt, iid=None, search_siblings=False)
    _e, _b, info = ME.build(structure=ME.structure_from_text(txt), evidence=ev,
                            relax=_constant_total_energy, NR=2, MIN_A=8.0, seed=0, **kw)
    rec = build_record(None, {'engine': info, 'distribution': {}, 'ensemble_files': {}}, {}, evidence=ev)
    return rec['generation_recipe']


def test_a_corrected_same_excl_run_is_distinguishable_from_a_default_one():
    """The measured failure: the two recipes were byte-identical, so a corrected build and an
    uncorrected one scored the same."""
    base = _recipe(MERGE_FIXTURE)
    fixed = _recipe(MERGE_FIXTURE, same_excl=3.0)
    assert json.dumps(base, sort_keys=True, default=str) != json.dumps(fixed, sort_keys=True, default=str)
    cuts = fixed['exclusion_merge']['same_element_cut_per_element']
    assert cuts and set(cuts.values()) == {3.0}, cuts
    assert base['exclusion_merge']['same_element_cut_per_element'] != cuts


def test_the_recipe_carries_the_same_element_cut_at_all():
    """`exclusion_merge_A` is the CROSS-element cut. The one `--same-excl` moves has to be there too."""
    r = _recipe(MERGE_FIXTURE)
    assert 'exclusion_merge' in r and r['exclusion_merge'], 'the report itself must be copied'
    assert 'same_element_cut_per_element' in r['exclusion_merge']
    assert r['exclusion_merge_A'] is not None, 'the cross-element cut stays where it was'


def test_merged_false_says_whether_exclusion_is_enforced():
    """D35's exact misread. One structure carries BOTH cases, so the distinction is measured rather
    than hypothetical: `merged=false` + `pairwise` (enforced) beside `merged=false` + `none` (not)."""
    orbits = _recipe(CHAIN_FIXTURE)['group_orbits']
    modes = {oid: (o['merged'], o['exclusion']['mode'], o['exclusion']['n_pairs'])
             for oid, o in orbits.items()}
    enforced = [m for m in modes.values() if m[1] == 'pairwise']
    inert = [m for m in modes.values() if m[1] == 'none']
    assert enforced and inert, modes
    for merged, _mode, n_pairs in enforced:
        assert merged is False and n_pairs > 0, 'pairwise exclusion lives on UNmerged orbits'
    for _merged, _mode, n_pairs in inert:
        assert n_pairs == 0


def test_a_clique_merged_orbit_says_so():
    orbits = _recipe(MERGE_FIXTURE)['group_orbits']
    merged = [o for o in orbits.values() if o['merged']]
    assert merged, 'this fixture is here because it merges'
    for o in merged:
        assert o['exclusion']['mode'] == 'clique-merged' and o['exclusion']['n_pairs'] == 0


def test_every_orbit_states_its_exclusion_mode():
    """No orbit may leave it unsaid -- silence is what `merged` alone was."""
    for fx in (MERGE_FIXTURE, CHAIN_FIXTURE, 'cod_9003141.cif'):
        for oid, o in _recipe(fx)['group_orbits'].items():
            assert o['exclusion']['mode'] in ('clique-merged', 'pairwise', 'none'), (fx, oid, o)


# ---- D42 + D43: the same-element cut read the wrong radius and ate a full sublattice ------------
#
# D42 was real: the widest-gap derivation only inspected same-element pairs under a 1.3 A window, so
# a split pair sitting ABOVE it left no cut set at all and the two images were free to co-occupy.
# The IMPOSSIBILITY SPLIT closed that -- a pair under 0.9 * 2 r_cov cannot be a real bond of that
# element, wherever the widest gap happens to fall.
#
# D43 is what that fix cost. `covalent_radii` is a COVALENT SINGLE-BOND radius, and metallic
# nearest-neighbour spacing at CN 8-12 sits below 0.9 * 2 r_cov for much of the transition-metal and
# rare-earth block -- nine elements fail on their pure metal alone, and 38 of 65 sit within 0.3 A of
# the threshold, which ordinary compound compression erases. On a fully-occupied MIXED sublattice
# (an intermetallic solid solution) the split therefore fired, union-find swallowed the sublattice
# into one exclusion network, and every decoration on every seed came back short-filled. The engine
# already computed the number that refutes it -- `suspicious_merges` summed the occupancies and said
# in its own note that no code branched on it.
#
# Distance still selects the candidates -- the guard sits behind the split, not in front of it -- and
# occupancy then vetoes them: an alternate needs a vacancy to alternate INTO. A pair the split never
# reached is untouched and stays an open chemistry question in `out_of_window` (D42, half closed). The
# bcc fixture is a synthetic idealisation of that shape (a = 3.83 A, one mixed site at occ 1.0,
# Tm-Tm = 3.317 A against a 3.42 A threshold), written here rather than taken from a database.

FULL_MIXED_FIXTURE = 'synth_bcc_full_mixed.cif'


def test_a_fully_occupied_same_element_orbit_is_never_merged():
    """D43. occ sum 1.0 on every site -> no two of them can be images of one position."""
    info = _prov_build(FULL_MIXED_FIXTURE)
    em = info['exclusion_merge']
    assert 'Tm' not in em['same_element_cut_per_element'], em['same_element_cut_per_element']
    prov = em['provenance']['Tm']
    assert prov['state'] == 'derived' and prov['evidence']['all_sites_full'] is True, prov
    for oid, o in (info['group_orbits'] or {}).items():
        assert o['exclusion']['mode'] == 'none' and o['exclusion']['n_pairs'] == 0, (oid, o)


def test_the_guard_reads_occupancy_and_not_distance():
    """D43. The pair that used to trigger the merge is still there -- only the verdict changed."""
    info = _prov_build(FULL_MIXED_FIXTURE)
    basis = info['exclusion_merge']['basis']['Tm']
    assert 'fully occupied' in basis and 'no cut set' in basis, basis
    assert 'alternates' not in info['exclusion_merge']['same_element_cut_per_element'], basis


def test_a_partial_occupancy_split_pair_still_merges():
    """D42 must stay closed: the guard is occupancy-gated, so real split images are untouched."""
    em = _prov_build(MERGE_FIXTURE)['exclusion_merge']
    assert em['same_element_cut_per_element'].get('Fe'), em['same_element_cut_per_element']
    assert 'cannot be a real Fe-Fe bond' in em['basis']['Fe'], em['basis']['Fe']


def test_the_override_still_reaches_a_full_orbit():
    """--same-excl is the documented escape hatch and must outrank the guard."""
    em = _prov_build(FULL_MIXED_FIXTURE, same_excl=1.3)['exclusion_merge']
    assert em['same_element_cut_per_element'].get('Tm') == 1.3, em
    assert em['provenance']['Tm']['state'] == 'overridden', em['provenance']['Tm']
