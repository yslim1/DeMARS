"""The convex-hull stability read (`demars_core.hull`).

No MLIP, no GPU, and **no network**: the hull math runs on ASE's EMT over a Cu-Au system, with the
reference phases injected. That is the whole reason `compute_hull(references=...)` exists -- a
stability number whose arithmetic is only ever exercised through a paid API key is a number nobody
can check.

The other half of this file is the refusal states. This is the one DeMARS check that depends on an
external service, so "could not know" has to be reportable, distinguishable from "on the hull", and
never a bare null.
"""
import json
import os

import pytest
from ase.build import bulk

from demars_core.hull import _tier_is_mp_scale, compute_hull, mp_available

pytest.importorskip('ase.calculators.emt')


def _emt():
    from ase.calculators.emt import EMT
    return EMT()


def _cu_au_refs():
    return [('Cu', bulk('Cu', 'fcc', a=3.61, cubic=True)),
            ('Au', bulk('Au', 'fcc', a=4.08, cubic=True))]


def _cu_au_target():
    at = bulk('Cu', 'fcc', a=3.85, cubic=True)         # L1_0-like CuAu ordering
    at[0].symbol = 'Au'
    at[2].symbol = 'Au'
    return at


# ---- the arithmetic ---------------------------------------------------------

def test_a_hull_is_actually_computed_offline():
    res = compute_hull(_cu_au_target(), calculator=_emt(), references=_cu_au_refs())
    assert res['state'] == 'derived', res
    assert res['reason'] is None
    assert res['mode'] == 'self-consistent' and res['self_consistent'] is True
    assert res['chemsys'] == 'Au-Cu'
    m = res['mar']
    assert m['composition'] == 'Au2Cu2' and m['n_atoms'] == 4
    assert isinstance(m['E_above_hull_eV_per_atom'], float)
    # the decomposition of an ordered CuAu against the two elementals is the tie-line midpoint
    assert set(m['decomposition']) == {'Au', 'Cu'}, m
    assert res['n_refs_used'] == 2
    assert res['mlip'], 'the tier that produced these energies must be stamped'


def test_the_target_and_the_references_are_relaxed_at_ONE_tier():
    """The whole validity of an E_above_hull is that target and vertices share a scale. The batch is
    built target-first + references, one `relax` call, so nothing can drift between them."""
    res = compute_hull(_cu_au_target(), calculator=_emt(), references=_cu_au_refs())
    assert res['tier'] == 'ase-generic:EMT', res['tier']
    for row in res['references']:
        assert row['relaxed'] is True and 'E_per_atom' in row, row


def test_an_element_with_no_reference_is_ambiguous_not_a_number():
    """An incomplete simplex is not a hull. Giving Cu-Au a Cu reference only must NOT yield a
    confident-looking number computed against a half-hull."""
    res = compute_hull(_cu_au_target(), calculator=_emt(),
                       references=[('Cu', bulk('Cu', 'fcc', a=3.61, cubic=True))])
    assert res['state'] == 'ambiguous', res
    assert 'Au' in res['reason'], res['reason']
    assert 'E_above_hull_eV_per_atom' not in res.get('mar', {})


def test_no_references_at_all_is_ambiguous():
    res = compute_hull(_cu_au_target(), calculator=_emt(), references=[])
    assert res['state'] == 'ambiguous', res
    assert res['reason']


# ---- the refusals -----------------------------------------------------------

def test_a_disordered_input_is_refused_rather_than_silently_hulled():
    """ASE cannot hold partial occupancies: reading the INPUT CIF drops the minority species, so the
    hull would describe a composition nobody built. The bundled (K,Na)Cl fixture loses its Na."""
    fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'fixtures', 'cod_9003141.cif')
    res = compute_hull(fixture, calculator=_emt(), references=_cu_au_refs())
    assert res['state'] == 'not_run', res
    assert 'DISORDERED' in res['reason'], res['reason']
    assert 'representative_final' in res['reason'], 'the reason must say what to run it on instead'


def test_an_unreachable_materials_project_is_not_run_with_a_reason():
    """No key / no client is UNCHECKED. It must never look like a clean hull, and must say which of
    the two it was so the fix is obvious."""
    ok, why = mp_available()
    if ok:
        pytest.skip('this environment CAN reach MP; the refusal path cannot be exercised')
    res = compute_hull(_cu_au_target(), calculator='omni')       # no references -> needs MP
    assert res['state'] == 'not_run', res
    assert res['reason'] == why
    assert 'mar' not in res, 'a refused hull must not carry a stability number'


def test_every_return_carries_a_state():
    """`hull.json` is read by stage 6, which decides what the record may claim. A path that returned
    a dict with no `state` would be read as a successful hull."""
    from demars_core import hull as H
    import inspect
    src = inspect.getsource(H.compute_hull)
    assert 'return {' not in src, 'compute_hull must return through _result(), which stamps state'
    for res in (compute_hull(_cu_au_target(), calculator=_emt(), references=_cu_au_refs()),
                compute_hull(_cu_au_target(), calculator=_emt(), references=[])):
        assert res['state'] in ('derived', 'ambiguous', 'not_run'), res


def test_the_artifact_is_written_even_when_the_hull_refused(tmp_path):
    """An audited campaign needs the REASON on disk, not just on someone's terminal."""
    res = compute_hull(_cu_au_target(), calculator=_emt(), references=[], out_dir=str(tmp_path))
    written = json.load(open(tmp_path / 'hull.json', encoding='utf-8'))
    assert written['state'] == res['state'] == 'ambiguous'
    assert written['reason']


def test_it_writes_the_tier_relaxed_MAR_beside_the_artifact(tmp_path):
    """A hull run also upgrades the deliverable to the tier it was measured at, so `out_dir` gets `representative_final.{vasp,cif}` next to `hull.json`.
    Point `out_dir` at `_work/<tag>/` -- in the run dir these sit beside the engine's own copies."""
    from ase.io import read as ase_read
    compute_hull(_cu_au_target(), calculator=_emt(), references=_cu_au_refs(), out_dir=str(tmp_path))
    assert sorted(q.name for q in tmp_path.iterdir()) == [
        'hull.json', 'representative_final.cif', 'representative_final.vasp']
    back = ase_read(tmp_path / 'representative_final.vasp')
    assert back.get_chemical_formula() == 'Au2Cu2', back.get_chemical_formula()


def test_the_structure_is_written_after_the_relax_and_before_the_hull(tmp_path):
    """Written as soon as the MAR relaxation succeeds, so the
    tier-relaxed structure survives a hull that cannot then be read. A refusal that happens BEFORE
    the relaxation -- no references, MP unreachable -- writes nothing, because there is no relaxed
    structure yet; that ordering is RAW's too (coverage is checked before the batch)."""
    import inspect

    from demars_core import hull as H
    src = inspect.getsource(H.compute_hull)
    assert src.index('representative_final.vasp') > src.index('E, rel, val = relax(batch)'), \
        'the structure must be written only after the relaxation that produced it'
    assert src.index('representative_final.vasp') < src.index('read = _pd_and_target'), \
        'and before the phase diagram, so a hull that cannot be read does not lose it'

    res = compute_hull(_cu_au_target(), calculator=_emt(), references=[], out_dir=str(tmp_path))
    assert res['state'] == 'ambiguous'
    assert sorted(q.name for q in tmp_path.iterdir()) == ['hull.json'], (
        'a refusal before the relaxation has no structure to write')


def test_every_reference_fetcher_spells_the_meta_keys_the_same_way():
    """Caught live: `mp_entries` returned `n_entries` while the result assembly reads
    `n_candidates`, so every `mp-direct` hull reported `n_candidates: null` -- a field that looks
    like "MP had nothing" when MP had 84 entries. No network needed to pin the contract."""
    import inspect

    from demars_core import hull as H
    for fn in (H.mp_entries, H.mp_reference_structures):
        src = inspect.getsource(fn)
        for key in ("'n_candidates'", "'covered'", "'source'"):
            assert key in src, f'{fn.__name__} does not set {key}'
    assert "ref_meta.get('n_candidates')" in inspect.getsource(H.compute_hull)


# ---- the energy scale (MP2020 corrections) ----------------------------------

def test_the_vasp_settings_are_MPs_own_not_invented():
    """MaterialsProject2020Compatibility DROPS an entry with no run_type/hubbards, so an MLIP energy
    cannot be corrected without them -- and inventing them would silently change the correction by
    ~1.2 eV/atom. They come from MPRelaxSet, so the U assignment is MP's: U on Fe in an oxide, none
    in a sulfide (MP applies U to oxides and fluorides only)."""
    from pymatgen.core import Lattice, Structure

    from demars_core.hull import _mp_entry_parameters
    ox = _mp_entry_parameters(Structure(Lattice.cubic(4.0), ['Fe', 'O'], [[0, 0, 0], [.5, .5, .5]]))
    assert ox['run_type'] == 'GGA+U' and ox['is_hubbard'] is True
    assert ox['hubbards'].get('Fe') == 5.3, ox['hubbards']
    su = _mp_entry_parameters(Structure(Lattice.cubic(4.0), ['Fe', 'S'], [[0, 0, 0], [.5, .5, .5]]))
    assert su['run_type'] == 'GGA' and su['is_hubbard'] is False and not su['hubbards']


def test_our_correction_reproduces_MPs_own_number():
    """The check that the synthesized settings are RIGHT and not merely accepted: MP's own FeO
    entries carry -1.4715 eV/atom, and correcting an entry we built must land on the same value.
    Offline -- the correction table ships with pymatgen."""
    from pymatgen.core import Lattice, Structure

    from demars_core.hull import mp2020_correct
    feo = Structure(Lattice.cubic(4.3), ['Fe', 'O'], [[0, 0, 0], [.5, .5, .5]])
    ent = mp2020_correct(feo, -20.0)
    assert ent is not None, 'MP2020 refused an entry built with MP-derived settings'
    assert round(ent.correction / ent.composition.num_atoms, 4) == -1.4715, ent.correction


def test_an_uncorrectable_entry_is_refused_not_dropped():
    """The scheme returns None rather than raising. Swallowing that leaves one entry on the raw
    scale inside a corrected hull, and the resulting number looks perfectly reasonable."""
    from pymatgen.core import Lattice, Structure
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from pymatgen.entries.computed_entries import ComputedStructureEntry

    bare = ComputedStructureEntry(
        Structure(Lattice.cubic(4.3), ['Fe', 'O'], [[0, 0, 0], [.5, .5, .5]]), -20.0)
    assert MaterialsProject2020Compatibility().process_entry(bare, clean=True) is None, (
        'fixture check: a bare entry must be the case that gets dropped')

    import inspect

    from demars_core import hull as H
    src = inspect.getsource(H.compute_hull)
    assert 'refusing to hull a mixed scale' in src, (
        'compute_hull must abort when an entry cannot be put on the requested scale')


def test_a_correction_free_chemsys_is_unchanged_by_the_scale():
    """Cu-Au has no anion and no U, so both scales must give the SAME hull -- a scale switch that
    moved a metal alloy would mean the correction was being applied asymmetrically."""
    none = compute_hull(_cu_au_target(), calculator=_emt(), references=_cu_au_refs(),
                        corrections='none')
    mp20 = compute_hull(_cu_au_target(), calculator=_emt(), references=_cu_au_refs(),
                        corrections='mp2020')
    assert none['state'] == mp20['state'] == 'derived', (none, mp20)
    assert (none['mar']['E_above_hull_eV_per_atom']
            == mp20['mar']['E_above_hull_eV_per_atom'])
    assert mp20['mar']['correction_eV'] == 0.0


def test_the_scale_is_stamped_in_the_artifact_and_in_the_gate():
    """A `pass` without the scale beside it is not reproducible: on a mixed-valence oxide the two
    scales can differ by ~0.1 eV/atom, which straddles any threshold set near that."""
    from demars_core._engine.mar_record import hull_gate
    for scheme in ('none', 'mp2020'):
        r = compute_hull(_cu_au_target(), calculator=_emt(), references=_cu_au_refs(),
                         corrections=scheme)
        assert r['corrections'] == scheme, r
        g = hull_gate(r)
        assert g['corrections'] == scheme and scheme in g['basis'], g


def test_an_unknown_scale_is_rejected_loudly():
    with pytest.raises(ValueError, match='corrections'):
        compute_hull(_cu_au_target(), calculator=_emt(), references=_cu_au_refs(),
                     corrections='mp2019')


def test_the_default_scale_is_MPs_own_hull_scale():
    """MP's hull IS the corrected hull: across Fe-Ti-O, 75 of 79 materials' published
    `energy_above_hull` reproduce on the mp2020 scale and none on the raw one. A hull built from MP
    references therefore belongs on that scale -- `uncorrected_energy` is internally consistent
    but is not the hull MP publishes. Pinned as a scientific decision."""
    import inspect

    from demars_core.hull import compute_hull as ch
    assert inspect.signature(ch).parameters['corrections'].default == 'mp2020'

    # and the CLI, which drifted once: the library default changed while tools/demars_hull.py kept
    # passing the other scale, so every command-line run silently used it.
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    cli = open(os.path.join(root, 'tools', 'demars_hull.py'), encoding='utf-8').read()
    assert "'--corrections', default='mp2020'" in cli, (
        "tools/demars_hull.py must default to the same scale as compute_hull")


# ---- the mode rule ----------------------------------------------------------

def test_the_default_mode_recomputes_the_references():
    """Pinned as a scientific decision. `mp-direct` measures an MLIP target against MP's DFT
    references, so the MLIP-vs-DFT offset lands on the target alone: three MARs read 12, 32 and
    67 meV/atom above the hull that way and all three sit ON it once the references are recomputed
    at the same tier. The cost is ~20-90 relaxations instead of one, about 2x this stage."""
    import inspect

    from demars_core.hull import MODES, compute_hull as ch
    assert inspect.signature(ch).parameters['mode'].default == 'self-consistent'
    assert MODES == ('self-consistent', 'mp-direct')

    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    cli = open(os.path.join(root, 'tools', 'demars_hull.py'), encoding='utf-8').read()
    assert "'--mode', default='self-consistent'" in cli, (
        'tools/demars_hull.py must default to the same mode as compute_hull')


def test_mp_direct_at_a_tier_off_MPs_scale_is_refused_not_silently_swapped():
    """Falling back would answer a question nobody asked. A data gap is a fallback; a tier that
    cannot share MP's scale is a caller error."""
    res = compute_hull(_cu_au_target(), calculator=_emt(), mode='mp-direct')
    assert res['state'] == 'not_run', res
    assert 'omni-mpa' in res['reason'] and 'self-consistent' in res['reason'], res['reason']


def test_an_unknown_mode_is_rejected_loudly():
    with pytest.raises(ValueError, match='mode'):
        compute_hull(_cu_au_target(), calculator=_emt(), references=_cu_au_refs(), mode='mp_direct')


def test_no_reference_cap_drops_a_candidate_silently():
    """A cap of 90 reference compositions drops candidates silently. Measured, the 5-element
    chemsystems here have 95 and 111 candidates within REF_EHULL_TOL, so such a cap loses up to 21
    possible hull vertices to save ~15% of the atoms -- seconds. Uncapped by default; a cap that IS
    set must name what it dropped."""
    from demars_core.hull import MAX_REFS
    assert MAX_REFS is None, 'a reference cap must not be on by default'

    import inspect

    from demars_core import hull as H
    src = inspect.getsource(H.mp_reference_structures)
    assert "'truncated'" in src and 'max_refs' in src, (
        'a cap, when set, must report the candidates it dropped')


def test_only_omni_mpa_may_borrow_MPs_energies():
    """`mp-direct` takes MP's uncorrected GGA energies as vertices. That is legal ONLY because
    omni-mpa reproduces that scale -- it is a property of those weights, not of MLIPs. Any other
    calculator must re-relax the references instead."""
    assert _tier_is_mp_scale('omni', d3=False) is True
    assert _tier_is_mp_scale('7net-omni', d3=False) is True
    assert _tier_is_mp_scale('omni', d3=True) is False, 'dispersion moves it off MP scale'
    assert _tier_is_mp_scale('sevennet', d3=False) is False, 'nano is not the MP-scale tier'
    assert _tier_is_mp_scale('7net-nano', d3=False) is False
    assert _tier_is_mp_scale(_emt(), d3=False) is False, 'an injected ASE calculator never is'


def test_injected_references_force_the_self_consistent_mode():
    """Supplied references are structures, not energies -- so they must be recomputed, whatever the
    tier claims about MP's scale."""
    res = compute_hull(_cu_au_target(), calculator=_emt(), references=_cu_au_refs())
    assert res['mode'] == 'self-consistent'
    assert res['mar']['below_mp_hull'] is None, (
        "below_mp_hull compares against MP's own hull; in self-consistent mode that comparison "
        'does not exist, and False would assert it does')


# ---- what the record does with it -------------------------------------------

def _minimal_engine():
    return {'engine': {}, 'distribution': {}, 'ensemble_files': {}}


def _evidence():
    from demars_core._engine import mar_evidence as MEV
    fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'fixtures', 'cod_1544358.cif')
    with open(fixture, encoding='utf-8') as fh:
        return MEV.evidence_from_text(fh.read(), iid=None, search_siblings=False)


def test_a_refused_hull_survives_into_the_record_instead_of_going_null():
    """`hull: null` means the stage was never run. A hull that ran and refused used to be nulled out
    here, collapsing "we did not ask" into "we asked and could not know"."""
    from demars_core._engine.mar_record import build_record
    refused = compute_hull(_cu_au_target(), calculator=_emt(), references=[])
    rec = build_record(None, _minimal_engine(), {}, evidence=_evidence(), hull=refused)

    hb = rec['hull']
    assert hb is not None, 'a refused hull must not become null'
    assert hb['state'] == 'ambiguous' and hb['reason']
    assert hb['E_above_hull_eV_per_atom'] is None
    assert rec['representative']['E_above_hull_eV_per_atom'] is None


def test_a_derived_hull_reaches_the_representative():
    from demars_core._engine.mar_record import build_record
    good = compute_hull(_cu_au_target(), calculator=_emt(), references=_cu_au_refs())
    rec = build_record(None, _minimal_engine(), {}, evidence=_evidence(), hull=good)

    hb = rec['hull']
    assert hb['state'] == 'derived' and hb['reason'] is None
    assert hb['E_above_hull_eV_per_atom'] == good['mar']['E_above_hull_eV_per_atom']
    assert rec['representative']['E_above_hull_eV_per_atom'] == hb['E_above_hull_eV_per_atom']
    assert hb['tier'] and hb['mode'] == 'self-consistent'


def test_no_hull_at_all_stays_null():
    """The one thing `null` is allowed to mean, so the state above can mean the other thing."""
    from demars_core._engine.mar_record import build_record
    rec = build_record(None, _minimal_engine(), {}, evidence=_evidence())
    assert rec['hull'] is None


# ---- gates.hull -------------------------------------------------------------

def test_the_hull_is_one_of_the_five_gates():
    from demars_core._engine.mar_record import build_record
    gates = build_record(None, _minimal_engine(), {}, evidence=_evidence())['gates']
    assert {'charge', 'fidelity', 'connectivity', 'sqs', 'hull'} <= set(gates), sorted(gates)


def test_no_hull_artifact_makes_the_gate_unchecked_not_clean():
    """The cost of promoting this to a gate: it is the one gate that cannot run itself, so a
    pipeline that skips stage 4b has an UNCHECKED gate -- and it must LOOK unchecked."""
    from demars_core._engine.mar_record import hull_gate
    g = hull_gate(None)
    assert g['state'] == 'not_run' and g['pass'] is None and g['confident'] is False
    assert 'demars_hull' in g['basis'], 'the basis must say how to fix it'


def test_a_refused_hull_artifact_is_not_a_pass():
    from demars_core._engine.mar_record import hull_gate
    for artifact, want in (({'state': 'not_run', 'reason': 'no key'}, 'not_run'),
                           ({'state': 'ambiguous', 'reason': 'no refs'}, 'ambiguous')):
        g = hull_gate(artifact)
        assert g['state'] == want and g['pass'] is None, g
        assert artifact['reason'] in g['basis']


def test_with_no_threshold_the_gate_reports_and_refuses_to_judge():
    """E_above_hull is computed and REPORTED, never turned into a pass/fail on its own. There is
    no threshold to inherit, so unset means the number is carried and `pass` stays null.
    A made-up default would be a verdict this project never earned."""
    from demars_core._engine.mar_record import HULL_TOL_EV_PER_ATOM, hull_gate
    assert HULL_TOL_EV_PER_ATOM is None, 'no threshold may be invented as a default'

    g = hull_gate({'state': 'derived', 'chemsys': 'A-B-C', 'corrections': 'none',
                   'mar': {'E_above_hull_eV_per_atom': 0.12, 'decomposition': {'AB': 1.0}}})
    assert g['state'] == 'derived', g
    assert g['pass'] is None, 'without a threshold the gate must not claim a verdict'
    assert g['E_above_hull_eV_per_atom'] == 0.12
    assert 'does not judge' in g['basis'], g['basis']


@pytest.mark.parametrize('e,tol,ok', [
    (0.05, 0.10, True),
    (0.10, 0.10, True),        # at the configured line -- inclusive
    (0.15, 0.10, False),
    (0.02, 0.01, False),       # a tighter policy is honoured, not clamped
])
def test_a_configured_threshold_is_the_one_applied(e, tol, ok):
    """The threshold is a POLICY INPUT -- whoever knows the campaign sets it, and the gate records
    which value it used so a verdict can be reproduced."""
    from demars_core._engine.mar_record import hull_gate
    g = hull_gate({'state': 'derived', 'chemsys': 'Au-Cu', 'corrections': 'none',
                   'mar': {'E_above_hull_eV_per_atom': e, 'decomposition': {'Au': 0.5}}}, tol=tol)
    assert (g['state'], g['pass']) == ('derived', ok), g
    assert g['tol_eV_per_atom'] == tol


def test_below_the_hull_is_flagged_not_judged():
    """`below_mp_hull` is a FLAG, not a verdict. Whether a MAR under the hull is a
    better ordering or a reference set missing a competitor was never something code decided."""
    from demars_core._engine.mar_record import hull_gate
    g = hull_gate({'state': 'derived', 'chemsys': 'Au-Cu', 'corrections': 'none',
                   'mar': {'E_above_hull_eV_per_atom': -0.30, 'decomposition': {'Au': 1.0}}})
    assert g['below_reference_hull'] is True and g['pass'] is None, g
    g2 = hull_gate({'state': 'derived', 'chemsys': 'Au-Cu', 'corrections': 'none',
                    'mar': {'E_above_hull_eV_per_atom': 0.02, 'decomposition': {'Au': 1.0}}})
    assert g2['below_reference_hull'] is False


def test_a_unary_chemsys_is_reported_like_any_other():
    """No special case: a one-element hull is the element itself, so
    E_above_hull is 0 by construction -- but nothing here invents a state for that. The number is
    reported and the reader draws the obvious conclusion."""
    from demars_core._engine.mar_record import hull_gate
    g = hull_gate({'state': 'derived', 'chemsys': 'Fe', 'corrections': 'none',
                   'mar': {'E_above_hull_eV_per_atom': 0.0, 'decomposition': {'Fe': 1.0}}})
    assert g['state'] == 'derived' and g['pass'] is None, g
    assert g['E_above_hull_eV_per_atom'] == 0.0


def test_a_derived_hull_reaches_the_gate_end_to_end():
    """With no threshold configured the number arrives and `pass` stays null; `hull_tol` is what
    turns it into a verdict, and stage 6 takes that from --hull-tol or demars.yaml."""
    from demars_core._engine.mar_record import build_record
    good = compute_hull(_cu_au_target(), calculator=_emt(), references=_cu_au_refs())
    ev, eng = _evidence(), _minimal_engine()

    g = build_record(None, eng, {}, evidence=ev, hull=good)['gates']['hull']
    assert g['state'] == 'derived' and g['pass'] is None, g
    assert g['E_above_hull_eV_per_atom'] == good['mar']['E_above_hull_eV_per_atom']

    judged = build_record(None, eng, {}, evidence=ev, hull=good,
                          hull_tol=0.10)['gates']['hull']
    assert judged['pass'] is True and judged['tol_eV_per_atom'] == 0.10, judged
