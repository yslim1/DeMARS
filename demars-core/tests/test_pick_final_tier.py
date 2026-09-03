"""A repicked frame's final tier (`mar_record.pick_final_tier`).

The invariant: the shipped structure always carries a final-tier energy. Two of the three sources
hold it by construction — `engine-lowest`, which is what the engine itself relaxes at the final
tier, and `custom-build`, realised through `--from --final`, which relaxes. A repick is the third,
with neither: the final tier ran on the enumeration's lowest, not on the frame that shipped.

`representative_pick.final_from` names the `--from` run that finalized the picked frame and restores
the invariant, WITHOUT turning the reselection back into a custom build (D19(a)) or letting a second
engine.json supply the ensemble (D19(b)).

No MLIP, no GPU, no network.
"""
import json

import pytest

from demars_core._engine.mar_record import pick_final_tier


def _from_run(tmp_path, e_final=-7.8875, n_atoms=317, mode='from', name='engine.json', **extra):
    d = tmp_path / '_work' / 'r0_pick_final'
    d.mkdir(parents=True, exist_ok=True)
    doc = {'mode': mode, 'input': str(tmp_path / '_work' / 'r0' / 'frame.vasp'),
           'files': {'representative_final': str(d / 'representative_final.vasp')},
           'single_MAR': {'E_nano_per_atom': -7.859, 'n_atoms': n_atoms, 'modal': 'mpa'}}
    if e_final is not None:
        doc['single_MAR']['E_final_per_atom'] = e_final
    doc.update(extra)
    p = d / name
    p.write_text(json.dumps(doc))
    return str(p)


# ---- the happy path ---------------------------------------------------------

def test_a_named_from_run_supplies_the_missing_final_tier(tmp_path):
    src = _from_run(tmp_path)
    got, why = pick_final_tier({'config_label': 'rand15', 'final_from': src},
                               {'n_atoms': 317, 'config_label': 'rand15'})
    assert why is None, why
    assert got['E_final_eV_per_atom'] == -7.8875
    assert got['file_final'].endswith('representative_final.vasp')
    assert got['final_from'] == src


def test_no_final_from_is_not_an_error(tmp_path):
    """Most picks will not have one yet; the record says so in `E_final_unavailable_reason` and the
    CLI warns. It is a gap to close, not a malformed judgment."""
    got, why = pick_final_tier({'config_label': 'rand15'}, {'n_atoms': 317})
    assert got is None and why is None


# ---- what it refuses --------------------------------------------------------

def test_a_run_that_is_not_a_from_run_is_refused(tmp_path):
    """A picked frame is finalized by relaxing THAT frame. Pointing at another enumeration would
    take an energy from a different structure entirely."""
    src = _from_run(tmp_path, mode='enumerate')
    got, why = pick_final_tier({'final_from': src}, {'n_atoms': 317})
    assert got is None and 'not a --from run' in why, why


def test_a_from_run_whose_final_tier_did_not_run_is_refused(tmp_path):
    """`--from` records the asked-for-but-could-not-run marker rather than an energy (D10/D23). That
    must not become a silent None on the representative."""
    d = tmp_path / '_work' / 'x'; d.mkdir(parents=True)
    p = d / 'engine.json'
    p.write_text(json.dumps({'mode': 'from', 'single_MAR': {
        'n_atoms': 317, 'final_MAR': {'available': False, 'requested': True,
                                      'reason': 'CUDA out of memory'}}}))
    got, why = pick_final_tier({'final_from': str(p)}, {'n_atoms': 317})
    assert got is None
    assert 'CUDA out of memory' in why, why


def test_a_different_frame_is_refused(tmp_path):
    """The frame that was finalized must be the frame that shipped. Atom count is the cheap check
    both sides carry, and a mismatch means some other structure was relaxed."""
    src = _from_run(tmp_path, n_atoms=200)
    got, why = pick_final_tier({'final_from': src}, {'n_atoms': 317})
    assert got is None and 'not the same frame' in why, why


def test_an_unreadable_path_is_reported_not_raised(tmp_path):
    got, why = pick_final_tier({'final_from': str(tmp_path / 'nope.json')}, {'n_atoms': 317})
    assert got is None and 'could not read' in why, why


# ---- through the record ------------------------------------------------------

def _evidence():
    import os

    from demars_core._engine import mar_evidence as MEV
    fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'fixtures', 'cod_1544358.cif')
    with open(fixture, encoding='utf-8') as fh:
        return MEV.evidence_from_text(fh.read(), iid=None, search_siblings=False)


def _engine_with_ensemble(tmp_path):
    """An engine.json whose ensemble really exists, so `resolve_pick` can find a frame."""
    from ase import Atoms
    from ase.io import write
    d = tmp_path / '_work' / 'r0'
    d.mkdir(parents=True, exist_ok=True)
    frames = []
    for lab, e in (('rand0', -7.80), ('rand15', -7.86)):
        at = Atoms('Cu4', positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]],
                   cell=[8, 8, 8], pbc=True)
        at.info.update(label=lab, mode='random', nano_E_per_atom=e)
        frames.append(at)
    write(str(d / 'ensemble.xyz'), frames, format='extxyz')
    return {'engine': {}, 'distribution': {'lowest': {'label': 'rand0', 'mode': 'random'}},
            'ensemble_files': {'ensemble': str(d / 'ensemble.xyz'), 'n_frames': 2}}


def test_the_repick_keeps_its_own_vocabulary_and_gains_the_energy(tmp_path):
    """D19 stays fixed: the record still says `engine-repick` with a `pick_reason`, not a custom
    build — `final_from` contributes an energy and a file, never the ensemble."""
    from demars_core._engine.mar_record import build_record
    eng = _engine_with_ensemble(tmp_path)
    src = _from_run(tmp_path, e_final=-7.8875, n_atoms=4)
    judg = {'representative_pick': {'config_label': 'rand15', 'reason': 'lowest was a probe',
                                    'final_from': src}}
    rec = build_record(None, eng, judg, evidence=_evidence(), out_dir=str(tmp_path))
    rep = rec['representative']
    assert rep['source'] == 'engine-repick' and rep['config_label'] == 'rand15'
    assert rep['pick_reason'] == 'lowest was a probe'
    assert rep['E_final_eV_per_atom'] == -7.8875
    assert rep['E_final_unavailable_reason'] is None
    assert rep['final_from'], 'the provenance of that energy must be recorded'
    # the ensemble still comes from the ONE --engine given here (D19(b))
    assert rec['ensemble']['file'] == rec['representative']['file']


def test_a_repick_without_it_says_what_to_do(tmp_path):
    from demars_core._engine.mar_record import build_record
    eng = _engine_with_ensemble(tmp_path)
    judg = {'representative_pick': {'config_label': 'rand15', 'reason': 'lowest was a probe'}}
    rec = build_record(None, eng, judg, evidence=_evidence(), out_dir=str(tmp_path))
    rep = rec['representative']
    assert rep['source'] == 'engine-repick'
    assert rep['E_final_eV_per_atom'] is None
    assert 'final_from' in rep['E_final_unavailable_reason']


def test_a_rejected_final_from_is_named_in_the_record(tmp_path):
    """Silently falling back to 'the final tier ran on the lowest' would hide that the analyst DID
    point at a run and it was refused."""
    from demars_core._engine.mar_record import build_record
    eng = _engine_with_ensemble(tmp_path)
    src = _from_run(tmp_path, n_atoms=999)                 # wrong frame
    judg = {'representative_pick': {'config_label': 'rand15', 'final_from': src}}
    rec = build_record(None, eng, judg, evidence=_evidence(), out_dir=str(tmp_path))
    why = rec['representative']['E_final_unavailable_reason']
    assert 'rejected' in why and 'not the same frame' in why, why
