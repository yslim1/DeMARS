"""A custom build must leave the same files behind as the stock engine.

When `deaverage()` cannot handle a structure the analyst writes a bespoke build script, and that
script inherits none of the engine's guarantees. One such script printed its energies and saved
nothing: a finished 21-minute relaxation left no ensemble, no representative and nothing to gate,
so the whole relaxation was repeated. These tests pin the contract that closes that hole -- same
three filenames, same sample schema, same reader.
"""
import json
import os

import numpy as np
import pytest
from ase import Atoms

from demars_core.record import write_custom_ensemble

FILES = ('ensemble.xyz', 'representative.cif', 'representative.vasp', 'deaverage_output.json')


def _frames(n=3):
    out = []
    for i in range(n):
        at = Atoms('NaCl', positions=[[0, 0, 0], [2.8 + 0.01 * i, 0, 0]],
                   cell=[6.0, 6.0, 6.0], pbc=True)
        out.append(at)
    return out


def _write(tmp_path, **kw):
    rel = kw.pop('relaxed', None) or _frames()
    E = kw.pop('E', None) or [-10.0, -10.5, -9.8]
    val = kw.pop('valid', None) or [True] * len(rel)
    return write_custom_ensemble(str(tmp_path), 'x.cif', 'sevennet', rel, E, val,
                                 oxidation_states={'Na': 1.0, 'Cl': -1.0}, **kw)


def test_every_file_the_stock_engine_writes_is_written(tmp_path):
    rec = _write(tmp_path)
    for f in FILES:
        assert os.path.isfile(tmp_path / f), f'{f} not written'
    assert rec.status == 'de-averaged'


def test_the_json_is_the_engines_own_name_not_the_stage_six_one(tmp_path):
    """`record.json` is the stage-6 mar-1.0 record. A custom build must not claim that name."""
    _write(tmp_path)
    assert not os.path.exists(tmp_path / 'record.json')
    d = json.loads((tmp_path / 'deaverage_output.json').read_text())
    assert 'distribution' in d


def test_the_representative_is_the_lowest_energy_frame(tmp_path):
    rec = _write(tmp_path)
    rep = rec.distribution['representative']
    assert rep['E_per_atom'] == pytest.approx(-10.5 / 2, abs=1e-4)
    assert rec.distribution['n_relaxed'] == 3
    assert rec.distribution['spread_meV'] == pytest.approx(350.0, abs=1.0)


def test_samples_carry_the_fields_downstream_reads(tmp_path):
    """The stage-6 record and the reference check read these by name off `representative`."""
    rec = _write(tmp_path)
    rep = rec.distribution['representative']
    for k in ('label', 'mode', 'valid', 'E_per_atom', 'charge', 'n_atoms',
              'composition', 'min_dist_A', 'min_dist_pair'):
        assert k in rep, f'representative is missing {k}'
    assert rep['charge'] == pytest.approx(0.0)          # Na(+1) + Cl(-1)
    assert rep['composition'] == {'Na': 1, 'Cl': 1}


def test_an_invalid_frame_is_excluded_but_still_counted(tmp_path):
    rec = _write(tmp_path, valid=[True, False, True])
    assert rec.distribution['n_relaxed'] == 2
    assert rec.distribution['n_total'] == 3
    frames = (tmp_path / 'ensemble.xyz').read_text().count('Lattice')
    assert frames == 2


def test_the_provenance_survives(tmp_path):
    """Without this the record is silent about which weights produced it."""
    rec = _write(tmp_path, mlip={'model_tag': 'nano', 'checkpoint_sha256': 'abc'})
    d = json.loads((tmp_path / 'deaverage_output.json').read_text())
    assert d['mlip']['checkpoint_sha256'] == 'abc'
    assert rec.mlip['model_tag'] == 'nano'


def test_the_final_tier_is_recorded_when_it_ran(tmp_path):
    rec = _write(tmp_path, final_MAR={'E_final_per_atom': -5.3, 'n_atoms': 2, 'modal': 'mpa'},
                 final_struct=_frames(1)[0])
    assert os.path.isfile(tmp_path / 'representative_final.cif')
    assert rec.final_MAR['E_final_per_atom'] == -5.3


def test_the_gallery_can_read_what_a_custom_build_wrote(tmp_path):
    """The whole point: downstream cannot tell a custom build from a stock one."""
    _write(tmp_path)
    from demars_core.gallery import _find_results
    found = _find_results(str(tmp_path))
    assert found, 'gallery found nothing where a custom build wrote its output'


def test_it_refuses_to_invent_an_ensemble_from_nothing(tmp_path):
    rec = write_custom_ensemble(str(tmp_path), 'x.cif', 'sevennet', _frames(2),
                                [np.nan, np.nan], [False, False])
    assert rec.status == 'error'
    assert not os.path.exists(tmp_path / 'ensemble.xyz')
