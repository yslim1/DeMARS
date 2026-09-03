"""End-to-end tests (need an MLIP backend — marked `mlip`): run the full
`deaverage()` pipeline and check the deliverable is well-formed. Parametrised
over the SMALL fixtures only (few atoms) so the relaxations stay cheap; the large
and complex fixtures are exercised by the fast detection tests. Settings are
deliberately tiny (n_samples=2, min_cell=7) — this is a smoke/integration check,
not a production run (which uses n_samples=30, min_cell>=15)."""
import json
import os

import pytest

from demars_core import deaverage

FIXDIR = os.path.join(os.path.dirname(__file__), 'fixtures')
MANIFEST = {k: v for k, v in json.load(open(os.path.join(FIXDIR, 'manifest.json'))).items()
            if k.endswith('.cif')}
SMALL = [n for n in sorted(MANIFEST) if MANIFEST[n]['sites'] <= 12]   # cheap to relax end-to-end

pytestmark = pytest.mark.mlip


def _p(name):
    return os.path.join(FIXDIR, name)


@pytest.mark.parametrize('name', SMALL)
def test_deaverage_end_to_end(name, calculator, tmp_path):
    rec = deaverage(_p(name), calculator=calculator, n_samples=2, min_cell=7.0,
                    max_rounds=1, final=False, out_dir=str(tmp_path))
    assert rec.status == 'de-averaged', f'{name}: {rec.status} ({rec.error})'
    d = rec.distribution
    assert 1 <= d['n_relaxed'] <= d['n_total']
    assert d['spread_meV'] >= 0
    rep = d['representative']
    assert rep['min_dist_A'] > 0.7, 'catastrophic atomic overlap in the representative'
    assert rep['n_atoms'] > 0
    for fn in ('ensemble.xyz', 'representative.cif', 'deaverage_output.json'):
        assert (tmp_path / fn).exists(), f'{fn} not written to out_dir'
    assert (tmp_path / 'ensemble.xyz').stat().st_size > 0


def test_deaverage_constructed_is_charge_neutral(calculator):
    """A constructed disordered rock salt (Mg2+/Ni2+/O2-, no external data at all)
    de-averages to a charge-neutral representative — the ideal-stoichiometry build."""
    from demars_core._smoketest import _disordered_rocksalt
    rec = deaverage(_disordered_rocksalt(), calculator=calculator,
                    n_samples=2, min_cell=7.0, max_rounds=1, final=False)
    assert rec.status == 'de-averaged'
    assert rec.distribution['n_relaxed'] >= 1
    assert abs(rec.distribution['representative']['charge']) < 0.01


def test_ordered_input_reports_no_disorder(calculator):
    """An ordered structure (no partial occupancy) is reported as no-disorder,
    not force-de-averaged."""
    from pymatgen.core import Structure, Lattice, Species
    mgo = Structure(Lattice.cubic(4.2),
                    [Species('Mg', 2), Species('O', -2)],
                    [[0, 0, 0], [0.5, 0.5, 0.5]])
    rec = deaverage(mgo, calculator=calculator, n_samples=2, min_cell=6.0,
                    max_rounds=1, final=False)
    assert rec.status == 'no-disorder'
