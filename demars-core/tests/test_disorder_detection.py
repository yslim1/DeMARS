"""Fast tests (no MLIP / no relaxation): input handling + ICSD-free disorder
detection on every COD fixture. These cover the plumbing that runs before any
structure is relaxed, so they are quick and GPU-free."""
import json
import os

import pytest
from pymatgen.io.cif import CifParser

from demars_core.io import to_cif_text
from demars_core._engine import mar_engine as engine
from demars_core._engine import mar_evidence as evidence

FIXDIR = os.path.join(os.path.dirname(__file__), 'fixtures')
MANIFEST = {k: v for k, v in json.load(open(os.path.join(FIXDIR, 'manifest.json'))).items()
            if k.endswith('.cif')}
ALL = sorted(MANIFEST)
COMPLEX = [n for n in ALL if MANIFEST[n]['complexity'] == 'complex']


def _p(name):
    return os.path.join(FIXDIR, name)


@pytest.mark.parametrize('name', ALL)
def test_fixture_parses_disordered(name):
    """Every fixture is genuinely disordered (partial occupancy / mixed sites)."""
    s = engine.structure_from_text(to_cif_text(_p(name)))
    assert not s.is_ordered, f'{name} parsed as ordered — not a disorder fixture'


@pytest.mark.parametrize('name', ALL)
def test_evidence_is_icsd_free_and_detects_disorder(name):
    """evidence_from_text derives disorder with NO ICSD context (iid=None): at
    least one disordered orbit, oxidation states present, and — the tell that the
    ICSD-free path ran — the ordered-sibling DB search is skipped."""
    ev = evidence.evidence_from_text(to_cif_text(_p(name)))
    assert ev['disorder']['n_disordered_orbits'] >= 1
    assert 'oxidation_states' in ev
    assert ev['ordered_sibling'] == 'not searched'


@pytest.mark.parametrize('name', COMPLEX)
def test_complex_fixtures_have_rich_disorder(name):
    """The 'complex' fixtures (multi-sublattice / split-site) show >1 disordered
    orbit — the engine sees the richer structure, not a single mixed site."""
    ev = evidence.evidence_from_text(to_cif_text(_p(name)))
    assert ev['disorder']['n_disordered_orbits'] >= 2, \
        f'{name} expected multiple disordered orbits, got {ev["disorder"]["n_disordered_orbits"]}'


def test_to_cif_text_accepts_path_text_and_structure():
    """Input normalisation handles a CIF path, raw CIF text, and a pymatgen
    Structure — all funnel to CIF text with disorder preserved."""
    p = _p(ALL[0])
    text = to_cif_text(p)                                    # from a path
    assert '_atom_site' in text
    assert to_cif_text(text) == text                         # raw CIF text passes through unchanged
    s = CifParser.from_str(text, occupancy_tolerance=1.15).parse_structures(primitive=False)[0]
    assert '_atom_site' in to_cif_text(s)                    # from a pymatgen Structure (disorder kept)
    assert not s.is_ordered
