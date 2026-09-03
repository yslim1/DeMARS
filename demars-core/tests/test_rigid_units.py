"""Generality tests for discrete-unit detection (no MLIP / no GPU).

The rigid-unit constraint, the connectivity gate and the coordination descriptors all share ONE
definition of "a unit" -- `mar_engine.former_coordination`. These tests exist to keep that
definition GENERAL: covalent-radii bond detection plus the coordination number actually observed,
never a per-compound table. A change that makes tetrahedral oxo-anions work while quietly breaking
planar, octahedral or molecular-ion units will fail here.
"""
import numpy as np
import pytest
from ase import Atoms

from demars_core._engine.mar_engine import former_coordination, rigid_unit_bonds
from demars_core.calculators import _rigid_pairs

CENTRE = np.array([10.0, 10.0, 10.0])


def _tetrahedral(c, r):
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    return [c + r * x / np.linalg.norm(x) for x in v]


def _trigonal(c, r):
    return [c + r * np.array([np.cos(t), np.sin(t), 0.0]) for t in np.radians([0, 120, 240])]


def _octahedral(c, r):
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    return [c + r * x for x in v]


def _unit(centre_el, ligand_el, geom, r, n_lig, cell=20.0):
    return Atoms(centre_el + ligand_el * n_lig,
                 positions=[CENTRE] + geom(CENTRE, r), cell=[cell] * 3, pbc=True)


# (label, centre, ligand, geometry, bond length, expected coordination)
UNITS = [
    ('SO4  tetrahedral oxo-anion', 'S', 'O', _tetrahedral, 1.47, 4),
    ('PO4  tetrahedral oxo-anion', 'P', 'O', _tetrahedral, 1.54, 4),
    ('SiO4 tetrahedral oxo-anion', 'Si', 'O', _tetrahedral, 1.62, 4),
    ('ClO4 tetrahedral oxo-anion', 'Cl', 'O', _tetrahedral, 1.44, 4),
    ('BO3  planar trigonal', 'B', 'O', _trigonal, 1.37, 3),
    ('CO3  planar trigonal', 'C', 'O', _trigonal, 1.29, 3),
    ('NO3  planar trigonal', 'N', 'O', _trigonal, 1.24, 3),
    ('MoO6 octahedral', 'Mo', 'O', _octahedral, 1.95, 6),
    ('WO6  octahedral', 'W', 'O', _octahedral, 1.95, 6),
    ('NH4  molecular cation', 'N', 'H', _tetrahedral, 1.03, 4),
    ('BH4  molecular anion', 'B', 'H', _tetrahedral, 1.24, 4),
]


@pytest.mark.parametrize('label,centre,ligand,geom,r,cn', UNITS,
                         ids=[u[0].split()[0] for u in UNITS])
def test_coordination_is_read_from_geometry(label, centre, ligand, geom, r, cn):
    """Every unit shape resolves to its true coordination with no per-compound input:
    planar 3, tetrahedral 4, octahedral 6, and molecular ions with H ligands alike."""
    at = _unit(centre, ligand, geom, r, cn)
    bonds, modal, med = former_coordination(at)
    assert modal.get(centre) == cn, f'{label}: got {modal}'
    assert len(bonds) == 1, f'{label}: expected exactly one centre, got {len(bonds)}'
    assert med[(centre, ligand)] == pytest.approx(r, abs=1e-3)


@pytest.mark.parametrize('label,centre,ligand,geom,r,cn', UNITS,
                         ids=[u[0].split()[0] for u in UNITS])
def test_every_unit_bond_is_held_rigid(label, centre, ligand, geom, r, cn):
    at = _unit(centre, ligand, geom, r, cn)
    pairs, report = rigid_unit_bonds(at)
    assert len(pairs) == cn, f'{label}: {len(pairs)} pairs for a {cn}-coordinate unit'
    assert report['modal_cn'][centre] == cn
    assert report['skipped_length_outlier'] == 0


def test_structure_with_no_discrete_unit_yields_no_constraint():
    """A rocksalt halide has no former-ligand unit: the detector must return nothing so the
    caller degrades to its ordinary free relax rather than freezing arbitrary contacts."""
    a = 5.64
    nacl = Atoms('NaCl', positions=[[0, 0, 0], [a / 2, 0, 0]], cell=[a] * 3, pbc=True).repeat(2)
    bonds, modal, _ = former_coordination(nacl)
    assert bonds == {} and modal == {}
    assert rigid_unit_bonds(nacl)[0] == []
    assert _rigid_pairs(nacl, 'auto') == []


def test_two_different_units_in_one_cell_are_both_resolved():
    """A mixed framework must not have one unit's coordination leak into the other's."""
    mix = _unit('S', 'O', _tetrahedral, 1.47, 4, cell=24.0)
    c2 = CENTRE + np.array([8.0, 0.0, 0.0])
    mix += Atoms('B' + 'O' * 3, positions=[c2] + _trigonal(c2, 1.37))
    _, modal, med = former_coordination(mix)
    assert modal['S'] == 4 and modal['B'] == 3
    assert med[('S', 'O')] == pytest.approx(1.47, abs=1e-3)
    assert med[('B', 'O')] == pytest.approx(1.37, abs=1e-3)
    assert len(rigid_unit_bonds(mix)[0]) == 7


def test_length_outlier_is_not_frozen():
    """A bond far off its element-pair median on a freshly built cell is more likely an averaging
    artifact than a real bond -- freezing a wrong length is worse than leaving it free.

    The displaced ligand is kept INSIDE the covalent bond cutoff (S-O detects out to ~2.14 A) but
    outside the median band, so this exercises the length filter and not merely bond detection."""
    at = _unit('S', 'O', _tetrahedral, 1.47, 4, cell=24.0)
    c2 = CENTRE + np.array([8.0, 0.0, 0.0])
    good = _tetrahedral(c2, 1.47)
    stretched = [c2 + (good[0] - c2) * 1.30] + good[1:]     # ~1.91 A: still bonded, clearly off-median
    at += Atoms('S' + 'O' * 4, positions=[c2] + stretched)
    _, modal, _ = former_coordination(at)
    assert modal['S'] == 4, 'the displaced ligand must still register as a bond'
    pairs, report = rigid_unit_bonds(at)
    assert report['skipped_length_outlier'] == 1
    assert len(pairs) == 7


def test_stripped_centre_is_reported_as_zero_coordination():
    """A former that lost every ligand is the worst defect; it must not vanish from the audit
    just because it has no bonds left to detect."""
    from demars_core.connectivity import audit_frame
    at = _unit('S', 'O', _tetrahedral, 1.47, 4, cell=24.0)
    at += Atoms('S', positions=[CENTRE + np.array([8.0, 0.0, 0.0])])
    summary, defects = audit_frame(at)
    assert summary['S']['expected'] == 4
    assert any(d['coordination'] == 0 for d in defects)


def test_rigid_units_routes_to_the_constrained_path():
    """The engine's default backend is batched torch-sim, which has no bond constraint -- so
    `rigid_units` must actually switch backends, or the flag would be silently inert. Checked here
    with an injected ASE calculator so no MLIP is needed."""
    from demars_core.api import _make_relax

    seen = {}

    class _Spy:
        def get_potential_energy(self):
            raise RuntimeError('not called')

    import demars_core.calculators as C
    real = C.ase_relax_batch

    def _spy(atoms_list, calc, **kw):
        seen.update(kw)
        return real(atoms_list, calc, **kw)

    C.ase_relax_batch = _spy
    try:
        relax, mode, _ = _make_relax(_Spy(), d3=False, rigid_units=True)
        relax([_unit('S', 'O', _tetrahedral, 1.47, 4)])
        assert mode == 'ase-generic'
        assert seen.get('constrain') == 'auto', 'the constraint never reached the relaxer'
        seen.clear()
        relax, _, _ = _make_relax(_Spy(), d3=False, rigid_units=False)
        relax([_unit('S', 'O', _tetrahedral, 1.47, 4)])
        assert seen.get('constrain') is None, 'default must stay unconstrained'
    finally:
        C.ase_relax_batch = real


def test_d3_with_rigid_units_is_rejected():
    """SevenNetD3Model is a torch-sim model, so dispersion cannot be combined with the ASE-only
    constraint. Better to refuse than to silently drop one of the two."""
    from demars_core.api import _make_relax
    with pytest.raises(NotImplementedError, match='d3'):
        _make_relax('sevennet', d3=True, rigid_units=True)


def test_rigid_pairs_dispatch():
    at = _unit('S', 'O', _tetrahedral, 1.47, 4)
    assert _rigid_pairs(at, None) == []
    assert len(_rigid_pairs(at, 'auto')) == 4
    assert _rigid_pairs(at, lambda a: [(0, 1)]) == [(0, 1)]
    with pytest.raises(ValueError):
        _rigid_pairs(at, 'nonsense')
