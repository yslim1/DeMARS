"""Input normalisation: any structure representation -> CIF text.

The engine consumes CIF *text* (via the get_cif_text seam), so every accepted
input form is funnelled to a CIF string here. Partial occupancies and oxidation
states survive the round-trip through pymatgen's CifWriter, which is exactly what
a disordered input needs.
"""
import os

__all__ = ['to_cif_text']


def _looks_like_cif(s):
    return isinstance(s, str) and ('_cell_length_a' in s or '_atom_site_' in s
                                   or s.lstrip().startswith('data_'))


def _struct_to_cif(structure):
    from pymatgen.io.cif import CifWriter
    # write_magmoms off; keep occupancies + oxidation states as-is
    return str(CifWriter(structure))


def to_cif_text(source):
    """Accept a CIF path, a POSCAR/xyz path, raw CIF text, a pymatgen Structure,
    or an ase.Atoms, and return CIF text with disorder preserved."""
    # 1) raw CIF text
    if _looks_like_cif(source):
        return source

    # 2) a filesystem path
    if isinstance(source, str) and os.path.exists(source):
        if source.lower().endswith('.cif'):
            with open(source, encoding='utf-8') as fh:
                return fh.read()
        from pymatgen.core import Structure           # POSCAR / xyz / cssr / ...
        return _struct_to_cif(Structure.from_file(source))

    # 3) a pymatgen Structure (may carry partial occupancies)
    try:
        from pymatgen.core import Structure as _PS
        if isinstance(source, _PS):
            return _struct_to_cif(source)
    except Exception:
        pass

    # 4) an ase.Atoms (ordered only -- ASE cannot hold partial occupancies)
    try:
        from ase import Atoms
        if isinstance(source, Atoms):
            from pymatgen.io.ase import AseAtomsAdaptor
            return _struct_to_cif(AseAtomsAdaptor.get_structure(source))
    except Exception:
        pass

    raise TypeError(
        'deaverage() source must be a CIF path, structure-file path, CIF text, '
        f'pymatgen Structure, or ase.Atoms -- got {type(source).__name__}')
