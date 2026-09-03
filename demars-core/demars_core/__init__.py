"""demars-core -- de-average a disordered crystal into a Minimal Atomistic
Representation (MAR), from any CIF, with a pluggable universal-MLIP backend.

The DeMARS engine is carried inside the package under `_engine/`, which is the canonical
source and is edited in place. See `README.md`.

    from demars_core import deaverage
    rec = deaverage("disordered.cif", calculator="sevennet", out_dir="./mar")
    print(rec.summary())

The connectivity gate -- the one check that catches what charge and fidelity cannot -- is
`demars_core.connectivity.audit_frame`. Not imported here: it pulls the engine's
coordination machinery, and most callers only want `deaverage()`.
"""
from .api import deaverage
from .record import MARRecord, write_custom_ensemble

__version__ = '0.1.0'
__all__ = ['deaverage', 'MARRecord', 'write_custom_ensemble', '__version__']
