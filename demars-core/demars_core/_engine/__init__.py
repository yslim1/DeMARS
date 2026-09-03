"""The DeMARS engine: mar_engine (supercell + sampling + relax loop),
mar_evidence (CIF-mined evidence bundle), mar_record (record assembly).

Originally vendored from a separate in-tree work tree; that upstream is gone, so
these are now the canonical sources and are edited here directly.
Relaxation goes through demars_core._torchsim (torch-sim).

`_engine` is private by name but NOT by use: the repo's `tools/` stage CLIs import
`structure_from_text`, `run_self_driving`, `evidence_from_text` and `build_record`
directly. Read the INTERNAL CONTRACT note in ../api.py before changing those signatures.
"""
