"""Make the stage CLIs importable as modules for the contract tests.

`tools/` is a directory of scripts, not a package -- it is invoked as `python tools/x.py`, which
puts `tools/` on sys.path automatically. Under pytest nothing does that, so import it here. A
conftest is the right place: it runs before collection, and it keeps the sys.path line out of the
tests themselves.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
