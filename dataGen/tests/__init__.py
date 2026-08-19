"""Test suite for PRISM dataGen.

Every ``test_*.py`` here is a STANDALONE runner: it inserts
``$PRISM_ROOT/dataGen`` on ``sys.path`` itself and exits 0 on success, 1 on
failure. They are executed as scripts by ``python main.py selftest`` rather
than collected by a framework, because the project may not import pytest
(the environment is restricted to numpy + the standard library).

This file exists so that ``tests`` is a package like every other directory in
the layout; it deliberately imports nothing, so that adding it cannot change
how the standalone runners resolve their own imports.
"""
