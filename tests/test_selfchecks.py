"""Runs every module's own self-check.

The convention inherited from the previous project: asserts live at the bottom of the module they
check, under `if __name__ == "__main__"`, next to the code and the reasoning they belong to rather
than in a mirrored test file. This is what makes CI run them — add new modules to `MODULES`.
"""

import importlib

import pytest

MODULES = ["alpacatrader.data.candles", "alpacatrader.universe"]


@pytest.mark.parametrize("name", MODULES)
def test_selfcheck(name: str) -> None:
    importlib.import_module(name)._selfcheck()
