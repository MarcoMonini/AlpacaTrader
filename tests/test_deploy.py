"""The page must stand up without scipy, and nothing may reach it by accident.

`pandas.Series.corr(method="spearman")` imports scipy lazily, from inside `pandas.core.nanops`, so
it passes every import-time check and every test run in a venv that has one — and raises on the
host that serves the page. It took the previous project's page down in production, on a caption.

Two halves, and both are needed. This bans the call by an AST scan everywhere but `metrics`, which
has to make it to prove its replacement equal; and the runtime dependencies simply do not carry
scipy, which the image build would otherwise hide.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parents[1] / "src"
ALLOWED = {"metrics.py"}


def spearman_calls(tree: ast.AST) -> list[int]:
    """Lines calling `.corr(method="spearman")` — the one call that imports scipy behind pandas."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "corr":
            for kw in node.keywords:
                if kw.arg == "method" and isinstance(kw.value, ast.Constant) and kw.value.value == "spearman":
                    lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: p.name)
def test_no_lazy_scipy_import(path: Path) -> None:
    found = spearman_calls(ast.parse(path.read_text()))
    if path.name in ALLOWED:
        return
    assert not found, f"{path.name}:{found} uses corr(method='spearman') — call metrics.spearman instead"


def test_the_ban_would_catch_it() -> None:
    """The scan has to find the thing it bans, or it is a test that always passes."""
    assert spearman_calls(ast.parse('a.corr(b, method="spearman")')) == [1]
    assert spearman_calls(ast.parse('a.corr(b, method="pearson")')) == []
    assert spearman_calls(ast.parse("a.corr(b)")) == []


def test_metrics_makes_the_call_it_replaces() -> None:
    """`metrics` proves the equality, so it is the one file that must contain the call."""
    text = (Path(__file__).parents[1] / "src/alpacatrader/metrics.py").read_text()
    assert 'method="spearman"' in text, "without it, nothing asserts the replacement is equal"
