"""The API reference must describe the package that actually exists.

A reference page is worth having only if it cannot quietly fall out of step. These checks are
deliberately coarse -- they verify *coverage*, not prose -- because the page's job is to carry
the reasoning that cannot be generated, while signatures stay in the docstrings where they cannot
drift from the code.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "src" / "temporallens"
API_DOC = REPO_ROOT / "docs" / "api" / "README.md"


def _modules() -> list[str]:
    """Dotted names of every non-private module in the package."""
    out = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue
        out.append(str(path.relative_to(PACKAGE).with_suffix("")).replace("/", "."))
    return out


def test_the_api_reference_exists() -> None:
    assert API_DOC.is_file(), "docs/api/README.md is referenced from the package docs"


@pytest.mark.parametrize("module", _modules())
def test_every_module_is_described(module: str) -> None:
    """A module nobody documented is one nobody has to explain the purpose of."""
    assert module in API_DOC.read_text(), (
        f"`{module}` exists but docs/api/README.md does not mention it. Add a short entry "
        "saying why it exists and which decision it implements -- not its signatures."
    )


def test_nothing_documented_has_been_deleted() -> None:
    """The other drift direction: a module removed while its entry lingers."""
    text = API_DOC.read_text()
    documented = {
        line.removeprefix("### `").split("`")[0]
        for line in text.splitlines()
        if line.startswith("### `")
    }
    existing = set(_modules())
    stale = {d for d in documented if d not in existing}
    assert not stale, f"documented but no longer present: {sorted(stale)}"


@pytest.mark.parametrize("module", _modules())
def test_every_module_has_a_docstring(module: str) -> None:
    """Signatures live in the code, so the module docstring is the real documentation."""
    path = PACKAGE / (module.replace(".", "/") + ".py")
    tree = ast.parse(path.read_text())
    assert ast.get_docstring(tree), f"{module} has no module docstring"


@pytest.mark.parametrize("module", _modules())
def test_every_public_callable_has_a_docstring(module: str) -> None:
    path = PACKAGE / (module.replace(".", "/") + ".py")
    tree = ast.parse(path.read_text())
    undocumented = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and not node.name.startswith("_")
        and not ast.get_docstring(node)
    ]
    assert not undocumented, f"{module}: undocumented public names {undocumented}"
