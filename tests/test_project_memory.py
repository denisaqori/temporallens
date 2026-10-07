"""Structural invariants on STATUS.md and DECISIONS.md.

These exist because the same three bookkeeping errors kept recurring by hand: a blank line
silently ending a Markdown table, a `blocked_on` entry naming a field that does not exist, and a
decision recorded as settled in one place while another still lists it as open. Each is invisible
on a casual read and each misleads the next reader, so they are checked mechanically rather than
remembered.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS = REPO_ROOT / "docs" / "project" / "STATUS.md"
DECISIONS = REPO_ROOT / "docs" / "project" / "DECISIONS.md"

#: Pending rows deliberately absent from STATUS's "Still open" list: repo housekeeping rather
#: than protocol choices that gate an experiment.
HOUSEKEEPING = {
    "github issues",
    "changelog",
    "whether the",
    "per class metrics",  # non-blocking nicety, not an owner decision gating an experiment
}


def _table_blocks(text: str) -> list[list[str]]:
    """Consecutive runs of pipe-prefixed lines — one block per rendered table."""
    blocks, current = [], []
    for line in text.split("\n"):
        if line.startswith("|"):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _titles(rows: list[str]) -> set[str]:
    """First three significant words of each row's leading bolded title, lowercased.

    This imposes a naming constraint worth knowing about: two rows must differ within their
    first three significant words, and a STATUS bullet must open with the same three as its
    Pending row. "G3 adaptation objective" and "G3 adaptation objective candidate grid" collide,
    which is why the latter is named "G3 adaptation candidate grid" instead.
    """
    out = set()
    for row in rows:
        first = row.strip().strip("|").split("|")[0]
        bold = re.search(r"\*\*(.+?)\*\*", first)
        label = bold.group(1) if bold else first
        words = re.findall(r"[a-z0-9]+", label.lower())
        if words:
            out.add(" ".join(words[:3]))
    return out


def _section(text: str, start: str, end: str | None) -> str:
    _, _, rest = text.partition(start)
    assert rest, f"missing section heading {start!r}"
    return rest.partition(end)[0] if end else rest


# --- the rendering bug ---------------------------------------------------------------------


@pytest.mark.parametrize("path", (STATUS, DECISIONS), ids=lambda p: p.name)
def test_no_blank_line_splits_a_markdown_table(path: Path) -> None:
    """A blank line ends a table, so later rows render as a separate headerless one.

    The failure is invisible in a diff: every row is present and correctly formatted, and only
    the rendered output is wrong.
    """
    for block in _table_blocks(path.read_text()):
        if len(block) < 3:
            continue
        header = block[0]
        # A real table's second line is the delimiter. A block whose first row is a data row
        # means an earlier blank line orphaned it from its header.
        assert re.match(r"^\|[\s:|-]+\|$", block[1]), (
            f"{path.name}: a table block starts without a delimiter row, so a blank line above "
            f"it split the table. First row: {header[:90]}"
        )


@pytest.mark.parametrize("path", (STATUS, DECISIONS), ids=lambda p: p.name)
def test_markdown_table_rows_keep_the_header_column_count(path: Path) -> None:
    """Catch any two table rows concatenated onto one physical line.

    Counting unescaped pipes applies to every rendered table in both files, rather than only
    decision IDs beginning D1/D2/D3. A concatenated row necessarily carries a second row's full set
    of delimiters and therefore cannot match its table header.
    """

    def unescaped_pipe_count(line: str) -> int:
        return len(re.findall(r"(?<!\\)\|", line))

    for block in _table_blocks(path.read_text()):
        if len(block) < 2 or not re.match(r"^\|[\s:|-]+\|$", block[1]):
            continue
        expected = unescaped_pipe_count(block[0])
        for row in block[1:]:
            assert unescaped_pipe_count(row) == expected, (
                f"{path.name}: table row has a different column count from its header; "
                f"it may contain two concatenated rows: {row[:110]}"
            )


# --- fail-closed configs must fail closed on something real --------------------------------


def test_every_blocked_on_entry_names_a_field_that_exists() -> None:
    """`blocked_on` is a promise that a named field is unresolved; a typo makes it a no-op."""
    missing: dict[str, list[str]] = {}
    for path in sorted((REPO_ROOT / "configs" / "experiment").rglob("*.yaml")):
        config = yaml.safe_load(path.read_text()) or {}
        for dotted in (config.get("protocol") or {}).get("blocked_on") or []:
            node: object = config
            for part in dotted.split("."):
                # Presence, not truthiness: `field: null` IS the fail-closed pattern, so a
                # value check would reject exactly the configs that are correctly blocked.
                if not isinstance(node, dict) or part not in node:
                    missing.setdefault(path.name, []).append(dotted)
                    break
                node = node[part]
    assert not missing, f"blocked_on entries that resolve to nothing: {missing}"


# --- a decision is open or settled, never both ---------------------------------------------


def test_pending_and_resolved_tables_do_not_share_a_topic() -> None:
    text = DECISIONS.read_text()
    pending = _titles(_table_blocks(_section(text, "## Pending", "## Resolved proposals"))[0][2:])
    resolved = _titles(_table_blocks(_section(text, "## Resolved proposals", None))[0][2:])
    both = pending & resolved
    assert not both, f"topics listed as open and resolved at once: {sorted(both)}"


def test_status_still_open_list_matches_the_pending_table() -> None:
    """STATUS drifts by omission: a decision lands and its 'Still open' bullet is left behind."""
    decisions, status = DECISIONS.read_text(), STATUS.read_text()
    pending = _titles(
        _table_blocks(_section(decisions, "## Pending", "## Resolved proposals"))[0][2:]
    )
    pending = {t for t in pending if not any(t.startswith(h) for h in HOUSEKEEPING)}

    bullets = re.findall(r"^\s*- \*\*(.+?)\*\*", _section(status, "**Still open**", "\n2."), re.M)
    listed = _titles([f"| **{b}** |" for b in bullets])

    # Symmetric on purpose. Drift happens in both directions: a decision lands and its STATUS
    # bullet is left behind, or a new Pending row appears and STATUS never hears about it.
    # Checking one direction only caught the first and missed the second.
    stale, unlisted = listed - pending, pending - listed
    assert (
        not stale
    ), f"STATUS lists as still open what DECISIONS no longer has Pending: {sorted(stale)}"
    assert (
        not unlisted
    ), f"DECISIONS has Pending rows STATUS never lists as open: {sorted(unlisted)}"
