"""Prevent manual prose wrapping from becoming visible line breaks in Markdown viewers.

Parsed paragraph boundaries matter: indentation alone cannot distinguish nested-list prose from
code, and joining every nonblank block would flatten tables, equations and nested lists.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from markdown_it import MarkdownIt

REPO_ROOT = Path(__file__).resolve().parents[1]
# Include every tracked or new, non-ignored Markdown file, even outside docs/. Do not scan
# .venv, downloaded data, build output or other ignored third-party documents.
MARKDOWN_FILES = tuple(
    sorted(
        {
            REPO_ROOT / name
            for name in subprocess.check_output(
                [
                    "git",
                    "ls-files",
                    "-z",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "--",
                    "*.md",
                ],
                cwd=REPO_ROOT,
                text=True,
            ).split("\0")
            if name
        }
    )
)


def _wrapped_prose_paragraphs(text: str) -> list[tuple[int, int]]:
    """Return zero-based source ranges containing unintended Markdown soft breaks."""
    tokens = MarkdownIt("commonmark").enable("table").parse(text)
    wrapped = []
    for index, token in enumerate(tokens):
        if token.type != "paragraph_open" or token.map is None:
            continue
        inline = tokens[index + 1]
        content = inline.content.strip()
        # CommonMark does not parse display math; its internal line layout is intentional.
        if (content.startswith("$$") and content.endswith("$$")) or (
            content.startswith(r"\[") and content.endswith(r"\]")
        ):
            continue
        children = inline.children or []
        # Separate image/badge source lines are not wrapped prose.
        if children and all(
            child.type in {"image", "softbreak", "hardbreak"} for child in children
        ):
            continue
        if any(
            child.type == "softbreak"
            and not (
                position > 0
                and children[position - 1].type == "html_inline"
                and re.fullmatch(r"<br\s*/?>", children[position - 1].content, re.IGNORECASE)
            )
            for position, child in enumerate(children)
        ):
            wrapped.append(tuple(token.map))
    return wrapped


@pytest.mark.parametrize("path", MARKDOWN_FILES, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_markdown_prose_has_no_manual_hard_wrapping(path: Path) -> None:
    """Keep paragraphs and list-item paragraphs on single source lines (AGENTS.md)."""
    wrapped = _wrapped_prose_paragraphs(path.read_text())
    starts = ", ".join(str(start + 1) for start, _ in wrapped)
    assert not wrapped, (
        f"{path.relative_to(REPO_ROOT)}: manually wrapped prose starting at lines {starts}; "
        "join each paragraph onto one source line and let the viewer soft-wrap it. "
        "Preserve code, tables, math, nested lists and intentional Markdown hard breaks."
    )


@pytest.mark.parametrize(
    "text",
    [
        "First part\ncontinues here.\n",
        "- List item\n  continues here.\n",
        "1. Ordered item\n   continues here.\n",
        "- Parent\n  - Child item\n    continues here.\n",
        "> Quoted prose\n> continues here.\n",
        "> - Quoted item\n>   continues here.\n",
        "- [ ] Task item\n      continues here.\n",
        "![Image](image.png)\nThis paragraph also has prose.\n",
        "Intentional break  \nthen an accidental\ncontinuation.\n",
        "An escaped backslash\\\\\nneeds normal wrapping.\n",
    ],
)
def test_prose_wrap_guard_detects_continuations(text: str) -> None:
    """Catch ordinary, quoted, ordered, nested and task-list continuations."""
    assert _wrapped_prose_paragraphs(text)


@pytest.mark.parametrize(
    "text",
    [
        "One naturally wrapping paragraph.\n",
        "First paragraph.\n\nSecond paragraph.\n",
        "- First item\n- Second item\n",
        "- Parent\n  - Child\n  - Another child\n",
        "1. Parent\n\n   Another paragraph.\n\n   - Child\n",
        "> First paragraph.\n>\n> Second paragraph.\n",
        "```text\nmultiple\ncode lines\n```\n",
        "~~~text\nmultiple\ncode lines\n~~~\n",
        "    indented code\n    stays multiline\n",
        "| Name | Value |\n|---|---|\n| one | two |\n",
        '[source]: https://example.com\n  "A multiline reference definition"\n',
        "$$first line\n\\qquad\nsecond line$$\n",
        "\\[first line\nsecond line\\]\n",
        "<p>Raw HTML\nretains its layout.</p>\n",
        "![First](one.png)\n![Second](two.png)\n",
        "Intentional Markdown break  \ncontinues here.\n",
        "Intentional Markdown break\\\ncontinues here.\n",
        "Intentional HTML break<br>\ncontinues here.\n",
    ],
)
def test_prose_wrap_guard_preserves_structural_breaks(text: str) -> None:
    """Do not confuse formatting syntax or explicit hard breaks with arbitrary prose wrapping."""
    assert not _wrapped_prose_paragraphs(text)
