"""The README renders on PyPI as well as on GitHub.

PyPI shows README.md as the project description. A relative link or image
there resolves under pypi.org and leads nowhere, a Mermaid block shows as
its source, and PyPI loads MathJax, which reads `$` as a math delimiter.
GitHub handles all three, so a README that looks right on GitHub can still
break on PyPI; these tests catch that before a release is built.
"""
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


def _prose(text: str) -> str:
    """The README with fenced code blocks and inline code spans removed."""
    text = re.sub(r"^```.*?^```[ \t]*$", "", text, flags=re.M | re.S)
    return re.sub(r"`[^`\n]*`", "", text)


def _targets(text: str) -> list[str]:
    """Every link and image target: Markdown inline, reference and HTML."""
    prose = _prose(text)
    inline = re.findall(r"\]\(\s*<?([^)\s>]+)", prose)
    reference = re.findall(r"^\s*\[[^\]]+\]:\s*<?(\S+?)>?(?:\s|$)", prose, flags=re.M)
    html = re.findall(r"""\b(?:href|src)\s*=\s*["']([^"']+)["']""", prose)
    return inline + reference + html


def _relative(text: str) -> list[str]:
    return [target for target in _targets(text) if not target.startswith("https://")]


def _has_mermaid(text: str) -> bool:
    return re.search(r"^```\s*mermaid\b", text, flags=re.M) is not None


def _dollars(text: str) -> list[str]:
    return [line.strip() for line in _prose(text).splitlines() if "$" in line]


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def test_links_and_images_are_absolute(readme):
    relative = _relative(readme)
    assert not relative, f"README.md has targets that break on PyPI (use https:// URLs): {relative}"


def test_no_mermaid(readme):
    assert not _has_mermaid(readme), \
        "README.md has a Mermaid block, which PyPI shows as source; use an image (see img/growth.mmd)"


def test_no_dollar_outside_code(readme):
    lines = _dollars(readme)
    assert not lines, f"README.md has '$' outside code, which PyPI's MathJax reads as math: {lines}"


def test_checks_catch_what_they_guard():
    sample = "\n".join([
        "[docs](docs/page.md) and [site](https://example.org/)",
        "![img](img/a.svg)",
        "[ref]: LICENSE",
        '<a href="img/b.svg"><img src="img/b.svg"></a>',
        "Costs $5.",
        "Inline `$HOME` and [`code`](https://example.org/code) are fine.",
        "```mermaid",
        "flowchart TB",
        "```",
        "```bash",
        'echo "$PATH" [x](not/a/link.md)',
        "```",
    ])
    assert _relative(sample) == ["docs/page.md", "img/a.svg", "LICENSE", "img/b.svg", "img/b.svg"]
    assert _has_mermaid(sample)
    assert _dollars(sample) == ["Costs $5."]
