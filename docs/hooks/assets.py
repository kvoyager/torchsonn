"""MkDocs build hook: the tutorials' diagrams and READMEs on the site.

The tutorial READMEs and their network diagrams live next to the tutorial
code, where GitHub shows them, and MkDocs builds only what is under
``docs/``. This hook brings them onto the site without copying anything
into ``docs/``:

* ``on_files`` publishes every SVG in ``img/`` and ``tutorials/<name>/`` at
  ``assets/<the same path>``, so a page can link or embed it as
  ``../assets/tutorials/<name>/<file>.svg``.
* ``on_page_markdown`` replaces a line of the form::

      --8<-- "tutorials/<name>/README.md"

  with that README, after three changes. Its first-level title is dropped,
  because the page has its own. Its relative links point at the published
  diagrams for images and at the file on GitHub for everything else, so
  they work on the site as they do on GitHub. Its ``<details>`` blocks
  become collapsible admonitions, which the site renders with Markdown
  inside.

The line is also a ``pymdownx.snippets`` include, so without the hook the
README is still included, with its links unchanged.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path

from mkdocs.structure.files import File

ASSET_ROOT = "assets"
IMAGE_SUFFIXES = (".svg", ".png", ".jpg", ".jpeg", ".gif")

_INCLUDE = re.compile(r'^--8<--\s+"(?P<path>tutorials/[^"]+/README\.md)"\s*$', re.M)
_MD_LINK = re.compile(r"(?P<head>!?\[[^\]\n]*\]\()(?P<target>[^)\s]+)(?P<tail>(?:\s+\"[^\"]*\")?\))")
_HTML_ATTR = re.compile(r'(?P<head>\b(?:src|href)=")(?P<target>[^"]+)(?P<tail>")')
_DETAILS = re.compile(
    r"^<details(?P<open>\s+open)?>\s*\n<summary>(?P<title>.*?)</summary>\s*\n(?P<body>.*?)\n</details>\s*$",
    re.M | re.S,
)


def _repo_root(config) -> Path:
    return Path(config.config_file_path).resolve().parent


def on_files(files, config):
    """Publish the diagrams under img/ and tutorials/<name>/ at assets/."""
    root = _repo_root(config)
    for pattern in ("img/*.svg", "tutorials/*/*.svg"):
        for path in sorted(root.glob(pattern)):
            src_uri = f"{ASSET_ROOT}/{path.relative_to(root).as_posix()}"
            files.append(File.generated(config, src_uri, abs_src_path=str(path)))
    return files


def _is_relative(target: str) -> bool:
    return not (target.startswith(("#", "/", "<")) or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target))


def _rewrite(text: str, readme_dir: str, page, config, *, html: bool) -> str:
    """Point the relative links of one README line at the site or GitHub."""
    blob = config.repo_url.rstrip("/") + "/blob/main/"
    page_dir = posixpath.dirname(page.file.src_uri)
    url_dir = page.url if page.url.endswith("/") else posixpath.dirname(page.url)

    def fix(match: re.Match) -> str:
        target = match.group("target")
        if not _is_relative(target):
            return match.group(0)
        path, _, fragment = target.partition("#")
        repo_path = posixpath.normpath(posixpath.join(readme_dir, path))
        if repo_path.lower().endswith(IMAGE_SUFFIXES):
            asset = f"{ASSET_ROOT}/{repo_path}"
            # Markdown links are resolved by MkDocs from the page's source
            # file; raw HTML is resolved by the browser from the page's URL.
            new = posixpath.relpath(asset, url_dir or ".") if html else posixpath.relpath(asset, page_dir or ".")
        else:
            new = blob + repo_path + (f"#{fragment}" if fragment else "")
        return match.group("head") + new + match.group("tail")

    pattern = _HTML_ATTR if html else _MD_LINK
    return pattern.sub(fix, text)


def _details_to_admonitions(text: str) -> str:
    def convert(match: re.Match) -> str:
        marker = "???+" if match.group("open") else "???"
        title = match.group("title").strip().replace('"', "'")
        body = "\n".join(("    " + line) if line.strip() else "" for line in match.group("body").split("\n"))
        return f'{marker} example "{title}"\n\n{body.strip(chr(10))}\n'

    return _DETAILS.sub(convert, text)


def _include(readme: Path, readme_dir: str, page, config) -> str:
    lines = readme.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is not None and lines[first].startswith("# "):
        del lines[first]
    out, fence = [], None
    for line in lines:
        stripped = line.lstrip()
        if fence is None and stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
        elif fence is not None and stripped.startswith(fence):
            fence = None
        elif fence is None:
            line = _rewrite(line, readme_dir, page, config, html=False)
            line = _rewrite(line, readme_dir, page, config, html=True)
        out.append(line)
    return _details_to_admonitions("\n".join(out))


def on_page_markdown(markdown, page, config, files):
    """Replace each tutorial README include with the adapted README."""
    root = _repo_root(config)

    def replace(match: re.Match) -> str:
        rel = match.group("path")
        return _include(root / rel, posixpath.dirname(rel), page, config)

    return _INCLUDE.sub(replace, markdown)
