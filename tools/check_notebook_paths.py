#!/usr/bin/env python3
r"""Fail if a notebook has a machine-specific absolute path baked into it.

Jupyter stores cell outputs inside the .ipynb, so every path a tutorial prints
is committed along with it. That publishes the OS account name (the Windows
temp dir is C:\Users\<name>\AppData\Local\Temp), where the project sits on
disk, and - when the repo lives on a mapped network drive - the file server's
hostname and share name.

The tutorial notebooks defend against this at the source: they route anything
path-shaped through a local `redact()` helper that rewrites those prefixes to
<tmp>, <repo> and ~. This script is the backstop for what the helper cannot
reach - most importantly tracebacks, which Jupyter stores with full source
paths for every frame, and which no in-notebook helper can intercept.

Usage:
    python tools/check_notebook_paths.py              # every .ipynb in the repo
    python tools/check_notebook_paths.py a.ipynb ...  # specific notebooks

Exits 1 when anything is found, so it doubles as a pre-commit hook.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Redaction placeholders (<tmp>, <repo>) and the comments documenting them
# (C:\Users\<name>\...) both use angle brackets, which are illegal in real
# Windows path names. A match containing one is therefore already redacted.
PLACEHOLDER = re.compile(r"[<>]")

# Trailing class excludes quotes/punctuation so a path at the end of a sentence
# or inside a repr doesn't swallow the delimiter.
_TAIL = r"[^\s\"',;)]*"

# The (?<![A-Za-z]) guard keeps "https://..." from matching the drive-letter
# rule, which would otherwise fire on the "s:" in "https:".
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("UNC network path", re.compile(r"\\\\[A-Za-z0-9._-]+\\" + _TAIL)),
    ("drive-letter path", re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]" + _TAIL)),
    ("POSIX home dir", re.compile(r"/(?:home|Users)/" + _TAIL)),
    ("POSIX temp dir", re.compile(r"(?<![A-Za-z0-9_])/tmp/" + _TAIL)),
    ("site-packages path", re.compile(_TAIL + r"site-packages[\\/]" + _TAIL)),
]

# Base64-encoded images live on one enormous line. Skipping those avoids
# matching path-like noise inside the payload, and they can't leak a path
# anyway since they are not text.
MAX_LINE = 2000

# MIME types worth scanning. image/png and friends are binary payloads.
TEXT_MIMES = (("text/plain", "result"), ("image/svg+xml", "svg"))


def _flatten(value: str | list[str]) -> str:
    """nbformat stores multi-line text as either a str or a list of lines."""
    return "".join(value) if isinstance(value, list) else value


def _texts(cell: dict) -> "list[tuple[str, str]]":
    """Every human-readable chunk of a cell, tagged with where it came from."""
    chunks: list[tuple[str, str]] = [("source", _flatten(cell.get("source", [])))]
    for out in cell.get("outputs", []):
        if "text" in out:
            chunks.append(("stream", _flatten(out["text"])))
        data = out.get("data", {})
        for mime, kind in TEXT_MIMES:
            if mime in data:
                chunks.append((kind, _flatten(data[mime])))
        if out.get("output_type") == "error":
            chunks.append(("traceback", "\n".join(out.get("traceback", []))))
    return chunks


def _first_leak(line: str) -> "tuple[str, str] | None":
    for label, pattern in PATTERNS:
        for match in pattern.finditer(line):
            if not PLACEHOLDER.search(match.group(0)):
                return label, match.group(0)
    return None


def scan(path: Path) -> "list[tuple[int, str, str, str, str]]":
    """Return (cell_index, kind, label, matched_path, containing_line)."""
    notebook = json.loads(path.read_bytes())
    findings = []
    for index, cell in enumerate(notebook.get("cells", [])):
        for kind, text in _texts(cell):
            for line in text.splitlines():
                if len(line) > MAX_LINE:
                    continue
                leak = _first_leak(line)
                if leak is not None:
                    findings.append((index, kind, leak[0], leak[1], line.strip()))
    return findings


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help="notebooks to check (default: every .ipynb in the repo)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    if args.notebooks:
        targets = args.notebooks
    else:
        targets = sorted(
            p for p in repo_root.rglob("*.ipynb")
            if ".ipynb_checkpoints" not in p.parts
        )

    total = 0
    for path in targets:
        findings = scan(path)
        total += len(findings)
        if not findings:
            continue
        try:
            shown = path.resolve().relative_to(repo_root)
        except ValueError:
            shown = path
        print(f"\n{shown}")
        for index, kind, label, matched, line in findings:
            print(f"  cell {index:<3} [{kind}] {label}")
            print(f"    {matched}")
            if line != matched:
                print(f"    in: {line[:120]}")

    plural = "" if len(targets) == 1 else "s"
    if total:
        print(
            f"\n{total} absolute path(s) in {len(targets)} notebook{plural}.\n"
            "Route the printing site through the notebook's redact() helper, "
            "and clear the output of any cell that raised - tracebacks embed "
            "full source paths."
        )
        return 1

    print(f"{len(targets)} notebook{plural} checked, no absolute paths stored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())