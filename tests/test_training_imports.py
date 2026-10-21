"""Import rules for the internal `torchsonn.training` package.

Each module must import on its own, in a fresh interpreter, so an import
cycle shows up as soon as it is introduced. No module may import
`torchsonn.trainer`: the dependency runs from `trainer.py` to `training`
only.
"""
import ast
import pkgutil
import subprocess
import sys
from pathlib import Path

import torchsonn.training

TRAINING_DIR = Path(torchsonn.training.__file__).parent


def _module_names() -> list[str]:
    names = ["torchsonn.training"]
    names += [f"torchsonn.training.{m.name}" for m in pkgutil.iter_modules([str(TRAINING_DIR)])]
    return names


def _imports_trainer(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == "torchsonn.trainer" or a.name.startswith("torchsonn.trainer.") for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "torchsonn.trainer" or node.module.startswith("torchsonn.trainer."):
                return True
            if node.module == "torchsonn" and any(a.name in ("trainer", "Trainer") for a in node.names):
                return True
    return False


def test_each_module_imports_alone():
    """Import every module first in its own clean state: purge `torchsonn`
    from `sys.modules` before each one, in a child interpreter so the test
    process keeps its own module objects."""
    script = (
        "import importlib, sys\n"
        "failed = []\n"
        f"for name in {_module_names()!r}:\n"
        "    for key in [k for k in sys.modules if k == 'torchsonn' or k.startswith('torchsonn.')]:\n"
        "        del sys.modules[key]\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        failed.append(f'{name}: {type(exc).__name__}: {exc}')\n"
        "print('\\n'.join(failed))\n"
        "sys.exit(1 if failed else 0)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_module_imports_trainer():
    offenders = [
        path.name for path in sorted(TRAINING_DIR.glob("*.py"))
        if _imports_trainer(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert offenders == []
