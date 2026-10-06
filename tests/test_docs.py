"""The configuration reference (docs/reference/config.md) stays in step with the code.

Every configuration key, every option a `model.ref_functions` entry accepts
and every argument an optimizer takes must appear on the page,
and every key the page names must exist. Adding, renaming or removing one
without updating the page fails here.
"""
import dataclasses
import inspect
import re
from pathlib import Path

import pytest

from torchsonn.config.schemas import SONNConfig, _default_optimizer_params
from torchsonn.neurons import BasePolynomNeuron
from torchsonn.optimizers import optimizer_map

REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "reference" / "config.md"

# Constructor arguments the model fills in itself; an entry cannot set them.
FAMILY_BUILDER_ARGS = {"self", "num_feat", "num_src_feat", "layer_index", "start_index"}
# Optimizer arguments the trainer passes itself.
OPTIMIZER_BUILDER_ARGS = {"self", "params", "shared_param_names"}


@pytest.fixture(scope="module")
def page() -> str:
    return REFERENCE.read_text(encoding="utf-8")


def _section(page: str, heading: str) -> str:
    """Text of the `## heading` section, up to the next level-2 heading."""
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", page, flags=re.M | re.S)
    assert match, f"config.md has no '## {heading}' section"
    return match.group(1)


def _config_paths() -> list[str]:
    """Dotted path of every leaf key of the schema, nested dataclasses expanded."""
    paths = []

    def walk(cls: type, prefix: str) -> None:
        for f in dataclasses.fields(cls):
            default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
            if dataclasses.is_dataclass(default):
                walk(type(default), f"{prefix}{f.name}.")
            else:
                paths.append(f"{prefix}{f.name}")
                if f.name == "shortcut" and isinstance(default, dict):
                    paths.extend(f"{prefix}shortcut.{key}" for key in default)

    walk(SONNConfig, "")
    return paths


def _families() -> list[type]:
    """Every concrete neuron family, as registered by `BasePolynomNeuron`."""
    return [cls for name, cls in BasePolynomNeuron._registry.items()
            if not inspect.isabstract(cls) and not name.startswith("Base")]


def _arguments(cls: type, exclude: set[str]) -> list[str]:
    return [name for name in inspect.signature(cls.__init__).parameters if name not in exclude]


def test_every_config_key_is_documented(page):
    missing = [path for path in _config_paths() if f"`{path}`" not in page]
    assert not missing, f"config.md does not document: {missing}"


def test_every_documented_key_exists(page):
    known = set(_config_paths())
    prefixes = {path.rsplit(".", 1)[0] for path in known if "." in path}
    named = set(re.findall(r"`((?:model|train)\.[a-z_.]+)`", page))
    unknown = sorted(key for key in named if key not in known and key not in prefixes)
    assert not unknown, f"config.md documents keys the schema does not have: {unknown}"


@pytest.mark.parametrize("family", _families(), ids=lambda cls: cls.__name__)
def test_every_family_option_is_documented(page, family):
    section = _section(page, "Neuron family options")
    missing = [name for name in _arguments(family, FAMILY_BUILDER_ARGS) if f"`{name}`" not in section]
    assert not missing, f"options of {family.__name__} missing from 'Neuron family options': {missing}"


def test_every_family_is_covered():
    names = {cls.__name__ for cls in _families()}
    assert {"LinearPolynomNeuron", "LinearCovPolynomNeuron", "QuadraticPolynomNeuron",
            "CubicPolynomNeuron", "PolyQuadratic", "LegendrePolynomNeuron",
            "ChebyshevPolynomNeuron", "RBFNeuron"} <= names


@pytest.mark.parametrize("name", sorted(optimizer_map))
def test_every_optimizer_argument_is_documented(page, name):
    section = _section(page, "Optimizer parameters")
    assert f"`{name}`" in page, f"optimizer {name!r} is not named in config.md"
    missing = [arg for arg in _arguments(optimizer_map[name], OPTIMIZER_BUILDER_ARGS)
               if f"`{arg}`" not in section]
    assert not missing, f"arguments of optimizer {name!r} missing from 'Optimizer parameters': {missing}"


def test_every_default_optimizer_param_is_documented(page):
    section = _section(page, "Optimizer parameters")
    missing = [key for key in _default_optimizer_params() if f"`{key}`" not in section]
    assert not missing, f"default optimizer_params keys missing from 'Optimizer parameters': {missing}"
