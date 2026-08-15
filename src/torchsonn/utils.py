import logging
import math
import time
from argparse import Namespace
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence

# Local logger — don't pull one from torchsonn.trainer (that would create a cycle:
# trainer imports timed_block from here, and utils-importing-trainer-at-module-
# load would re-enter trainer mid-initialization).
logger = logging.getLogger(__name__)


class ParamNamespace(Namespace):

    def get(self, name: str, def_value: Any = None) -> Any:
        return getattr(self, name, def_value)

    def set(self, name: str, value: Any) -> None:
        self.__dict__[name] = value

    def __getitem__(self, item: str) -> Any:
        return self.__dict__[item]

    def __setitem__(self, item: str, value: Any) -> None:
        self.__dict__[item] = value

    def pop(self, item: str) -> Any:
        if self.__dict__.get(item) is not None:
            return self.__dict__.pop(item)
        else:
            return None

    @property
    def dict(self) -> "dict[str, Any]":
        return self.__dict__

    @classmethod
    def from_dict(cls, d: "dict[str, Any]") -> "ParamNamespace":
        params = ParamNamespace(**d)
        for k, v in params.dict.items():
            if isinstance(v, dict):
                params.dict[k] = ParamNamespace(**v)
        return params


@contextmanager
def timed_block(name: str | None = None, verbose: bool = True) -> Iterator[None]:
    t0 = time.time()
    yield
    t1 = time.time()
    if verbose:
        label = f"{name}" if name else ""
        logger.info(f"Executed {label} in {t1 - t0:0.2f} sec")


def fmt_err(value: float, decimals: int = 3) -> str:
    """Format an error value, dropping to scientific notation when fixed-point
    would round it away.

    `f"{3.2e-4:.3f}"` renders as "0.000", which hides the difference between a
    small error and no error at all — under `error_normalization: energy` on an
    un-centered target the layer-error log reads "0.000" at every layer. Values
    that survive the rounding keep the more readable fixed-point form, so the
    common case is unchanged.
    """
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return f"{value:.{decimals}e}"
    # `.Nf` rounds to all zeros below half an ulp of the last printed digit.
    # Exact zero is genuinely zero, so it keeps the fixed-point form.
    if value == 0.0 or abs(value) >= 0.5 * 10 ** (-decimals):
        return f"{value:.{decimals}f}"
    return f"{value:.{decimals}e}"


def abbrev_floats(
    values: Sequence[float],
    edgeitems: int = 3,
    threshold: int = 8,
    fmt: "str | Callable[[float], str]" = "{:.3f}",
) -> str:
    """PyTorch-style abbreviated list-of-floats formatter.

    With <= `threshold` items it prints them all. Past that, it shows
    `edgeitems` from each end with `...` in the middle, matching how
    `torch.set_printoptions(edgeitems=3, threshold=N)` renders 1-D tensors
    (without the `tensor(...)` prefix).

    `fmt` is either a format string applied as `fmt.format(v)` or a callable
    taking the value and returning its rendering (e.g. `fmt_err`).
    """
    render = fmt if callable(fmt) else fmt.format
    if len(values) <= threshold:
        return "[" + ", ".join(render(v) for v in values) + "]"
    head = ", ".join(render(v) for v in values[:edgeitems])
    tail = ", ".join(render(v) for v in values[-edgeitems:])
    return f"[{head}, ..., {tail}]"
