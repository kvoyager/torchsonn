"""Tests for `torchsonn.training.selection`: the survivor selection
algorithms on small synthetic candidate pools, no trained layer needed.

Candidates are rows of `z_val` built from three orthogonal, zero mean
directions over six dev samples, so which candidate each method keeps is
known exactly.
"""
from types import SimpleNamespace

import pytest
import torch

from torchsonn.training.selection import (
    SELECTION_METHODS,
    TARGET_AWARE_METHODS,
    select_candidates,
    selection_target,
)

A = torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
B = torch.tensor([0.0, 0.0, 1.0, -1.0, 0.0, 0.0])
C = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, -1.0])


def _picks(method, err, z_val, nbest, target=None, **kw) -> list[int]:
    return select_candidates(method, torch.tensor(err), torch.stack(z_val), nbest, target, **kw).tolist()


def test_registry_lists_the_config_methods():
    assert set(SELECTION_METHODS) == {"plain", "greedy_orth", "max_volume", "omp", "omp_mixed"}
    assert TARGET_AWARE_METHODS == {"omp", "omp_mixed"}


def test_plain_takes_lowest_errors_in_order():
    assert _picks("plain", [0.3, 0.1, 0.2, 0.5], [A, B, C, A + B], 3) == [1, 2, 0]


def test_greedy_orth_skips_a_collinear_candidate():
    # Candidate 1 is candidate 0 scaled: nothing novel survives orthogonalization.
    assert _picks("greedy_orth", [0.1, 0.2, 0.3], [A, 2 * A, B], 2) == [0, 2]


def test_greedy_orth_walks_in_error_order():
    assert _picks("greedy_orth", [0.3, 0.2, 0.1], [A, B, C], 3) == [2, 1, 0]


def test_max_volume_ignores_error_and_skips_collinear():
    # Largest norm first (3B), then the duplicate of B has no residual left.
    assert _picks("max_volume", [0.1, 0.2, 0.3], [A, 3 * B, B], 3) == [1, 0]


def test_max_volume_returns_fewer_than_nbest_on_constant_candidates():
    flat = torch.ones(6)
    assert _picks("max_volume", [0.1, 0.2], [flat, 2 * flat], 2) == []


def test_omp_picks_the_target_and_stops_on_noise():
    # The target is B; once B is projected out nothing explains the rest.
    target = B.view(-1, 1)
    assert _picks("omp", [0.1, 0.2, 0.3], [A, B, C], 3, target) == [1]


def test_omp_mixed_fills_the_reserve_with_novel_candidates():
    # One OMP slot finds B; the reserve then takes the largest novel
    # direction (2C) and skips the near copy of B.
    target = B.view(-1, 1)
    picks = _picks("omp_mixed", [0.1, 0.2, 0.3, 0.4], [A, B, 2 * C, B + 1e-3 * A], 3, target,
                   reserve_frac=2 / 3, orth_threshold=0.05)
    assert picks == [1, 2, 0]


def test_target_aware_method_needs_the_target():
    with pytest.raises(ValueError, match="dev targets"):
        _picks("omp", [0.1, 0.2], [A, B], 1)


def test_unknown_method_lists_the_choices():
    with pytest.raises(NotImplementedError, match="'plain', 'greedy_orth', 'max_volume', 'omp', 'omp_mixed'"):
        _picks("nope", [0.1], [A], 1)


def _model(kind: str, num_classes: int = 1):
    return SimpleNamespace(param=SimpleNamespace(model=SimpleNamespace(type=kind, num_classes=num_classes)))


def test_selection_target_one_hot_for_multi_class():
    y = torch.tensor([0, 2, 1])
    out = selection_target(_model("multi-class", 3), y, torch.float64)
    assert out.dtype == torch.float64
    assert torch.equal(out, torch.eye(3, dtype=torch.float64)[[0, 2, 1]])


def test_selection_target_column_for_regression():
    out = selection_target(_model("regressor"), torch.tensor([1.0, 2.0]), torch.float32)
    assert out.shape == (2, 1)
