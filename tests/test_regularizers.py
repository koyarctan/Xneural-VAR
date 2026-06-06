import pytest

torch = pytest.importorskip("torch")

from xneural_var.regularizers import (
    group_lasso_penalty,
    hierarchical_group_lasso_penalty,
    prox_group_lasso_,
)


def test_group_lasso_penalty_groups_over_lags():
    gate = torch.zeros(2, 2, 2)
    gate[:, 0, 1] = torch.tensor([3.0, 4.0])

    assert group_lasso_penalty(gate).item() == pytest.approx(5.0)


def test_group_lasso_prox_sets_whole_group_to_exact_zero():
    gate = torch.zeros(2, 1, 1)
    gate[:, 0, 0] = torch.tensor([3.0, 4.0])

    prox_group_lasso_(gate, lam=10.0, step_size=1.0)

    assert torch.count_nonzero(gate).item() == 0


def test_group_lasso_prox_shrinks_group_norm():
    gate = torch.zeros(2, 1, 1)
    gate[:, 0, 0] = torch.tensor([3.0, 4.0])

    prox_group_lasso_(gate, lam=1.0, step_size=1.0)

    assert gate[:, 0, 0].tolist() == pytest.approx([2.4, 3.2])


def test_hierarchical_penalty_matches_nested_prefixes():
    gate = torch.ones(3, 1, 1)

    penalty = hierarchical_group_lasso_penalty(gate)

    assert penalty.item() == pytest.approx(1.0 + 2.0**0.5 + 3.0**0.5)


def test_hierarchical_prox_uses_oldest_lag_prefixes():
    gate = torch.zeros(3, 1, 1)
    gate[:, 0, 0] = torch.tensor([1.0, 1.0, 100.0])

    from xneural_var.regularizers import prox_hierarchical_group_lasso_

    prox_hierarchical_group_lasso_(gate, lam=2.0, step_size=1.0)

    assert gate[0, 0, 0].item() == 0.0
    assert gate[1, 0, 0].item() == 0.0
    assert gate[2, 0, 0].item() > 0.0
