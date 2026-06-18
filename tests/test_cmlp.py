import numpy as np
import pytest

pytest.importorskip("torch")

from xneural_var import CMLPTrainingConfig, fit_cmlp


def test_fit_cmlp_sparse_group_lasso_smoke():
    data = np.random.default_rng(0).normal(size=(20, 3)).astype("float32")
    config = CMLPTrainingConfig(
        order=2,
        hidden_layer_size=4,
        max_epochs=2,
        batch_size=8,
        learning_rate=1e-2,
        regularizer="sparse_group_lasso",
        sparse_group_lambda=1e-3,
        sparse_l1_lambda=1e-4,
        optimizer="ista",
        verbose=0,
    )

    result = fit_cmlp(data, config)

    assert len(result.history["loss"]) == 2
    assert result.causal_strength.shape == (3, 3)
    assert result.causal_graph.shape == (3, 3)


def test_cmlp_sparse_group_lasso_rejects_lambda_ngc():
    data = np.random.default_rng(1).normal(size=(12, 2)).astype("float32")
    config = CMLPTrainingConfig(
        order=2,
        hidden_layer_size=4,
        max_epochs=1,
        lambda_ngc=1e-3,
        regularizer="sparse_group_lasso",
        verbose=0,
    )

    with pytest.raises(ValueError, match="sparse_group_lasso does not use lambda_ngc"):
        fit_cmlp(data, config)
