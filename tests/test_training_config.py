import numpy as np
import pytest

pytest.importorskip("torch")

from xneural_var import GVARTrainingConfig, fit_gvar_ngc


def test_sparse_group_lasso_config_rejects_lambda_ngc():
    data = np.random.default_rng(0).normal(size=(12, 2)).astype("float32")
    config = GVARTrainingConfig(
        order=2,
        hidden_layer_size=4,
        max_epochs=1,
        lambda_ngc=1e-3,
        regularizer="sparse_group_lasso",
        verbose=0,
    )

    with pytest.raises(ValueError, match="sparse_group_lasso does not use lambda_ngc"):
        fit_gvar_ngc(data, config)


def test_hierarchical_group_lasso_config_uses_lambda_ngc():
    data = np.random.default_rng(0).normal(size=(12, 2)).astype("float32")
    config = GVARTrainingConfig(
        order=2,
        hidden_layer_size=4,
        max_epochs=1,
        lambda_ngc=1e-3,
        regularizer="hierarchical_group_lasso",
        verbose=0,
    )

    result = fit_gvar_ngc(data, config)

    assert len(result.history["loss"]) == 1
