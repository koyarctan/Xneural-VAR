import numpy as np

from xneural_var.data import construct_lagged_dataset


def test_construct_lagged_dataset_uses_oldest_to_newest_lag_order():
    data = np.arange(10, dtype=np.float32).reshape(5, 2)

    dataset = construct_lagged_dataset(data, order=2)

    assert dataset.predictors.shape == (3, 2, 2)
    np.testing.assert_array_equal(dataset.predictors[0], data[0:2])
    np.testing.assert_array_equal(dataset.responses[0], data[2])
