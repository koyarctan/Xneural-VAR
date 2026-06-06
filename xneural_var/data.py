from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LaggedDataset:
    predictors: np.ndarray
    responses: np.ndarray
    time_index: np.ndarray
    series_index: np.ndarray


def _as_series_list(data: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
    if isinstance(data, list):
        if not data:
            raise ValueError("data list must not be empty")
        return data
    return [data]


def construct_lagged_dataset(data: np.ndarray | list[np.ndarray], order: int) -> LaggedDataset:
    """Build lagged GVAR predictors.

    Predictors have shape ``[n_samples, order, n_vars]``. The lag axis follows
    the original GVAR repository convention: index 0 is the most distant lag
    ``t - order`` and index ``order - 1`` is the most recent lag ``t - 1``.
    """
    if order <= 0:
        raise ValueError("order must be positive")

    predictors = []
    responses = []
    time_index = []
    series_index = []
    expected_p = None

    offset = 0
    for r, series in enumerate(_as_series_list(data)):
        series = np.asarray(series, dtype=np.float32)
        if series.ndim != 2:
            raise ValueError("each time series must have shape [time, variables]")
        if series.shape[0] <= order:
            raise ValueError("each time series must be longer than order")
        if expected_p is None:
            expected_p = series.shape[1]
        elif series.shape[1] != expected_p:
            raise ValueError("all time series must have the same number of variables")

        for t in range(order, series.shape[0]):
            predictors.append(series[t - order : t])
            responses.append(series[t])
            time_index.append(offset + t)
            series_index.append(r)

        offset += series.shape[0] + order

    return LaggedDataset(
        predictors=np.stack(predictors).astype(np.float32),
        responses=np.stack(responses).astype(np.float32),
        time_index=np.asarray(time_index, dtype=np.int64),
        series_index=np.asarray(series_index, dtype=np.int64),
    )
