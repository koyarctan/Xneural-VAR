import numpy as np

from xneural_var import construct_lagged_dataset

from experiments.run_all_benchmarks import (
    average_precision,
    benjamini_hochberg,
    fit_linear_var_with_tests,
    path_survival_score,
    simulate_linear_var,
)


def test_average_precision_groups_equal_scores() -> None:
    labels = np.array([1, 0, 1, 0])
    scores = np.array([1.0, 1.0, 0.0, 0.0])

    assert average_precision(scores, labels) == 0.5


def test_benjamini_hochberg_uses_only_the_requested_hypotheses() -> None:
    pvalues = np.array(
        [
            [0.0, 0.001, 0.80],
            [0.02, 0.0, 0.60],
            [0.90, 0.70, 0.0],
        ]
    )
    offdiag = ~np.eye(3, dtype=bool)

    selected = benjamini_hochberg(pvalues, q=0.05, mask=offdiag)

    expected = np.zeros((3, 3), dtype=bool)
    expected[0, 1] = True
    assert np.array_equal(selected, expected)


def test_path_survival_score_uses_exact_nonzero_graphs() -> None:
    rows = [
        {"lambda": 0.1, "graph": [[1, 1], [1, 0]]},
        {"lambda": 0.2, "graph": [[1, 0], [1, 0]]},
        {"lambda": 0.4, "graph": [[0, 0], [1, 0]]},
    ]

    score, violations = path_survival_score(rows)

    np.testing.assert_allclose(score, [[0.2, 0.1], [0.4, 0.0]])
    assert violations == 0


def test_linear_var_f_test_detects_a_known_grouped_lag_effect() -> None:
    coeffs = np.zeros((1, 3, 3), dtype=np.float64)
    coeffs[0, np.arange(3), np.arange(3)] = 0.25
    coeffs[0, 1, 0] = 0.65
    data = simulate_linear_var(
        coeffs=coeffs,
        T=1000,
        burn_in=300,
        noise_sd=0.5,
        innovation_correlation=0.0,
        seed=123,
    )

    result = fit_linear_var_with_tests(data, order=1, fdr_level=0.05)

    assert result["pvalues"][1, 0] < 1e-20
    assert result["graph"][1, 0] == 1
    assert not np.diag(result["graph"]).any()
    np.testing.assert_allclose(result["coeffs_actual_order"][0, 1, 0], 0.65, atol=0.08)

    dataset = construct_lagged_dataset(data, order=1)
    design = np.column_stack([np.ones(dataset.responses.shape[0]), dataset.predictors.reshape(dataset.responses.shape[0], -1)])
    full_beta = np.linalg.lstsq(design, dataset.responses[:, 1], rcond=None)[0]
    full_rss = np.sum((dataset.responses[:, 1] - design @ full_beta) ** 2)
    restricted_design = np.delete(design, 1, axis=1)
    restricted_beta = np.linalg.lstsq(restricted_design, dataset.responses[:, 1], rcond=None)[0]
    restricted_rss = np.sum((dataset.responses[:, 1] - restricted_design @ restricted_beta) ** 2)
    direct_f = (restricted_rss - full_rss) / (full_rss / (design.shape[0] - design.shape[1]))
    np.testing.assert_allclose(result["f_statistics"][1, 0], direct_f, rtol=1e-4)
