import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cfpr.config import BacktestConfig
from cfpr.statistics import analyze_predictions, hac_mean_test


def orthogonal_sample(n=40, date="2026-01-05"):
    rng = np.random.default_rng(71)
    raw = rng.normal(size=(n, 4))
    raw -= raw.mean(axis=0)
    predictors, _ = np.linalg.qr(raw)
    predictors *= math.sqrt(n)
    factor = pd.DataFrame(
        {
            "date": pd.Timestamp(date),
            "symbol": [f"S{i:03d}" for i in range(n)],
            "factor": predictors[:, 0],
            "hf_signal": predictors[:, 0] + predictors[:, 1],
            "control_momentum": predictors[:, 1],
            "control_reversal": predictors[:, 2],
            "control_volume": predictors[:, 3],
            "eligible": True,
        }
    )
    label = factor[["date", "symbol"]].copy()
    label["forward_return"] = predictors @ np.array([3.0, 1.0, 2.0, 4.0]) / 100
    return factor, label


def test_hac_matches_manual_bartlett_calculation():
    values = np.array([1.0, 2.0, 4.0, 5.0])
    residual = values - values.mean()
    expected_variance = (residual @ residual + residual[1:] @ residual[:-1]) / 16
    result = hac_mean_test(values, 1)
    assert result["mean"] == 3.0
    assert result["standard_error"] == pytest.approx(math.sqrt(expected_variance))
    assert result["t_stat"] == pytest.approx(3 / math.sqrt(expected_variance))


def test_hac_missing_dates_do_not_compress_time():
    result = hac_mean_test(np.array([1.0, np.nan, 3.0]), 1)
    assert result["n_obs"] == 2
    assert result["standard_error"] == pytest.approx(math.sqrt(2) / 2)
    compressed = hac_mean_test(np.array([1.0, 3.0]), 1)
    assert result["standard_error"] != compressed["standard_error"]


def test_hac_no_observations_or_zero_variance_does_not_invent_t_stat():
    assert hac_mean_test(np.array([np.nan]), 4)["mean"] is None
    assert hac_mean_test(np.array([2.0]), 4)["standard_error"] is None
    assert hac_mean_test(np.array([2.0, 2.0]), 4)["t_stat"] is None
    with pytest.raises(ValueError, match="nonnegative integer"):
        hac_mean_test(np.array([1.0]), -1)


def test_incremental_regression_standardizes_predictors_and_outcome():
    factor, label = orthogonal_sample()
    output = analyze_predictions(factor, label)
    row = output["regression_daily"].iloc[0]
    assert row["status"] == "ok"
    assert row["n_obs"] == len(factor)
    assert row["intercept"] == pytest.approx(0.0, abs=1e-14)
    assert row["beta_factor"] == pytest.approx(3 / math.sqrt(30))
    assert row["beta_momentum"] == pytest.approx(1 / math.sqrt(30))
    assert row["beta_reversal"] == pytest.approx(2 / math.sqrt(30))
    assert row["beta_volume"] == pytest.approx(4 / math.sqrt(30))


def test_hac_lag_floor_respects_overlapping_horizon():
    factor, label = orthogonal_sample()
    output = analyze_predictions(factor, label, BacktestConfig(horizon=5, hac_lags=0))
    assert output["summary"]["effective_hac_lags"] == 4
    assert output["summary"]["rank_ic"]["lags"] == 4
    assert output["summary"]["incremental_beta"]["lags"] == 4


def test_rank_ic_counts_eligible_zeros_and_ties_are_not_split():
    factor, label = orthogonal_sample(20)
    factor["factor"] = [0.0] * 15 + [1.0, 2.0, 3.0, 4.0, 5.0]
    label["forward_return"] = factor["factor"] / 100
    output = analyze_predictions(factor, label)
    row = output["ic_daily"].iloc[0]
    assert row["rank_ic"] == pytest.approx(1.0)
    assert row["n_obs"] == 20
    quantiles = output["quantile_returns"]
    assert quantiles["n_obs"].sum() == 20
    zero_bucket = quantiles.loc[quantiles["mean_forward_return"] == 0]
    assert len(zero_bucket) == 1
    assert zero_bucket.iloc[0]["n_obs"] == 15
    assert len(quantiles) < 5


def test_constant_signal_is_explicitly_skipped_instead_of_ranked_by_symbol():
    factor, label = orthogonal_sample()
    factor["factor"] = 0.0
    output = analyze_predictions(factor, label)
    assert output["ic_daily"].iloc[0]["status"] == "constant_signal"
    assert math.isnan(output["ic_daily"].iloc[0]["rank_ic"])
    assert output["quantile_returns"].empty
    assert output["regression_daily"].iloc[0]["status"] == "constant_predictor"
    assert output["summary"]["rank_ic"]["n_obs"] == 0
    assert output["summary"]["rank_ic"]["status_counts"] == {"constant_signal": 1}


def test_rank_deficient_controls_are_not_silently_inverted():
    factor, label = orthogonal_sample()
    factor["control_momentum"] = factor["factor"]
    output = analyze_predictions(factor, label)
    assert output["regression_daily"].iloc[0]["status"] == "rank_deficient"
    assert output["summary"]["incremental_beta"]["n_obs"] == 0


def test_unlabeled_dates_are_retained_and_min_cross_section_is_reported():
    first, first_labels = orthogonal_sample(date="2026-01-05")
    second, second_labels = orthogonal_sample(date="2026-01-06")
    second_labels["forward_return"] = np.nan
    factors = pd.concat([first, second], ignore_index=True)
    labels = pd.concat([first_labels, second_labels], ignore_index=True)
    output = analyze_predictions(factors, labels)
    assert len(output["ic_daily"]) == 2
    assert output["ic_daily"].iloc[-1]["n_obs"] == 0
    assert output["ic_daily"].iloc[-1]["status"] == "insufficient_observations"
    assert output["summary"]["rank_ic"]["n_obs"] == 1
    assert output["summary"]["n_labeled_eligible_rows"] == 40


def test_untradable_observation_and_input_order_do_not_change_results():
    factor, label = orthogonal_sample()
    factor.loc[0, "eligible"] = False
    original_factor, original_label = factor.copy(deep=True), label.copy(deep=True)
    expected = analyze_predictions(factor, label)
    actual = analyze_predictions(
        factor.sample(frac=1, random_state=1), label.sample(frac=1, random_state=2)
    )
    for name in ["ic_daily", "baseline_ic", "quantile_returns", "regression_daily"]:
        assert_frame_equal(expected[name], actual[name], atol=1e-12, rtol=1e-12)
    assert expected["ic_daily"].iloc[0]["n_obs"] == 39
    assert_frame_equal(factor, original_factor)
    assert_frame_equal(label, original_label)


def test_duplicate_labels_and_nonboolean_eligibility_are_rejected():
    factor, label = orthogonal_sample()
    with pytest.raises(ValueError, match="duplicate"):
        analyze_predictions(factor, pd.concat([label, label.iloc[:1]]))
    factor["eligible"] = "False"
    with pytest.raises(ValueError, match="boolean"):
        analyze_predictions(factor, label)


def test_empty_input_has_no_fake_passed_diagnostics():
    factor, label = orthogonal_sample()
    output = analyze_predictions(factor.iloc[:0], label.iloc[:0])
    assert output["ic_daily"].empty
    assert output["regression_daily"].empty
    assert output["summary"]["n_signal_dates"] == 0
    assert output["summary"]["rank_ic"]["n_obs"] == 0


def test_repeated_decimal_hac_has_no_spurious_enormous_t_stat():
    result = hac_mean_test(np.repeat(0.01, 10), 4)
    assert result["mean"] == pytest.approx(0.01)
    assert result["standard_error"] == 0.0
    assert result["t_stat"] is None


def test_identical_decimal_forward_returns_are_not_counted_as_valid_regressions():
    factor, label = orthogonal_sample(10)
    label["forward_return"] = 0.01
    output = analyze_predictions(factor, label)
    assert output["regression_daily"].iloc[0]["status"] == "constant_forward_return"
    assert output["summary"]["incremental_beta"]["n_obs"] == 0


def test_identical_decimal_control_is_not_mistaken_for_varying_predictor():
    factor, label = orthogonal_sample(10)
    factor["control_volume"] = 0.01
    output = analyze_predictions(factor, label)
    assert output["regression_daily"].iloc[0]["status"] == "constant_predictor"
