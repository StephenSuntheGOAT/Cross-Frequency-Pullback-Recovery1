"""Independent arithmetic and information-set tests for the proposed factor."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from cfpr.data import DataValidationError
from cfpr.factor import compute_factor, pullback_kernel


def sample_data(days=55, symbols=("A", "B")):
    dates = pd.bdate_range("2024-01-02", periods=days)
    daily_rows, minute_frames = [], []
    for number, symbol in enumerate(symbols):
        daily_returns = 0.005 + 0.001 * np.sin(np.arange(days) * 0.7 + number)
        daily_returns[36:40] = [-0.005, -0.007, -0.006, -0.004]
        closes = 100 * np.cumprod(1 + daily_returns)
        for index, date in enumerate(dates):
            daily_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": closes[index] / (1 + daily_returns[index]),
                    "close": closes[index],
                    "volume": 100_000 + 1000 * index,
                }
            )
            tail_return = 0.002 + 0.004 * np.sin(index * 0.9 + number)
            if index == 40:
                tail_return = 0.01
            prices = np.linspace(closes[index] / (1 + tail_return), closes[index], 61)
            minute_frames.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.date_range(
                            date + timedelta(hours=14), periods=60, freq="min"
                        ),
                        "symbol": symbol,
                        "open": prices[:-1],
                        "close": prices[1:],
                        "volume": 100 + index,
                    }
                )
            )
    return pd.DataFrame(daily_rows), pd.concat(minute_frames, ignore_index=True), dates


def test_formula_and_default_anchors_match_independent_arithmetic():
    daily, minutes, dates = sample_data()
    scores = compute_factor(daily, minutes)
    row = scores.loc[scores.date.eq(dates[40]) & scores.symbol.eq("A")].iloc[0]
    close = daily.loc[daily.symbol.eq("A")].set_index("date").close
    sigma = close.pct_change().iloc[20:40].std(ddof=1)
    trend = close.iloc[35] / close.iloc[15] - 1
    pullback = close.iloc[39] / close.iloc[35] - 1
    tail_returns = 0.002 + 0.004 * np.sin(np.arange(20, 40) * 0.9)
    expected_m = trend / (sigma * np.sqrt(20))
    expected_d = max(-pullback, 0) / (sigma * 2)
    expected_h = 0.01 / np.std(tail_returns, ddof=1)
    expected_q = (100 + 40) / np.mean(100 + np.arange(20, 40))
    expected = (
        expected_m * expected_d * np.exp(-(expected_d**2) / 2) * expected_h * min(expected_q, 2)
    )
    assert row.eligible
    assert row.M == pytest.approx(expected_m)
    assert row.D == pytest.approx(expected_d)
    assert row.H == pytest.approx(expected_h)
    assert row.Q == pytest.approx(expected_q)
    assert row.factor == pytest.approx(expected)


def test_pullback_kernel_peaks_at_one_and_penalizes_deep_drawdown():
    values = np.linspace(0, 5, 501)
    kernel = pullback_kernel(values)
    assert values[kernel.argmax()] == pytest.approx(1)
    assert pullback_kernel(0.0) == 0
    assert pullback_kernel(3.0) < pullback_kernel(1.0) / 10


def test_negative_repair_is_valid_zero_not_missing():
    daily, minutes, dates = sample_data()
    mask = minutes.symbol.eq("A") & minutes.timestamp.dt.normalize().eq(dates[40])
    minutes.loc[mask, "close"] *= 0.98
    scores = compute_factor(daily, minutes)
    row = scores.loc[scores.date.eq(dates[40]) & scores.symbol.eq("A")].iloc[0]
    assert row.H < 0
    assert row.eligible
    assert row.factor == 0


def test_negative_prior_trend_gates_score_to_zero():
    daily, minutes, dates = sample_data()
    mask = daily.symbol.eq("A") & daily.date.eq(dates[15])
    daily.loc[mask, "close"] *= 2
    scores = compute_factor(daily, minutes)
    row = scores.loc[scores.date.eq(dates[40]) & scores.symbol.eq("A")].iloc[0]
    assert row.M < 0
    assert row.eligible
    assert row.factor == 0


def test_no_drawdown_is_valid_zero():
    daily, minutes, dates = sample_data()
    start = daily.loc[daily.symbol.eq("A") & daily.date.eq(dates[35]), "close"].iloc[0]
    daily.loc[daily.symbol.eq("A") & daily.date.eq(dates[39]), "close"] = start * 1.01
    row = compute_factor(daily, minutes).query("symbol == 'A'").set_index("date").loc[dates[40]]
    assert row.D == 0
    assert row.eligible
    assert row.factor == 0


def test_today_excluded_from_all_reference_baselines_and_volume_capped():
    daily, minutes, dates = sample_data()
    before = compute_factor(daily, minutes).query("symbol == 'A'").set_index("date").loc[dates[40]]
    daily.loc[daily.symbol.eq("A") & daily.date.eq(dates[40]), ["close", "volume"]] *= 5
    mask = minutes.symbol.eq("A") & minutes.timestamp.dt.normalize().eq(dates[40])
    minutes.loc[mask, "volume"] *= 10
    minutes.loc[mask, "close"] *= 1.1
    after = compute_factor(daily, minutes).query("symbol == 'A'").set_index("date").loc[dates[40]]
    for name in ["M", "D", "daily_vol", "tail_vol", "tail_volume_mean"]:
        assert after[name] == pytest.approx(before[name])
    assert after.Q == pytest.approx(before.Q * 10)
    assert after.volume_multiplier == 2


def test_future_data_cannot_change_prior_scores():
    daily, minutes, dates = sample_data()
    before = compute_factor(daily, minutes)
    daily.loc[daily.date.gt(dates[40]), ["open", "close", "volume"]] *= 2
    minutes.loc[minutes.timestamp.dt.normalize().gt(dates[40]), ["open", "close", "volume"]] *= 3
    after = compute_factor(daily, minutes)
    pd.testing.assert_frame_equal(
        before.loc[before.date.le(dates[40])], after.loc[after.date.le(dates[40])]
    )


def test_missing_symbol_day_does_not_compress_lookback():
    daily, minutes, dates = sample_data()
    daily = daily.loc[~(daily.symbol.eq("A") & daily.date.eq(dates[37]))]
    scores = compute_factor(daily, minutes)
    row = scores.loc[scores.date.eq(dates[40]) & scores.symbol.eq("A")].iloc[0]
    missing = scores.loc[scores.date.eq(dates[37]) & scores.symbol.eq("A")].iloc[0]
    assert not row.eligible
    assert np.isnan(row.factor)
    assert not missing.eligible
    assert missing.invalid_reason == "missing_daily"


def test_explicit_calendar_preserves_session_missing_for_every_stock():
    daily, minutes, dates = sample_data(symbols=("A",))
    daily = daily.loc[~daily.date.eq(dates[37])]
    scores = compute_factor(daily, minutes, calendar=dates)
    row = scores.set_index("date").loc[dates[40]]
    assert not row.eligible
    assert len(scores) == len(dates)


def test_incomplete_today_tail_and_past_tail_invalidate_distinctly():
    daily, minutes, dates = sample_data()
    mask = minutes.symbol.eq("A") & minutes.timestamp.eq(dates[40] + timedelta(hours=14, minutes=1))
    minutes = minutes.loc[~mask]
    scores = compute_factor(daily, minutes).query("symbol == 'A'").set_index("date")
    assert scores.loc[dates[40], "invalid_reason"] == "incomplete_tail"
    assert not scores.loc[dates[41], "eligible"]
    assert scores.loc[dates[41], "invalid_reason"] == "insufficient_tail_history"


def test_constant_reference_tail_volatility_is_unavailable_not_infinity():
    daily, minutes, dates = sample_data()
    minutes["close"] = minutes["open"]
    # Force every whole window to have exactly zero return, regardless of its
    # internal bars, so a zero reference scale is tested rather than rounding.
    for _, indexes in minutes.groupby([minutes.timestamp.dt.normalize(), "symbol"]).groups.items():
        minutes.loc[indexes, ["open", "close"]] = 100
    scores = compute_factor(daily, minutes)
    assert not scores.loc[scores.date.eq(dates[40]), "eligible"].any()
    assert scores.loc[scores.date.eq(dates[40]), "factor"].isna().all()


def test_untradeable_day_does_not_become_valid_zero():
    daily, minutes, dates = sample_data()
    daily["tradable"] = True
    daily.loc[daily.symbol.eq("A") & daily.date.eq(dates[40]), "tradable"] = False
    row = compute_factor(daily, minutes).query("symbol == 'A'").set_index("date").loc[dates[40]]
    assert not row.eligible
    assert row.invalid_reason == "not_tradable"
    assert np.isnan(row.factor)


def test_unknown_intraday_symbol_is_not_silently_ignored():
    daily, minutes, _ = sample_data()
    minutes.loc[0, "symbol"] = "UNKNOWN"
    with pytest.raises(DataValidationError, match="symbols absent"):
        compute_factor(daily, minutes)


def test_intraday_session_without_calendar_date_requires_complete_calendar():
    daily, minutes, dates = sample_data()
    daily = daily.loc[~daily.date.eq(dates[37])]
    with pytest.raises(DataValidationError, match="sessions absent"):
        compute_factor(daily, minutes)
