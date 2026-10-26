"""Input validation is part of the research result's reliability."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from cfpr.config import FactorConfig
from cfpr.data import (
    DataValidationError,
    aggregate_tail,
    load_daily,
    validate_daily,
    validate_intraday,
)


def daily_row():
    return pd.DataFrame(
        [{"date": "2024-01-02", "symbol": "000001", "open": 10, "close": 11, "volume": 100}]
    )


def tail_bars():
    timestamps = pd.date_range("2024-01-02 14:00", periods=60, freq="min")
    prices = np.linspace(10, 10.6, 61)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "000001",
            "open": prices[:-1],
            "close": prices[1:],
            "volume": np.arange(60) + 100,
        }
    )


def test_complete_tail_uses_first_open_and_last_close():
    bars = tail_bars()
    result = aggregate_tail(bars).iloc[0]
    assert result.tail_complete
    assert result.tail_bars == 60
    assert result.tail_return == pytest.approx(0.06)
    assert result.tail_volume == pytest.approx(bars.volume.sum())


def test_missing_minute_is_not_filled():
    result = aggregate_tail(tail_bars().drop(index=12)).iloc[0]
    assert not result.tail_complete
    assert result.tail_bars == 59
    assert np.isnan(result.tail_return)
    assert np.isnan(result.tail_volume)


def test_outside_window_bars_are_ignored():
    bars = tail_bars()
    extras = bars.iloc[[0, -1]].copy()
    extras["timestamp"] = pd.to_datetime(["2024-01-02 13:59", "2024-01-02 15:00"])
    extras["volume"] = 1e12
    pd.testing.assert_frame_equal(aggregate_tail(bars), aggregate_tail(pd.concat([extras, bars])))


def test_aware_times_convert_to_market_timezone():
    bars = tail_bars()
    expected = aggregate_tail(bars)
    bars["timestamp"] = bars["timestamp"].dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")
    pd.testing.assert_frame_equal(expected, aggregate_tail(bars))


def test_csv_preserves_leading_zero_symbols(tmp_path):
    path = tmp_path / "daily.csv"
    daily_row().to_csv(path, index=False)
    assert load_daily(path).iloc[0].symbol == "000001"


@pytest.mark.parametrize(
    "field,value", [("open", 0), ("close", -1), ("volume", -1), ("close", np.nan), ("open", np.inf)]
)
def test_invalid_numbers_rejected(field, value):
    daily = daily_row()
    daily[field] = value
    with pytest.raises(DataValidationError):
        validate_daily(daily)


def test_duplicate_daily_rejected():
    row = daily_row()
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_daily(pd.concat([row, row]))


def test_duplicate_minute_rejected():
    bars = tail_bars()
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_intraday(pd.concat([bars, bars.iloc[[0]]]))


def test_non_minute_timestamp_rejected():
    bars = tail_bars()
    bars.loc[0, "timestamp"] += timedelta(seconds=1)
    with pytest.raises(DataValidationError, match="one-minute"):
        validate_intraday(bars)


@pytest.mark.parametrize("bad_symbol", [None, "", "  "])
def test_missing_symbols_rejected(bad_symbol):
    daily = daily_row()
    daily["symbol"] = bad_symbol
    with pytest.raises(DataValidationError, match="symbol"):
        validate_daily(daily)


def test_daily_date_must_be_date_only():
    daily = daily_row()
    daily["date"] = "2024-01-02 15:00"
    with pytest.raises(DataValidationError, match="midnight"):
        validate_daily(daily)


def test_untradeable_flag_is_not_string_truthiness():
    daily = daily_row()
    daily["tradable"] = "false"
    assert not validate_daily(daily).iloc[0].tradable
    daily["tradable"] = "maybe"
    with pytest.raises(DataValidationError, match="tradable"):
        validate_daily(daily)


def test_different_tail_window():
    bars = tail_bars()
    result = aggregate_tail(bars, FactorConfig(tail_minutes=30)).iloc[0]
    assert result.tail_bars == 30
    assert result.tail_return == pytest.approx(10.6 / 10.3 - 1)
