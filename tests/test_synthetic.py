"""Synthetic data must be reproducible and obey the real input contract."""

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from cfpr.config import FactorConfig
from cfpr.synthetic import generate_synthetic_data


def test_generation_is_reproducible_and_seed_changes_prices():
    daily, minutes = generate_synthetic_data(symbols=3, days=4, seed=19)
    again_daily, again_minutes = generate_synthetic_data(symbols=3, days=4, seed=19)
    assert_frame_equal(daily, again_daily)
    assert_frame_equal(minutes, again_minutes)
    other_daily, _ = generate_synthetic_data(symbols=3, days=4, seed=20)
    assert not np.allclose(daily["close"], other_daily["close"])


def test_bar_paths_are_complete_and_match_daily_close_and_volume():
    daily, minutes = generate_synthetic_data(symbols=4, days=5)
    assert len(daily) == 20
    assert len(minutes) == 20 * 60
    assert str(minutes["timestamp"].dt.tz) == "Asia/Shanghai"
    assert (daily[["open", "close", "volume"]] > 0).all().all()
    assert (minutes[["open", "close"]] > 0).all().all()
    assert (minutes["volume"] >= 0).all()
    assert daily["tradable"].all()
    minutes = minutes.assign(date=minutes["timestamp"].dt.tz_localize(None).dt.normalize())
    grouped = minutes.groupby(["date", "symbol"], sort=True)
    assert grouped.size().eq(60).all()
    first = grouped.first()
    last = grouped.last()
    assert first["timestamp"].dt.strftime("%H:%M").eq("14:00").all()
    assert last["timestamp"].dt.strftime("%H:%M").eq("14:59").all()
    indexed_daily = daily.set_index(["date", "symbol"])
    assert (last["close"] == indexed_daily["close"]).all()
    assert (grouped["volume"].sum() <= indexed_daily["volume"]).all()
    for _, bars in grouped:
        np.testing.assert_array_equal(bars["open"].iloc[1:], bars["close"].iloc[:-1])


def test_generated_frames_validate_against_the_data_contract():
    from cfpr.data import aggregate_tail, validate_daily, validate_intraday

    daily, minutes = generate_synthetic_data(symbols=3, days=4)
    checked_daily = validate_daily(daily)
    checked_minutes = validate_intraday(minutes, FactorConfig())
    assert len(checked_daily) == 12
    assert len(checked_minutes) == 720
    tails = aggregate_tail(checked_minutes, FactorConfig())
    assert tails["tail_complete"].all()
    assert tails["tail_bars"].eq(60).all()


def test_custom_tail_window_and_timezone():
    config = FactorConfig(tail_minutes=30, session_close="16:00", market_timezone="UTC")
    daily, minutes = generate_synthetic_data(symbols=2, days=1, config=config)
    assert len(daily) == 2
    assert len(minutes) == 60
    assert str(minutes["timestamp"].dt.tz) == "UTC"
    assert minutes["timestamp"].dt.strftime("%H:%M").min() == "15:30"
    assert minutes["timestamp"].dt.strftime("%H:%M").max() == "15:59"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"symbols": 1},
        {"symbols": 2.5},
        {"symbols": True},
        {"days": 0},
        {"days": False},
        {"seed": -1},
        {"seed": True},
        {"start": "not-a-date"},
        {"start": "2023-01-02T00:00:00Z"},
    ],
)
def test_invalid_generation_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        generate_synthetic_data(**kwargs)
