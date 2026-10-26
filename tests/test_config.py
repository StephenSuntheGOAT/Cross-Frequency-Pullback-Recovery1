import pytest

from cfpr.config import BacktestConfig, FactorConfig


@pytest.mark.parametrize(
    "overrides",
    [
        {"trend_days": 0},
        {"pullback_days": -1},
        {"gap_days": True},
        {"volatility_days": 1},
        {"tail_history_days": 1},
        {"tail_minutes": 0},
        {"session_close": "15:00:00"},
        {"session_close": "25:00"},
        {"session_close": "00:10", "tail_minutes": 60},
        {"market_timezone": "not/a/timezone"},
        {"volume_cap": 0},
        {"epsilon": float("nan")},
    ],
)
def test_invalid_factor_parameters(overrides):
    with pytest.raises(ValueError):
        FactorConfig(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"horizon": 0},
        {"horizon": 1.5},
        {"top_fraction": 0},
        {"top_fraction": 1.01},
        {"cost_bps": -1},
        {"cost_bps": 10000},
        {"cost_bps": float("inf")},
        {"min_cross_section": 1},
        {"hac_lags": -1},
    ],
)
def test_invalid_backtest_parameters(overrides):
    with pytest.raises(ValueError):
        BacktestConfig(**overrides)
