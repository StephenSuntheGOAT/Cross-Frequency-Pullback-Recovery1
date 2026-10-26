"""Execution tests use prices with hand-computable cash flows."""

import numpy as np
import pandas as pd
import pytest

from cfpr.backtest import make_forward_returns, run_backtest
from cfpr.config import BacktestConfig


def daily_prices(n=12, symbols=("A", "B")):
    dates = pd.bdate_range("2024-01-02", periods=n)
    rows = []
    for i, date in enumerate(dates):
        for symbol in symbols:
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": 100 + 10 * max(i - 1, 0) if symbol == "A" else 100.0,
                    "close": 100 + 10 * i if symbol == "A" else 100.0,
                    "volume": 1000,
                    "tradable": True,
                }
            )
    return pd.DataFrame(rows)


def signals(daily):
    result = daily[["date", "symbol"]].copy()
    result["factor"] = result["symbol"].map({"A": 1.0, "B": 0.0}).fillna(1.0)
    result["eligible"] = True
    return result


def test_labels_enter_next_open_and_exit_fifth_close():
    daily = daily_prices()
    labels = make_forward_returns(daily)
    first = labels[(labels["symbol"] == "A")].iloc[0]
    dates = sorted(daily["date"].unique())
    assert first["forward_return"] == pytest.approx(0.5)
    assert first["entry_date"] == dates[1]
    assert first["exit_date"] == dates[5]
    assert labels.groupby("symbol").tail(5)["forward_return"].isna().all()


def test_labels_never_compress_missing_stock_dates():
    daily = daily_prices(n=9)
    dates = sorted(daily["date"].unique())
    daily = daily[~((daily["symbol"] == "A") & daily["date"].isin([dates[2], dates[5]]))]
    labels = make_forward_returns(daily).set_index(["date", "symbol"])
    assert np.isnan(labels.loc[(dates[0], "A"), "forward_return"])
    assert np.isnan(labels.loc[(dates[1], "A"), "forward_return"])
    assert labels.loc[(dates[0], "B"), "forward_return"] == 0
    assert labels.loc[(dates[0], "A"), "exit_date"] == dates[5]


def test_nonoverlap_cohorts_are_exactly_funded_and_marked():
    daily = daily_prices()
    result = run_backtest(daily, signals(daily), BacktestConfig(cost_bps=0))
    equity = result["equity"]
    trades = result["trades"]
    chosen = trades[trades["portfolio"] == "strategy"]
    assert result["summary"]["cohorts"] == 2
    assert len(chosen) == 2
    assert equity["strategy_nav"].iloc[1] == pytest.approx(1.1)
    assert equity["strategy_nav"].iloc[5] == pytest.approx(1.5)
    assert equity["strategy_nav"].iloc[10] == pytest.approx(2.0)
    assert equity["strategy_nav"].iloc[-1] == pytest.approx(2.0)  # incomplete cohort is cash
    assert equity["benchmark_nav"].iloc[10] == pytest.approx(1.25 * (1 + 1 / 6))
    assert chosen["entry_notional"].iloc[1] == pytest.approx(1.5)
    assert chosen["signal_date"].iloc[1] == chosen["exit_date"].iloc[0]
    assert chosen["entry_date"].iloc[1] > chosen["exit_date"].iloc[0]


def test_flat_prices_charge_both_sides_on_actual_notional():
    daily = daily_prices(n=2, symbols=("B",))
    factors = signals(daily)
    factors["factor"] = 1.0
    result = run_backtest(daily, factors, BacktestConfig(horizon=1, cost_bps=100, top_fraction=1))
    expected = 0.99 / 1.01
    assert result["equity"]["strategy_nav"].iloc[-1] == pytest.approx(expected)
    trade = result["trades"].query("portfolio == 'strategy'").iloc[0]
    assert trade["entry_notional"] == pytest.approx(1 / 1.01)
    assert trade["buy_cost"] == pytest.approx(0.01 / 1.01)
    assert trade["sell_cost"] == pytest.approx(0.01 / 1.01)
    assert trade["net_return"] == pytest.approx(expected - 1)


def test_untradable_next_open_retains_cash_slot_without_reweighting():
    daily = daily_prices(n=6)
    entry_date = daily["date"].unique()[1]
    daily.loc[(daily["symbol"] == "B") & (daily["date"] == entry_date), "tradable"] = False
    factors = signals(daily)
    factors["factor"] = 1.0
    result = run_backtest(daily, factors, BacktestConfig(cost_bps=0, top_fraction=1))
    assert result["equity"]["strategy_nav"].iloc[-1] == pytest.approx(0.5 * 1.5 + 0.5)
    assert result["summary"]["skipped_entry_positions"] == 2  # strategy and benchmark
    chosen = result["trades"].query("portfolio == 'strategy' and symbol == 'B'").iloc[0]
    assert chosen["status"] == "skipped_entry_untradable"
    assert chosen["entry_notional"] == 0


def test_future_untradable_exit_is_error_not_survivorship_filter():
    daily = daily_prices(n=6)
    exit_date = daily["date"].max()
    daily.loc[(daily["symbol"] == "A") & (daily["date"] == exit_date), "tradable"] = False
    with pytest.raises(ValueError, match="Cannot liquidate A"):
        run_backtest(daily, signals(daily), BacktestConfig(cost_bps=0))


def test_missing_future_mark_is_error_not_lookahead_selection():
    daily = daily_prices(n=6)
    factors = signals(daily)
    missing = daily["date"].unique()[3]
    daily = daily[~((daily["symbol"] == "A") & (daily["date"] == missing))]
    with pytest.raises(ValueError, match="Missing mark for held A"):
        run_backtest(daily, factors, BacktestConfig(cost_bps=0))


def test_zero_scores_remain_cash_and_eligible_benchmark_still_trades():
    daily = daily_prices(n=6)
    factors = signals(daily)
    factors["factor"] = 0.0
    result = run_backtest(daily, factors, BacktestConfig(cost_bps=0))
    assert result["summary"]["strategy_executed_positions"] == 0
    assert result["equity"]["strategy_nav"].iloc[-1] == 1
    assert result["equity"]["benchmark_nav"].iloc[-1] == pytest.approx(1.25)


def test_ties_use_symbol_and_positive_filter_without_future_returns():
    daily = daily_prices(n=6)
    factors = signals(daily)
    factors["factor"] = 1.0
    # Reversing rows must not change which tied stock is selected.
    result = run_backtest(daily, factors.iloc[::-1], BacktestConfig(cost_bps=0))
    traded = result["trades"].query("portfolio == 'strategy'")
    assert traded["symbol"].tolist() == ["A"]


def test_explicit_calendar_preserves_absent_market_session():
    daily = daily_prices(n=8, symbols=("A",))
    all_dates = pd.DatetimeIndex(daily["date"])
    daily = daily[daily["date"] != all_dates[5]]
    result = make_forward_returns(daily, calendar=all_dates)
    row = result.iloc[0]
    assert row["exit_date"] == all_dates[5]
    assert np.isnan(row["forward_return"])


def test_no_eligible_or_no_complete_horizon_returns_explicit_status():
    daily = daily_prices(n=3)
    factors = signals(daily)
    result = run_backtest(daily, factors)
    assert result["summary"]["status"] == "insufficient_forward_sessions"
    factors["eligible"] = False
    result = run_backtest(daily, factors)
    assert result["summary"]["status"] == "no_eligible_signals"
    assert result["equity"]["strategy_nav"].eq(1).all()


def test_duplicate_daily_rows_and_invalid_calendar_rejected():
    daily = daily_prices(n=6)
    with pytest.raises(ValueError, match="duplicate"):
        make_forward_returns(pd.concat([daily, daily.iloc[:1]]))
    with pytest.raises(ValueError, match="include every"):
        make_forward_returns(daily, calendar=sorted(daily["date"].unique())[1:])
