"""Calendar-aligned labels and an explicit, long-only execution experiment.

Signals are observed at close t. A cohort enters at open t+1, exits at close
t+horizon, and is fully liquidated before the next cohort enters. The benchmark
uses the same schedule, costs, and eligibility universe. No labels are used to
select holdings.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from typing import Any

import numpy as np
import pandas as pd

from .config import BacktestConfig


def _dates(values: Any, name: str) -> pd.DatetimeIndex:
    try:
        result = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain valid trading dates") from exc
    if result.hasnans:
        raise ValueError(f"{name} cannot contain missing dates")
    if result.tz is not None:
        result = result.tz_localize(None)
    return result.normalize()


def _daily_and_calendar(
    daily: pd.DataFrame, calendar: Sequence[Any] | None
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    required = {"date", "symbol", "open", "close"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Daily data is missing columns: {sorted(missing)}")
    frame = daily.copy()
    if frame.empty:
        raise ValueError("Daily data cannot be empty")
    frame["date"] = _dates(frame["date"], "daily.date")
    if frame["symbol"].isna().any():
        raise ValueError("Daily symbols cannot be missing")
    frame["symbol"] = frame["symbol"].astype(str)
    if frame["symbol"].str.strip().eq("").any():
        raise ValueError("Daily symbols cannot be empty")
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("Daily data contains duplicate (date, symbol) rows")
    for column in ("open", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        known = frame[column].notna()
        if (~np.isfinite(frame.loc[known, column])).any() or (frame.loc[known, column] <= 0).any():
            raise ValueError(f"Daily {column} must be finite and positive when present")
    if "tradable" not in frame:
        frame["tradable"] = True
    else:
        if frame["tradable"].isna().any() or not frame["tradable"].isin([True, False, 0, 1]).all():
            raise ValueError("Daily tradable must contain booleans, not strings or nulls")
        frame["tradable"] = frame["tradable"].astype(bool)
    observed = pd.DatetimeIndex(frame["date"].unique()).sort_values()
    if calendar is None:
        sessions = observed
    else:
        sessions = _dates(calendar, "calendar")
        if sessions.has_duplicates:
            raise ValueError("Calendar contains duplicate sessions")
        sessions = sessions.sort_values()
        if not observed.isin(sessions).all():
            raise ValueError("Calendar does not include every daily-data session")
    return frame.sort_values(["date", "symbol"]), sessions


def _validate_horizon(horizon: int) -> None:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")


def make_forward_returns(
    daily: pd.DataFrame,
    horizon: int = 5,
    calendar: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Return gross open(t+1)-to-close(t+horizon) labels on a shared calendar.

    The result contains the complete date/symbol grid. Missing or untradable
    endpoints produce NaN labels. A missing stock observation never shortens its
    holding horizon. These labels are for research, not for portfolio selection.
    """
    _validate_horizon(horizon)
    frame, sessions = _daily_and_calendar(daily, calendar)
    symbols = sorted(frame["symbol"].unique())
    opens = frame.pivot(index="date", columns="symbol", values="open").reindex(
        index=sessions, columns=symbols
    )
    closes = frame.pivot(index="date", columns="symbol", values="close").reindex(
        index=sessions, columns=symbols
    )
    tradable = frame.pivot(index="date", columns="symbol", values="tradable").reindex(
        index=sessions, columns=symbols, fill_value=False
    )
    entry = opens.shift(-1)
    exit_ = closes.shift(-horizon)
    permitted = tradable.shift(-1).eq(True) & tradable.shift(-horizon).eq(True)
    labels = (exit_.div(entry) - 1.0).where(permitted)
    entry_dates = pd.Series(sessions, index=sessions).shift(-1)
    exit_dates = pd.Series(sessions, index=sessions).shift(-horizon)
    count = len(symbols)
    return pd.DataFrame(
        {
            "date": np.repeat(sessions.to_numpy(), count),
            "symbol": np.tile(np.asarray(symbols, dtype=object), len(sessions)),
            "forward_return": labels.to_numpy().ravel(),
            "entry_date": np.repeat(entry_dates.to_numpy(), count),
            "exit_date": np.repeat(exit_dates.to_numpy(), count),
        }
    )


def _signal_frame(factors: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    needed = {"date", "symbol", "factor"}
    if not needed.issubset(factors.columns):
        raise ValueError(f"Factor data is missing columns: {sorted(needed - set(factors))}")
    frame = factors.copy()
    frame["date"] = _dates(frame["date"], "factors.date")
    if frame["symbol"].isna().any():
        raise ValueError("Factor symbols cannot be missing")
    frame["symbol"] = frame["symbol"].astype(str)
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("Factor data contains duplicate (date, symbol) rows")
    if not frame["date"].isin(sessions).all():
        raise ValueError("Factor dates must belong to the daily trading calendar")
    frame["factor"] = pd.to_numeric(frame["factor"], errors="raise")
    if "eligible" in frame:
        if frame["eligible"].isna().any() or not frame["eligible"].isin([True, False, 0, 1]).all():
            raise ValueError("Factor eligible must contain booleans")
        frame = frame[frame["eligible"].astype(bool)]
    frame = frame[np.isfinite(frame["factor"])]
    return frame.sort_values(["date", "symbol"])


def _cohort(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    holding_dates: pd.DatetimeIndex,
    signal_date: pd.Timestamp,
    starting_nav: float,
    cost: float,
    portfolio: str,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Execute a pre-selected cohort. Missing future data raises, never filters."""
    if selected.empty:
        return [starting_nav] * len(holding_dates), []
    allocation = starting_nav / len(selected)
    cash = starting_nav
    positions: list[tuple[str, float, dict[str, Any]]] = []
    trades: list[dict[str, Any]] = []
    entry_date, exit_date = holding_dates[0], holding_dates[-1]
    for row in selected.itertuples(index=False):
        symbol = str(row.symbol)
        key = (entry_date, symbol)
        if key not in prices.index:
            raise ValueError(
                f"Missing entry data for {symbol} on {entry_date.date()}; "
                "cannot infer an executable open from a data gap"
            )
        quote = prices.loc[key]
        trade: dict[str, Any] = {
            "signal_date": signal_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "symbol": symbol,
            "portfolio": portfolio,
            "score": float(row.factor),
            "entry_price": np.nan,
            "exit_price": np.nan,
            "gross_return": np.nan,
            "net_return": 0.0,
            "entry_allocation": allocation,
            "entry_notional": 0.0,
            "buy_cost": 0.0,
            "sell_cost": 0.0,
            "status": "skipped_entry_untradable",
        }
        if not bool(quote["tradable"]):
            trades.append(trade)
            continue
        opening = float(quote["open"])
        if not np.isfinite(opening) or opening <= 0:
            raise ValueError(f"Missing executable open for {symbol} on {entry_date.date()}")
        notional = allocation / (1.0 + cost)
        quantity = notional / opening
        trade.update(
            entry_price=opening,
            entry_notional=notional,
            buy_cost=notional * cost,
            status="executed",
        )
        cash -= allocation
        positions.append((symbol, quantity, trade))
        trades.append(trade)
    navs: list[float] = []
    for day in holding_dates:
        market_value = 0.0
        for symbol, quantity, trade in positions:
            key = (day, symbol)
            if key not in prices.index:
                raise ValueError(
                    f"Missing mark for held {symbol} on {day.date()}; "
                    "provide complete prices or a separate suspension/delisting model"
                )
            quote = prices.loc[key]
            close = float(quote["close"])
            if not np.isfinite(close) or close <= 0:
                raise ValueError(f"Missing close mark for held {symbol} on {day.date()}")
            value = quantity * close
            if day == exit_date:
                if not bool(quote["tradable"]):
                    raise ValueError(
                        f"Cannot liquidate {symbol} on {day.date()}: tradable=False; "
                        "a delayed-exit execution model is required"
                    )
                sale_cost = value * cost
                value -= sale_cost
                trade.update(
                    exit_price=close,
                    gross_return=close / trade["entry_price"] - 1.0,
                    net_return=value / allocation - 1.0,
                    sell_cost=sale_cost,
                )
            market_value += value
        navs.append(cash + market_value)
    return navs, trades


def _drawdown(values: pd.Series) -> float:
    return float((values / values.cummax() - 1.0).min())


def run_backtest(
    daily: pd.DataFrame,
    factors: pd.DataFrame,
    config: BacktestConfig = BacktestConfig(),
    calendar: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Run fully funded, nonoverlapping long-only cohorts and their benchmark.

    At every scheduled signal close select up to ceil(top_fraction * n_eligible)
    strictly positive scores. Exact score ties use symbol order. Each selected
    name receives an equal cash slot; an untradable next open leaves that slot
    in cash. No surviving names are reweighted using next-day information.

    Costs are charged on each side's actual traded notional. A cash slot A buys
    A/(1+c) worth of stock, where c = cost_bps/10000. Missing held marks or an
    untradable scheduled exit raise a clear error instead of fabricating P&L.
    """
    _validate_horizon(config.horizon)
    if not 0 < config.top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    if not np.isfinite(config.cost_bps) or not 0 <= config.cost_bps < 10000:
        raise ValueError("cost_bps must be finite and in [0, 10000)")
    frame, sessions = _daily_and_calendar(daily, calendar)
    signals = _signal_frame(factors, sessions)
    prices = frame.set_index(["date", "symbol"])
    eligible_by_day = {day: group for day, group in signals.groupby("date", sort=True)}
    first_signal = signals["date"].min() if not signals.empty else None
    start = int(sessions.get_indexer([first_signal])[0]) if first_signal is not None else 0
    equity = pd.DataFrame(
        {"date": sessions[start:], "strategy_nav": 1.0, "benchmark_nav": 1.0}
    ).set_index("date")
    strategy_nav, benchmark_nav = 1.0, 1.0
    trades: list[dict[str, Any]] = []
    cohort_count = 0
    cursor = start
    empty = pd.DataFrame(columns=["symbol", "factor"])
    while first_signal is not None and cursor + config.horizon < len(sessions):
        signal_date = sessions[cursor]
        eligible = eligible_by_day.get(signal_date, empty)
        candidates = eligible[eligible["factor"] > 0].sort_values(
            ["factor", "symbol"], ascending=[False, True], kind="stable"
        )
        top_count = int(ceil(config.top_fraction * len(eligible)))
        selected = candidates.head(top_count)
        holding_dates = sessions[cursor + 1 : cursor + config.horizon + 1]
        strategy_path, strategy_trades = _cohort(
            selected,
            prices,
            holding_dates,
            signal_date,
            strategy_nav,
            config.cost_bps / 10000.0,
            "strategy",
        )
        benchmark_path, benchmark_trades = _cohort(
            eligible,
            prices,
            holding_dates,
            signal_date,
            benchmark_nav,
            config.cost_bps / 10000.0,
            "benchmark",
        )
        equity.loc[holding_dates, "strategy_nav"] = strategy_path
        equity.loc[holding_dates, "benchmark_nav"] = benchmark_path
        strategy_nav, benchmark_nav = strategy_path[-1], benchmark_path[-1]
        trades.extend(strategy_trades)
        trades.extend(benchmark_trades)
        cohort_count += 1
        cursor += config.horizon
    # An incomplete final cohort is never opened; final leftover sessions are cash.
    if cursor + 1 < len(sessions):
        equity.loc[sessions[cursor + 1 :], "strategy_nav"] = strategy_nav
        equity.loc[sessions[cursor + 1 :], "benchmark_nav"] = benchmark_nav
    equity["strategy_return"] = equity["strategy_nav"].pct_change().fillna(0.0)
    equity["benchmark_return"] = equity["benchmark_nav"].pct_change().fillna(0.0)
    columns = [
        "signal_date",
        "entry_date",
        "exit_date",
        "symbol",
        "portfolio",
        "score",
        "entry_price",
        "exit_price",
        "gross_return",
        "net_return",
        "entry_allocation",
        "entry_notional",
        "buy_cost",
        "sell_cost",
        "status",
    ]
    trade_frame = pd.DataFrame(trades, columns=columns)
    executed = trade_frame[trade_frame["status"] == "executed"]
    skipped = trade_frame[trade_frame["status"] != "executed"]
    summary: dict[str, Any] = {
        "status": "ok"
        if cohort_count
        else ("no_eligible_signals" if first_signal is None else "insufficient_forward_sessions"),
        "holding_sessions": config.horizon,
        "execution": "next_open_to_horizon_close_nonoverlapping",
        "top_fraction": config.top_fraction,
        "cost_bps_per_side": config.cost_bps,
        "cohorts": cohort_count,
        "incomplete_final_cohorts_skipped": int(
            first_signal is not None and cursor + 1 < len(sessions)
        ),
        "strategy_total_return": float(strategy_nav - 1.0),
        "benchmark_total_return": float(benchmark_nav - 1.0),
        "relative_wealth_return": float(strategy_nav / benchmark_nav - 1.0),
        "strategy_max_drawdown": _drawdown(equity["strategy_nav"]),
        "benchmark_max_drawdown": _drawdown(equity["benchmark_nav"]),
        "strategy_executed_positions": int((executed["portfolio"] == "strategy").sum()),
        "benchmark_executed_positions": int((executed["portfolio"] == "benchmark").sum()),
        "skipped_entry_positions": len(skipped),
        "strategy_trading_cost": float(
            executed.loc[executed["portfolio"] == "strategy", ["buy_cost", "sell_cost"]]
            .to_numpy(dtype=float)
            .sum()
        ),
        "benchmark_trading_cost": float(
            executed.loc[executed["portfolio"] == "benchmark", ["buy_cost", "sell_cost"]]
            .to_numpy(dtype=float)
            .sum()
        ),
        "first_signal_date": str(first_signal.date()) if first_signal is not None else None,
        "last_liquidation_date": str(sessions[cursor].date()) if cohort_count else None,
        "last_data_date": str(sessions[-1].date()),
        "benchmark_description": "equal_weight_signal_eligible_universe_same_schedule_and_costs",
    }
    return {"equity": equity.reset_index(), "trades": trade_frame, "summary": summary}
