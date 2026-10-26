"""Cross-frequency pullback recovery factor, calculated using data through t.

All shifts and rolling histories operate on a shared trading-session calendar.
Today is excluded from both daily volatility and tail-window reference samples.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import FactorConfig
from .data import DataValidationError, aggregate_tail, validate_daily


def pullback_kernel(drawdown: np.ndarray | pd.Series | float):
    """Return g(D)=D exp(-D²/2); callers supply nonnegative normalized D."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return drawdown * np.exp(-0.5 * np.square(drawdown))


def _calendar(
    daily: pd.DataFrame, calendar: Sequence | pd.DatetimeIndex | None
) -> pd.DatetimeIndex:
    if calendar is None:
        return pd.DatetimeIndex(daily["date"].unique()).sort_values()
    try:
        index = pd.DatetimeIndex(pd.to_datetime(calendar, errors="raise"))
    except (ValueError, TypeError) as exc:
        raise DataValidationError("calendar must contain valid trading-session dates.") from exc
    if index.tz is not None or index.hasnans or not index.equals(index.normalize()):
        raise DataValidationError("calendar must contain timezone-naive dates at midnight.")
    if index.has_duplicates:
        raise DataValidationError("calendar contains duplicate trading sessions.")
    index = index.sort_values().astype("datetime64[ns]")
    if not daily["date"].isin(index).all():
        raise DataValidationError("daily data contains sessions absent from the supplied calendar.")
    return index


def compute_factor(
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
    config: FactorConfig = FactorConfig(),
    calendar: Sequence | pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Compute features and scores for every session/symbol in daily data.

    The default shared calendar is the union of daily dates.  For production
    research pass an official exchange calendar so a session missing for *all*
    symbols is not silently omitted.  Missing symbol dates are explicitly
    reindexed and produce unavailable history, never shortened lookback windows.

    A valid observation can have factor=0 (a gate did not fire).  Missing history,
    missing minute bars, zero reference volatility/volume, and untradeable days
    instead have factor=NaN and eligible=False.  These cases must not be ranked
    alongside valid zero signals.
    """
    daily = validate_daily(daily)
    if daily.empty:
        raise DataValidationError("Daily data must contain at least one observation.")
    sessions = _calendar(daily, calendar)
    symbols = sorted(daily["symbol"].unique())
    grid = pd.MultiIndex.from_product([sessions, symbols], names=["date", "symbol"])
    daily["_daily_present"] = True
    frame = daily.set_index(["date", "symbol"]).reindex(grid).reset_index()
    frame["_daily_present"] = frame["_daily_present"].eq(True)
    frame["tradable"] = frame["tradable"].eq(True)
    tails = aggregate_tail(intraday, config)
    if not tails["symbol"].isin(symbols).all():
        raise DataValidationError("Intraday data contains symbols absent from daily data.")
    if not tails["date"].isin(sessions).all():
        raise DataValidationError(
            "Intraday data contains sessions absent from the trading calendar; "
            "filter the input period or supply a complete exchange calendar."
        )
    frame = frame.merge(tails, on=["date", "symbol"], how="left", validate="one_to_one")
    frame["tail_complete"] = frame["tail_complete"].eq(True)
    frame["tail_bars"] = frame["tail_bars"].fillna(0).astype(int)
    frame.loc[~frame["tradable"], ["tail_return", "tail_volume"]] = np.nan
    outputs = []
    anchor = config.pullback_days + config.gap_days
    for _, original in frame.groupby("symbol", sort=True, observed=True):
        group = original.sort_values("date").copy()
        close = group["close"].where(group["tradable"])
        returns = close.pct_change(fill_method=None)
        group["trend_return"] = close.shift(anchor) / close.shift(anchor + config.trend_days) - 1.0
        group["pullback_return"] = close.shift(config.gap_days) / close.shift(anchor) - 1.0
        group["daily_vol"] = (
            returns.shift(1)
            .rolling(config.volatility_days, min_periods=config.volatility_days)
            .std(ddof=1)
        )
        group["tail_vol"] = (
            group["tail_return"]
            .shift(1)
            .rolling(config.tail_history_days, min_periods=config.tail_history_days)
            .std(ddof=1)
        )
        group["tail_volume_mean"] = (
            group["tail_volume"]
            .shift(1)
            .rolling(config.tail_history_days, min_periods=config.tail_history_days)
            .mean()
        )
        daily_scale = group["daily_vol"].where(group["daily_vol"].gt(config.epsilon))
        tail_scale = group["tail_vol"].where(group["tail_vol"].gt(config.epsilon))
        tail_volume_scale = group["tail_volume_mean"].where(
            group["tail_volume_mean"].gt(config.epsilon)
        )
        group["M"] = group["trend_return"] / (daily_scale * np.sqrt(config.trend_days))
        group["D"] = (-group["pullback_return"]).clip(lower=0) / (
            daily_scale * np.sqrt(config.pullback_days)
        )
        group["H"] = group["tail_return"] / tail_scale
        group["Q"] = group["tail_volume"] / tail_volume_scale
        group["pullback_kernel"] = pullback_kernel(group["D"])
        group["volume_multiplier"] = group["Q"].clip(upper=config.volume_cap)
        group["control_momentum"] = group["trend_return"]
        group["control_reversal"] = -group["pullback_return"]
        volume = group["volume"].where(group["tradable"])
        daily_volume_mean = (
            volume.shift(1)
            .rolling(config.volatility_days, min_periods=config.volatility_days)
            .mean()
        )
        relative_volume = volume / daily_volume_mean.where(daily_volume_mean.gt(config.epsilon))
        group["control_volume"] = np.log(relative_volume.where(relative_volume.gt(0)))
        reasons = pd.Series("", index=group.index, dtype="object")

        def mark(mask: pd.Series, reason: str) -> None:
            reasons.loc[mask & reasons.eq("")] = reason

        mark(~group["_daily_present"], "missing_daily")
        mark(~group["tradable"], "not_tradable")
        mark(
            group[["trend_return", "pullback_return"]].isna().any(axis=1),
            "insufficient_price_history",
        )
        mark(
            ~np.isfinite(group["daily_vol"]) | group["daily_vol"].le(config.epsilon),
            "invalid_daily_volatility",
        )
        mark(~group["tail_complete"], "incomplete_tail")
        mark(
            group[["tail_vol", "tail_volume_mean"]].isna().any(axis=1), "insufficient_tail_history"
        )
        mark(
            ~np.isfinite(group["tail_vol"]) | group["tail_vol"].le(config.epsilon),
            "invalid_tail_volatility",
        )
        mark(
            ~np.isfinite(group["tail_volume_mean"]) | group["tail_volume_mean"].le(config.epsilon),
            "invalid_tail_volume_baseline",
        )
        features_finite = np.isfinite(group[["M", "D", "H", "Q"]]).all(axis=1)
        mark(~features_finite, "nonfinite_features")
        group["eligible"] = reasons.eq("")
        group["hf_signal"] = group["H"].clip(lower=0) * group["volume_multiplier"]
        group.loc[~group["tradable"] | ~np.isfinite(group["hf_signal"]), "hf_signal"] = np.nan
        with np.errstate(over="ignore", invalid="ignore"):
            score = (
                group["M"].clip(lower=0)
                * group["pullback_kernel"]
                * group["H"].clip(lower=0)
                * group["volume_multiplier"]
            )
        mark(group["eligible"] & ~np.isfinite(score), "nonfinite_score")
        group["eligible"] = reasons.eq("")
        group["invalid_reason"] = reasons
        group["factor"] = score.where(group["eligible"])
        outputs.append(group)
    result = pd.concat(outputs, ignore_index=True).drop(columns="_daily_present")
    return result.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)
