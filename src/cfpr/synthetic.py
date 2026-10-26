"""Reproducible, internally consistent synthetic prices for offline examples.

These are simulated weekday observations, not exchange data.  In particular,
the generator neither embeds the proposed factor in future returns nor serves
as evidence that the factor predicts real stock returns.
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from .config import FactorConfig


def generate_synthetic_data(
    symbols: int = 40,
    days: int = 160,
    seed: int = 42,
    start: str = "2023-01-02",
    config: FactorConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return daily prices and complete one-minute tail-window bars.

    The return shocks on a future day are independent of the past factor.
    Names share contemporaneous market shocks and have different fixed
    volatilities. Prices are generated in log space, and every day's daily
    close equals its final minute close exactly. Minute timestamps denote
    the *start* of the bar. All quantities are in arbitrary simulated units.

    ``start`` must be a timezone-naive date. Business weekdays are used only
    to make a small offline example; statutory exchange holidays are not
    represented. Pass a ``FactorConfig`` to change the tail window or zone.
    """
    if isinstance(symbols, bool) or not isinstance(symbols, (int, np.integer)) or symbols < 2:
        raise ValueError("symbols must be an integer of at least 2")
    if isinstance(days, bool) or not isinstance(days, (int, np.integer)) or days < 1:
        raise ValueError("days must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    try:
        first_date = pd.Timestamp(start)
    except (TypeError, ValueError) as exc:
        raise ValueError("start must be a valid timezone-naive date") from exc
    if pd.isna(first_date) or first_date.tzinfo is not None:
        raise ValueError("start must be a valid timezone-naive date")
    cfg = config if config is not None else FactorConfig()
    rng = np.random.default_rng(seed)
    count, sessions, minutes = int(symbols), int(days), cfg.tail_minutes
    dates = pd.bdate_range(first_date.normalize(), periods=sessions)
    names = np.array([f"SYN{i + 1:04d}" for i in range(count)])

    # Independent day-to-day innovations. The fixed drift is identical for
    # every name; no signal-dependent expected return has been introduced.
    idio_vol = rng.uniform(0.009, 0.020, size=(1, count))
    components: list[np.ndarray] = []
    for variance_share, drift in ((0.08, 0.0), (0.67, 0.0002), (0.25, 0.0)):
        common_sigma = 0.004 * np.sqrt(variance_share)
        stock_sigma = idio_vol * np.sqrt(variance_share)
        common = rng.normal(0.0, common_sigma, size=(sessions, 1))
        individual = rng.normal(0.0, 1.0, size=(sessions, count)) * stock_sigma
        components.append(drift + common + individual - 0.5 * (common_sigma**2 + stock_sigma**2))
    overnight, morning, tail = components
    initial = rng.uniform(12.0, 90.0, size=count)
    closes = initial * np.exp(np.cumsum(overnight + morning + tail, axis=0))
    previous_closes = np.concatenate((initial[None, :], closes[:-1]), axis=0)
    opens = previous_closes * np.exp(overnight)
    tail_opens = opens * np.exp(morning)

    # A Gaussian bridge makes the minute path finish at the drawn tail
    # return, while leaving its intrawindow shape random and continuous.
    minute_increments = rng.normal(0.0, 1.0, size=(sessions, count, minutes))
    minute_increments *= idio_vol[:, :, None] * np.sqrt(0.25 / minutes)
    minute_increments += (tail - minute_increments.sum(axis=2))[:, :, None] / minutes
    minute_closes = tail_opens[:, :, None] * np.exp(np.cumsum(minute_increments, axis=2))
    minute_closes[:, :, -1] = closes
    minute_opens = np.concatenate((tail_opens[:, :, None], minute_closes[:, :, :-1]), axis=2)

    base_volume = rng.uniform(1_000_000, 6_000_000, size=(1, count))
    daily_volumes = np.maximum(
        1_000, np.rint(base_volume * np.exp(rng.normal(0.0, 0.28, size=(sessions, count))))
    ).astype(np.int64)
    fractions = rng.uniform(0.15, 0.42, size=(sessions, count))
    tail_volumes = np.maximum(1, np.rint(daily_volumes * fractions)).astype(np.int64)
    shape = np.linspace(0.8, 1.5, minutes)[None, None, :]
    volume_weights = rng.gamma(3.0, 1.0, size=(sessions, count, minutes)) * shape
    volume_weights /= volume_weights.sum(axis=2, keepdims=True)
    minute_volumes = (volume_weights * tail_volumes[:, :, None]).astype(np.int64)
    minute_volumes[:, :, -1] += tail_volumes - minute_volumes.sum(axis=2)

    daily = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), count),
            "symbol": np.tile(names, sessions),
            "open": opens.ravel(),
            "close": closes.ravel(),
            "volume": daily_volumes.ravel(),
            "tradable": True,
        }
    )
    close_time = time.fromisoformat(cfg.session_close)
    # Explicit nanoseconds avoid depending on pandas' default datetime unit.
    tail_starts_ns = (
        dates.tz_localize(cfg.market_timezone).as_unit("ns").asi8
        + (close_time.hour * 60 + close_time.minute - minutes) * 60_000_000_000
    )
    offsets = np.tile(np.arange(minutes, dtype=np.int64), sessions * count) * 60_000_000_000
    timestamps = pd.to_datetime(
        np.repeat(tail_starts_ns, count * minutes) + offsets, utc=True
    ).tz_convert(cfg.market_timezone)
    intraday = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": np.tile(np.repeat(names, minutes), sessions),
            "open": minute_opens.ravel(),
            "close": minute_closes.ravel(),
            "volume": minute_volumes.ravel(),
        }
    )
    return daily, intraday
