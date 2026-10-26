"""Strict, vendor-neutral daily and one-minute market-data ingestion.

Minute timestamps label the **start** of each bar.  Naive timestamps are
interpreted in the configured market timezone; timezone-aware input is converted
to that timezone.  Missing bars remain missing rather than being filled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import FactorConfig


class DataValidationError(ValueError):
    """Input market data violates the documented data contract."""


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"symbol": "string"})
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise DataValidationError(f"Unsupported data format {path.suffix!r}; use CSV or Parquet.")


def _required(frame: pd.DataFrame, columns: list[str], kind: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError(f"{kind} data must be a pandas DataFrame.")
    if frame.columns.duplicated().any():
        raise DataValidationError(f"{kind} data has duplicate column names.")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise DataValidationError(f"{kind} data is missing required columns: {', '.join(missing)}.")


def _symbols(frame: pd.DataFrame) -> None:
    if frame["symbol"].isna().any():
        raise DataValidationError("symbol cannot contain missing values.")
    frame["symbol"] = frame["symbol"].astype("string").str.strip()
    if frame["symbol"].eq("").any():
        raise DataValidationError("symbol cannot contain empty values.")


def _numbers(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise DataValidationError(f"{column} must contain numeric values.") from exc
        if not np.isfinite(frame[column].to_numpy()).all():
            raise DataValidationError(f"{column} must contain finite, nonmissing values.")
        if column == "volume":
            if frame[column].lt(0).any():
                raise DataValidationError("volume must be nonnegative.")
        elif frame[column].le(0).any():
            raise DataValidationError(f"{column} prices must be strictly positive.")


def _dates(values: pd.Series) -> pd.Series:
    try:
        dates = pd.to_datetime(values, errors="raise", format="mixed")
    except (ValueError, TypeError) as exc:
        raise DataValidationError("date must contain valid trading-session dates.") from exc
    if isinstance(dates.dtype, pd.DatetimeTZDtype) or not pd.api.types.is_datetime64_any_dtype(
        dates.dtype
    ):
        raise DataValidationError("Daily date must be timezone-naive; use YYYY-MM-DD.")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise DataValidationError("Daily date must be a nonmissing calendar date at midnight.")
    return dates.astype("datetime64[ns]")


def validate_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, normalized copy of daily data or raise a useful error.

    Required columns: ``date, symbol, open, close, volume``.  ``tradable`` is an
    optional boolean (also accepts 0/1 and true/false strings), defaulting to True.
    Suspended or otherwise untradeable observations should explicitly be False;
    their carried-forward prices do not enter factor histories.
    """
    _required(frame, ["date", "symbol", "open", "close", "volume"], "Daily")
    result = frame.copy()
    result["date"] = _dates(result["date"])
    _symbols(result)
    _numbers(result, ["open", "close", "volume"])
    if "tradable" not in result:
        result["tradable"] = True
    else:
        values = result["tradable"].map(
            lambda value: str(value).strip().lower() if not pd.isna(value) else ""
        )
        mapping = {"true": True, "false": False, "1": True, "0": False}
        if not values.isin(mapping).all():
            raise DataValidationError(
                "tradable must contain true/false or 1/0 without missing values."
            )
        result["tradable"] = values.map(mapping).astype(bool)
    if result.duplicated(["date", "symbol"]).any():
        raise DataValidationError("Daily data has duplicate (date, symbol) rows.")
    return result.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


def _timestamps(values: pd.Series, timezone: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, errors="raise", format="mixed")
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            parsed = parsed.dt.tz_convert(timezone)
        elif pd.api.types.is_datetime64_any_dtype(parsed.dtype):
            parsed = parsed.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
        else:
            # Different UTC offsets (for example a DST transition) are supported
            # if every timestamp is aware.  Mixing naive and aware times is not.
            timestamps = [pd.Timestamp(value) for value in values]
            if any(value.tzinfo is None for value in timestamps):
                raise ValueError("mixed naive and timezone-aware timestamps")
            parsed = pd.Series(
                pd.to_datetime(values, errors="raise", format="mixed", utc=True),
                index=values.index,
            ).dt.tz_convert(timezone)
    except (ValueError, TypeError, KeyError) as exc:
        raise DataValidationError(
            "timestamp must contain valid times with consistent timezone conventions."
        ) from exc
    if parsed.isna().any():
        raise DataValidationError("timestamp cannot contain missing values.")
    if not parsed.eq(parsed.dt.floor("min")).all():
        raise DataValidationError("timestamp must label the start of an exact one-minute bar.")
    return parsed


def validate_intraday(frame: pd.DataFrame, config: FactorConfig = FactorConfig()) -> pd.DataFrame:
    """Validate one-minute OHLCV bars and convert times to market timezone."""
    _required(frame, ["timestamp", "symbol", "open", "close", "volume"], "Intraday")
    result = frame.copy()
    result["timestamp"] = _timestamps(result["timestamp"], config.market_timezone)
    _symbols(result)
    _numbers(result, ["open", "close", "volume"])
    if result.duplicated(["timestamp", "symbol"]).any():
        raise DataValidationError("Intraday data has duplicate (timestamp, symbol) rows.")
    return result.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)


def load_daily(path: str | Path) -> pd.DataFrame:
    """Load and validate daily CSV or Parquet, preserving string stock codes."""
    return validate_daily(_read_table(path))


def load_intraday(path: str | Path, config: FactorConfig = FactorConfig()) -> pd.DataFrame:
    """Load and validate start-stamped, one-minute CSV or Parquet bars."""
    return validate_intraday(_read_table(path), config)


def aggregate_tail(intraday: pd.DataFrame, config: FactorConfig = FactorConfig()) -> pd.DataFrame:
    """Aggregate complete final-window bars without inventing missing trades.

    A window is usable only when all expected minute labels are present exactly
    once.  Thus the default window contains 14:00 through 14:59 inclusive,
    representing trading from 14:00 to 15:00.  Extra bars outside this window do
    not affect its result.  Incomplete windows retain their bar counts and have
    NaN returns and volume.
    """
    frame = validate_intraday(intraday, config)
    frame["date"] = frame["timestamp"].dt.tz_localize(None).dt.normalize()
    all_keys = frame[["date", "symbol"]].drop_duplicates()
    hour, minute = (int(part) for part in config.session_close.split(":"))
    close_minute = hour * 60 + minute
    minute_of_day = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    tail = frame.loc[
        minute_of_day.ge(close_minute - config.tail_minutes) & minute_of_day.lt(close_minute)
    ]
    grouped = tail.groupby(["date", "symbol"], sort=True, observed=True)
    aggregates = grouped.agg(
        tail_first_open=("open", "first"),
        tail_last_close=("close", "last"),
        tail_volume=("volume", "sum"),
        tail_bars=("timestamp", "size"),
    ).reset_index()
    result = all_keys.merge(aggregates, on=["date", "symbol"], how="left", validate="one_to_one")
    result["tail_bars"] = result["tail_bars"].fillna(0).astype(int)
    result["tail_complete"] = result["tail_bars"].eq(config.tail_minutes)
    result["tail_return"] = result["tail_last_close"] / result["tail_first_open"] - 1.0
    result.loc[~result["tail_complete"], ["tail_return", "tail_volume"]] = np.nan
    columns = ["date", "symbol", "tail_return", "tail_volume", "tail_bars", "tail_complete"]
    return result[columns].sort_values(["date", "symbol"]).reset_index(drop=True)
