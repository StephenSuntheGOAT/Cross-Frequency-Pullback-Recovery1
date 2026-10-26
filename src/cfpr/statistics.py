"""Cross-sectional diagnostics for an unvalidated research hypothesis.

Five-day forward returns overlap.  Daily ICs and Fama--MacBeth coefficient
means therefore use a Bartlett Newey--West standard error rather than an
independence assumption.  These are descriptive, in-sample diagnostics, not a
causal claim or a substitute for a held-out research design.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import BacktestConfig

SIGNALS = (
    "factor",
    "hf_signal",
    "control_momentum",
    "control_reversal",
    "control_volume",
)
PREDICTORS = ("factor", "control_momentum", "control_reversal", "control_volume")
COEFFICIENTS = ("beta_factor", "beta_momentum", "beta_reversal", "beta_volume")
IC_COLUMNS = ("date", "n_obs", "rank_ic", "status")
BASELINE_COLUMNS = ("date", "signal", "n_obs", "rank_ic", "status")
QUANTILE_COLUMNS = ("date", "quantile", "n_obs", "mean_forward_return")
REGRESSION_COLUMNS = ("date", "n_obs", "intercept", *COEFFICIENTS, "status")


def hac_mean_test(values: pd.Series | np.ndarray, lags: int) -> dict[str, Any]:
    """Summarize a time-series mean using a Bartlett HAC standard error.

    Missing entries stay at their original time positions.  Lag products never
    join two observations across a missing session.  The variance of the mean
    is the lag-weighted sum of residual products divided by ``n_obs**2``.
    No finite-sample degrees-of-freedom correction is applied.
    """
    if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)) or lags < 0:
        raise ValueError("lags must be a nonnegative integer")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    finite = np.isfinite(array)
    n_obs = int(finite.sum())
    result: dict[str, Any] = {
        "mean": None,
        "standard_error": None,
        "t_stat": None,
        "n_obs": n_obs,
        "lags": int(lags),
    }
    if not n_obs:
        return result
    mean = float(array[finite].mean())
    result["mean"] = mean
    if n_obs < 2:
        return result
    # Exact constants can still have tiny residuals after floating-point mean
    # reduction (e.g. ten repeats of 0.01). They have no estimable t statistic.
    if np.ptp(array[finite]) == 0:
        result["standard_error"] = 0.0
        return result
    residual = np.where(finite, array - mean, 0.0)
    variance_numerator = float(residual @ residual)
    for lag in range(1, min(int(lags), len(array) - 1) + 1):
        weight = 1.0 - lag / (int(lags) + 1.0)
        variance_numerator += 2.0 * weight * float(residual[lag:] @ residual[:-lag])
    variance = max(variance_numerator / n_obs**2, 0.0)
    standard_error = math.sqrt(variance)
    result["standard_error"] = standard_error
    if standard_error > 0.0:
        result["t_stat"] = mean / standard_error
    return result


def _keyed_copy(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = {"date", "symbol"} - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}")
    result = frame.copy()
    try:
        result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{name}.date must contain valid session dates") from exc
    if result["date"].isna().any() or result["symbol"].isna().any():
        raise ValueError(f"{name} has missing date or symbol keys")
    result["symbol"] = result["symbol"].astype(str)
    if result.duplicated(["date", "symbol"]).any():
        raise ValueError(f"{name} has duplicate date/symbol keys")
    return result


def _rank_ic(frame: pd.DataFrame, signal: str, min_count: int) -> tuple[int, float, str]:
    values = frame[[signal, "forward_return"]].to_numpy(dtype=float)
    values = values[np.isfinite(values).all(axis=1)]
    n_obs = len(values)
    if n_obs < min_count:
        return n_obs, np.nan, "insufficient_observations"
    if np.ptp(values[:, 0]) == 0.0:
        return n_obs, np.nan, "constant_signal"
    if np.ptp(values[:, 1]) == 0.0:
        return n_obs, np.nan, "constant_forward_return"
    ranks = pd.DataFrame(values).rank(method="average")
    return n_obs, float(ranks.iloc[:, 0].corr(ranks.iloc[:, 1])), "ok"


def _regression(frame: pd.DataFrame, min_count: int) -> dict[str, Any]:
    result = {"n_obs": 0, "intercept": np.nan, **dict.fromkeys(COEFFICIENTS, np.nan)}
    if set(PREDICTORS) - set(frame.columns):
        return {**result, "status": "missing_controls"}
    values = frame[[*PREDICTORS, "forward_return"]].to_numpy(dtype=float)
    values = values[np.isfinite(values).all(axis=1)]
    result["n_obs"] = len(values)
    if len(values) < max(min_count, len(PREDICTORS) + 2):
        return {**result, "status": "insufficient_observations"}
    ranges = np.ptp(values, axis=0)
    if ranges[-1] == 0.0:
        return {**result, "status": "constant_forward_return"}
    if (ranges[:-1] == 0.0).any():
        return {**result, "status": "constant_predictor"}
    scales = values.std(axis=0, ddof=0)
    if not np.isfinite(scales).all() or (scales <= 0.0).any():
        return {**result, "status": "invalid_numeric_scale"}
    standardized = (values - values.mean(axis=0)) / scales
    design = np.column_stack([np.ones(len(values)), standardized[:, :-1]])
    coefficients, _, rank, _ = np.linalg.lstsq(design, standardized[:, -1], rcond=None)
    if rank < design.shape[1]:
        return {**result, "status": "rank_deficient"}
    result.update(dict(zip(("intercept", *COEFFICIENTS), coefficients, strict=True)))
    return {**result, "status": "ok"}


def _status_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame["status"].value_counts().items()}


def analyze_predictions(
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    config: BacktestConfig = BacktestConfig(),
) -> dict[str, Any]:
    """Measure IC, tie-preserving quintiles and incremental daily regressions.

    Factor inputs are observable at signal close.  ``forward_return`` is a
    separately constructed next-open to fifth-session-close outcome.  Every
    signal date is retained in diagnostic tables, including skipped dates.
    ``eligible=False`` rows are excluded, while eligible zero scores remain.
    """
    left = _keyed_copy(factors, "factors")
    right = _keyed_copy(labels, "labels")
    if "factor" not in left or "forward_return" not in right:
        raise ValueError("factors requires factor; labels requires forward_return")
    if config.min_cross_section < 2:
        raise ValueError("min_cross_section must be at least 2")
    if config.horizon < 1:
        raise ValueError("horizon must be positive")
    requested_lags = config.hac_lags
    if requested_lags is not None and (
        isinstance(requested_lags, bool)
        or not isinstance(requested_lags, (int, np.integer))
        or requested_lags < 0
    ):
        raise ValueError("hac_lags must be a nonnegative integer or None")
    lags = max(config.horizon - 1, requested_lags or 0)
    signals = [signal for signal in SIGNALS if signal in left.columns]
    for column in signals:
        left[column] = pd.to_numeric(left[column], errors="raise")
    right["forward_return"] = pd.to_numeric(right["forward_return"], errors="raise")
    if "eligible" in left:
        eligibility = left["eligible"]
        if (
            eligibility.isna().any()
            or not eligibility.map(lambda value: isinstance(value, (bool, np.bool_))).all()
        ):
            raise ValueError("eligible must contain nonmissing boolean values")
    else:
        left["eligible"] = True
    # Select label columns explicitly: accidental overlapping input names cannot
    # substitute a label or introduce suffix-dependent behavior.
    merged = left.merge(
        right[["date", "symbol", "forward_return"]],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    baseline_records: list[dict[str, Any]] = []
    quantile_records: list[dict[str, Any]] = []
    regression_records: list[dict[str, Any]] = []
    for date, whole_day in merged.groupby("date", sort=True):
        day = whole_day.loc[whole_day["eligible"]]
        for signal in signals:
            n_obs, rank_ic, status = _rank_ic(day, signal, config.min_cross_section)
            baseline_records.append(
                {
                    "date": date,
                    "signal": signal,
                    "n_obs": n_obs,
                    "rank_ic": rank_ic,
                    "status": status,
                }
            )
        regression_records.append({"date": date, **_regression(day, config.min_cross_section)})
        quantile_data = (
            day[["factor", "forward_return"]].replace([np.inf, -np.inf], np.nan).dropna()
        )
        if len(quantile_data) < config.min_cross_section or quantile_data["factor"].nunique() < 2:
            continue
        # Identical scores always share a bucket.  Empty quintiles are expected
        # when, for example, most eligible stocks have a valid zero score.
        percentile = (quantile_data["factor"].rank(method="average") - 1) / (len(quantile_data) - 1)
        quantile_data = quantile_data.assign(
            quantile=(np.floor(percentile * 5).astype(int) + 1).clip(1, 5)
        )
        for quantile, group in quantile_data.groupby("quantile", sort=True):
            quantile_records.append(
                {
                    "date": date,
                    "quantile": int(quantile),
                    "n_obs": len(group),
                    "mean_forward_return": float(group["forward_return"].mean()),
                }
            )
    baseline_ic = pd.DataFrame(baseline_records, columns=BASELINE_COLUMNS)
    ic_daily = baseline_ic.loc[baseline_ic["signal"] == "factor", list(IC_COLUMNS)].reset_index(
        drop=True
    )
    quantile_returns = pd.DataFrame(quantile_records, columns=QUANTILE_COLUMNS)
    regression_daily = pd.DataFrame(regression_records, columns=REGRESSION_COLUMNS)
    baseline_summary = {}
    for signal in signals:
        diagnostics = baseline_ic.loc[baseline_ic["signal"] == signal]
        baseline_summary[signal] = {
            **hac_mean_test(diagnostics["rank_ic"], lags),
            "status_counts": _status_counts(diagnostics),
        }
    summary = {
        "n_signal_dates": int(left["date"].nunique()),
        "n_factor_rows": len(left),
        "n_eligible_rows": int(left["eligible"].sum()),
        "n_labeled_eligible_rows": int(
            (merged["eligible"] & np.isfinite(merged["forward_return"])).sum()
        ),
        "effective_hac_lags": int(lags),
        "rank_ic": {
            **hac_mean_test(ic_daily["rank_ic"], lags),
            "status_counts": _status_counts(ic_daily),
        },
        "incremental_beta": {
            **hac_mean_test(regression_daily["beta_factor"], lags),
            "status_counts": _status_counts(regression_daily),
            "units": "standard deviations of forward return per standard deviation of factor",
        },
        "baselines": baseline_summary,
        "n_quantile_dates": int(quantile_returns["date"].nunique()),
        "quantile_method": "average-rank bins; equal scores are never split; empty bins allowed",
        "inference": "descriptive in-sample diagnostics; overlapping-return HAC; not causal evidence",
    }
    return {
        "ic_daily": ic_daily,
        "baseline_ic": baseline_ic,
        "quantile_returns": quantile_returns,
        "regression_daily": regression_daily,
        "summary": summary,
    }
