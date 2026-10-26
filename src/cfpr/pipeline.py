"""End-to-end research pipeline shared by the CLI and Python API."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .backtest import make_forward_returns, run_backtest
from .config import BacktestConfig, FactorConfig
from .data import validate_daily
from .factor import compute_factor
from .reporting import plot_diagnostics, write_json, write_manifest, write_report
from .statistics import analyze_predictions


def run_research(
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
    output: str | Path,
    factor_config: FactorConfig = FactorConfig(),
    backtest_config: BacktestConfig = BacktestConfig(),
    calendar=None,
    synthetic: bool = False,
    input_paths: list[Path] | None = None,
) -> dict:
    """Calculate, validate and write auditable artifacts after all checks finish."""
    daily = validate_daily(daily)
    factors = compute_factor(daily, intraday, factor_config, calendar)
    if not factors["eligible"].any():
        raise ValueError(
            "No eligible factor observations. Check warmup history, complete tail minutes, "
            "and nonzero reference volatility; no research results were written."
        )
    labels = make_forward_returns(daily, horizon=backtest_config.horizon, calendar=calendar)
    labeled = factors[["date", "symbol", "eligible"]].merge(
        labels[["date", "symbol", "forward_return"]], on=["date", "symbol"], validate="one_to_one"
    )
    if not (labeled["eligible"] & labeled["forward_return"].notna()).any():
        raise ValueError(
            "No eligible observations have complete forward returns. "
            "Add later sessions for the requested horizon; no research results were written."
        )
    stats = analyze_predictions(factors, labels, backtest_config)
    backtest = run_backtest(daily, factors, backtest_config, calendar)
    summary = {
        "synthetic": synthetic,
        "dataset": {
            "start": daily["date"].min().strftime("%Y-%m-%d"),
            "end": daily["date"].max().strftime("%Y-%m-%d"),
            "symbols": int(daily["symbol"].nunique()),
            "daily_rows": len(daily),
            "intraday_rows": len(intraday),
            "calendar_source": "explicit" if calendar is not None else "union_of_daily_dates",
        },
        "factor": {
            "total_rows": len(factors),
            "eligible_rows": int(factors["eligible"].sum()),
            "positive_rows": int(factors["factor"].gt(0).sum()),
            "valid_zero_rows": int(factors["factor"].eq(0).sum()),
            "unavailable_rows": int((~factors["eligible"]).sum()),
        },
        "prediction": stats["summary"],
        "backtest": backtest["summary"],
    }
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    frames = {"factors": factors, "forward_returns": labels}
    frames.update({name: frame for name, frame in stats.items() if isinstance(frame, pd.DataFrame)})
    frames.update(
        {name: frame for name, frame in backtest.items() if isinstance(frame, pd.DataFrame)}
    )
    for name, frame in frames.items():
        frame.to_csv(destination / f"{name}.csv", index=False)
    write_json(destination / "summary.json", summary)
    write_json(
        destination / "config.json",
        {
            "factor": asdict(factor_config),
            "backtest": asdict(backtest_config),
            "synthetic": synthetic,
        },
    )
    plot_diagnostics(destination, factors, stats, backtest, synthetic=synthetic)
    write_report(destination, summary)
    write_manifest(destination, input_paths or [])
    return summary
