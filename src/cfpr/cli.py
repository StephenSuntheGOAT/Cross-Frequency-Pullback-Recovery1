"""Command-line interface; `cfpr demo` works entirely offline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import __version__
from .config import BacktestConfig, FactorConfig
from .data import load_daily, load_intraday
from .factor import compute_factor
from .pipeline import run_research
from .synthetic import generate_synthetic_data


def _factor_options(parser: argparse.ArgumentParser) -> None:
    defaults = FactorConfig()
    for name in (
        "trend_days",
        "pullback_days",
        "gap_days",
        "volatility_days",
        "tail_history_days",
        "tail_minutes",
    ):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name)
        )
    parser.add_argument("--market-timezone", default=defaults.market_timezone)
    parser.add_argument("--session-close", default=defaults.session_close)
    parser.add_argument("--volume-cap", type=float, default=defaults.volume_cap)


def _research_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--horizon", type=int, default=5, help="Holding sessions; exit at Hth close"
    )
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument(
        "--cost-bps", type=float, default=10.0, help="Basis points charged on EACH side"
    )
    parser.add_argument("--min-cross-section", type=int, default=10)
    parser.add_argument("--hac-lags", type=int, default=None, help="At least horizon-1 is enforced")


def _generation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbols", type=int, default=40)
    parser.add_argument("--days", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", default="2023-01-02", help="Synthetic weekday calendar start")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-frequency pullback-recovery factor research"
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "generate", "run", "factor"):
        command = commands.add_parser(name)
        command.add_argument(
            "--output", type=Path, required=name != "demo", default=Path("outputs/demo")
        )
        _factor_options(command)
        if name in ("demo", "run"):
            _research_options(command)
        if name in ("demo", "generate"):
            _generation_options(command)
        else:
            command.add_argument("--daily", type=Path, required=True)
            command.add_argument("--intraday", type=Path, required=True)
            command.add_argument("--calendar", type=Path, help="CSV or Parquet with a date column")
    return parser


def _factor_config(args: argparse.Namespace) -> FactorConfig:
    return FactorConfig(
        **{
            name: getattr(args, name)
            for name in FactorConfig.__dataclass_fields__
            if hasattr(args, name)
        }
    )


def _calendar(path: Path | None):
    if path is None:
        return None
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
    elif path.suffix.lower() in (".parquet", ".pq"):
        table = pd.read_parquet(path)
    else:
        raise ValueError("Calendar must be a CSV or Parquet file with a date column")
    if "date" not in table:
        raise ValueError("Calendar is missing its date column")
    return table["date"].tolist()


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        factor_config = _factor_config(args)
        inputs: list[Path] = []
        calendar = None
        if args.command in ("demo", "generate"):
            daily, intraday = generate_synthetic_data(
                symbols=args.symbols,
                days=args.days,
                seed=args.seed,
                start=args.start,
                config=factor_config,
            )
            directory = args.output / "data" if args.command == "demo" else args.output
            directory.mkdir(parents=True, exist_ok=True)
            daily.to_csv(directory / "daily.csv", index=False)
            intraday.to_csv(directory / "intraday.csv", index=False)
            pd.DataFrame({"date": daily["date"].drop_duplicates()}).to_csv(
                directory / "calendar.csv", index=False
            )
            inputs = [directory / name for name in ("daily.csv", "intraday.csv", "calendar.csv")]
            if args.command == "generate":
                print(
                    f"Synthetic data saved to {directory.resolve()} (no empirical alpha implied)."
                )
                return 0
            calendar = daily["date"].drop_duplicates().tolist()
        else:
            daily = load_daily(args.daily)
            intraday = load_intraday(args.intraday, factor_config)
            calendar = _calendar(args.calendar)
            inputs = [args.daily, args.intraday]
            if args.calendar is not None:
                inputs.append(args.calendar)
        if args.command == "factor":
            factors = compute_factor(daily, intraday, factor_config, calendar)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.suffix.lower() in (".parquet", ".pq"):
                factors.to_parquet(args.output, index=False)
            elif args.output.suffix.lower() == ".csv":
                factors.to_csv(args.output, index=False)
            else:
                raise ValueError("Factor output must end in .csv, .parquet or .pq")
            print(f"Factor rows: {len(factors)}; eligible: {int(factors['eligible'].sum())}")
            return 0
        config = BacktestConfig(
            **{name: getattr(args, name) for name in BacktestConfig.__dataclass_fields__}
        )
        summary = run_research(
            daily,
            intraday,
            args.output,
            factor_config,
            config,
            calendar=calendar,
            synthetic=args.command == "demo",
            input_paths=inputs,
        )
        print(f"Research artifacts: {args.output.resolve()}")
        print(
            f"Eligible rows: {summary['factor']['eligible_rows']}; "
            f"positive scores: {summary['factor']['positive_rows']}"
        )
        if args.command == "demo":
            print("SYNTHETIC DATA: results verify the workflow, not market effectiveness.")
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"cfpr: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
