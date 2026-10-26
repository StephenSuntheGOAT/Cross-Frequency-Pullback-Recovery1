"""Exercise installed entry points, artifacts, provenance and data round-trips."""

import hashlib
import json
import subprocess
import sys

import pandas as pd
import pytest

from cfpr.cli import main


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    output = tmp_path_factory.mktemp("demo")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "cfpr",
            "demo",
            "--output",
            str(output),
            "--symbols",
            "14",
            "--days",
            "55",
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return output, process


def test_demo_writes_meaningful_results_and_auditable_artifacts(demo):
    output, process = demo
    assert "SYNTHETIC DATA" in process.stdout
    assert "Read-only file system" not in process.stderr
    expected = {
        "factors.csv",
        "forward_returns.csv",
        "ic_daily.csv",
        "baseline_ic.csv",
        "quantile_returns.csv",
        "regression_daily.csv",
        "equity.csv",
        "trades.csv",
        "summary.json",
        "config.json",
        "manifest.json",
        "report.md",
        "diagnostics.png",
    }
    assert expected <= {path.name for path in output.iterdir()}
    summary = json.loads((output / "summary.json").read_text())
    assert summary["synthetic"] is True
    assert summary["factor"]["eligible_rows"] == 14 * 30
    assert summary["factor"]["positive_rows"] > 0
    assert summary["prediction"]["rank_ic"]["n_obs"] > 0
    assert summary["prediction"]["incremental_beta"]["n_obs"] > 0
    assert summary["prediction"]["effective_hac_lags"] == 4
    assert summary["backtest"]["cohorts"] == 5
    assert (output / "diagnostics.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "模拟数据" in (output / "report.md").read_text()
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["inputs"]) == 3
    for artifact in manifest["inputs"] + manifest["outputs"]:
        with open(artifact["path"], "rb") as stream:
            assert hashlib.sha256(stream.read()).hexdigest() == artifact["sha256"]
    # JSON artifacts must be interoperable: Python's nonstandard NaN isn't allowed.
    for name in ("summary.json", "config.json", "manifest.json"):
        json.loads((output / name).read_text(), parse_constant=lambda value: pytest.fail(value))


def test_csv_input_and_factor_only_command_match_demo(demo, tmp_path):
    output, _ = demo
    destination = tmp_path / "factor.parquet"
    assert (
        main(
            [
                "factor",
                "--daily",
                str(output / "data/daily.csv"),
                "--intraday",
                str(output / "data/intraday.csv"),
                "--calendar",
                str(output / "data/calendar.csv"),
                "--output",
                str(destination),
            ]
        )
        == 0
    )
    actual = pd.read_parquet(destination)
    expected = pd.read_csv(output / "factors.csv", parse_dates=["date"])
    pd.testing.assert_series_equal(actual["factor"], expected["factor"], rtol=1e-9, atol=1e-11)
    pd.testing.assert_series_equal(actual["eligible"], expected["eligible"])
    assert not (tmp_path / "forward_returns.csv").exists()


def test_parquet_input_full_pipeline_matches_demo(demo, tmp_path):
    output, _ = demo
    for name in ("daily", "intraday"):
        frame = pd.read_csv(output / f"data/{name}.csv", dtype={"symbol": str})
        frame.to_parquet(tmp_path / f"{name}.parquet", index=False)
    destination = tmp_path / "research"
    assert (
        main(
            [
                "run",
                "--daily",
                str(tmp_path / "daily.parquet"),
                "--intraday",
                str(tmp_path / "intraday.parquet"),
                "--output",
                str(destination),
            ]
        )
        == 0
    )
    actual = pd.read_csv(destination / "equity.csv")
    expected = pd.read_csv(output / "equity.csv")
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-9, atol=1e-11)
    assert json.loads((destination / "summary.json").read_text())["synthetic"] is False


def test_invalid_data_fails_before_writing_research(demo, tmp_path, capsys):
    output, _ = demo
    daily = pd.read_csv(output / "data/daily.csv")
    daily = pd.concat([daily, daily.iloc[:1]])
    path = tmp_path / "bad.csv"
    daily.to_csv(path, index=False)
    destination = tmp_path / "bad_results"
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--daily",
                str(path),
                "--intraday",
                str(output / "data/intraday.csv"),
                "--output",
                str(destination),
            ]
        )
    assert exc.value.code == 2
    assert "duplicate" in capsys.readouterr().err
    assert not destination.exists()


def test_too_short_demo_reports_no_validation_instead_of_empty_success(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["demo", "--output", str(tmp_path / "short"), "--days", "10", "--symbols", "3"])
    assert exc.value.code == 2
    assert "No eligible" in capsys.readouterr().err
    assert not (tmp_path / "short/summary.json").exists()
