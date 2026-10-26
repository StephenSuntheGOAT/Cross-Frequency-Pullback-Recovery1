"""Portable, headless research reports and provenance records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def file_digest(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _markdown_metrics(data: dict, prefix: str = "") -> list[str]:
    rows: list[str] = []
    for key, value in data.items():
        label = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            rows.extend(_markdown_metrics(value, label))
            continue
        if isinstance(value, (list, tuple)):
            text = ", ".join(str(item) for item in value)
        elif isinstance(value, (float, np.floating)):
            text = f"{value:.6g}" if math.isfinite(value) else "未定义"
        elif value is None:
            text = "未定义"
        else:
            text = str(value)
        rows.append(f"| {label.replace('|', '/')} | {text.replace('|', '/')} |")
    return rows


def write_report(output: Path, summary: dict) -> None:
    synthetic = summary["synthetic"]
    origin = (
        "**本报告来自模拟数据，仅用于验证计算与回测流程，不能作为真实市场有效性的证据。**"
        if synthetic
        else "本报告使用用户提供的数据；结果取决于复权、交易日历及历史股票池的质量。"
    )
    text = [
        "# 跨频段回撤修复因子研究报告",
        "",
        origin,
        "",
        "收盘后计算信号；次一交易日开盘建仓，第 H 个交易日收盘平仓。",
        "预测检验每天计算；交易回测采用互不重叠的 H 日持有组合。",
        "交易费用按买入、卖出分别收取；基准按相同日程和费率持有等权股票池。",
        "HAC t 值是样本内研究统计量，不是因子有效性结论。",
        "",
        "![研究诊断图](diagnostics.png)",
        "",
        "## 数据与因子覆盖",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        *_markdown_metrics(summary["dataset"], "dataset"),
        *_markdown_metrics(summary["factor"], "factor"),
        "",
        "## 预测检验",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        *_markdown_metrics(summary["prediction"]),
        "",
        "## 扣费交易回测",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        *_markdown_metrics(summary["backtest"]),
        "",
        "## 解释与下一步",
        "",
        "- 有效的零分与缺失值分别处理；热身期和不完整尾盘不参与排名。",
        "- 普通动量、短期反转、日成交量和单独尾盘修复的对照见 baseline_ic.csv。",
        "- 增量回归控制前期趋势、短期反转、日相对成交量；它不包含市值与行业控制。",
        "- 应使用包含退市股票的时点股票池、真实交易日历，并另做样本外检验和费率敏感性分析。",
        "- 实盘还需逐笔核实涨跌停、停牌、最小交易单位及真实费用。本项目不实现完整撮合。",
        "",
        "配置见 config.json；输入及产物的 SHA-256 见 manifest.json。",
        "",
    ]
    (output / "report.md").write_text("\n".join(text), encoding="utf-8")


def plot_diagnostics(
    output: Path, factors: pd.DataFrame, stats: dict, backtest: dict, synthetic: bool = False
) -> None:
    # The explicit Agg backend works on CI/cloud machines without a display.
    # Generated reports keep their cache in the output tree, which is writable
    # even on machines with read-only home directories. Respect user overrides.
    os.environ.setdefault("MPLCONFIGDIR", str(output.resolve() / ".cache" / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output.resolve() / ".cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    equity = backtest["equity"]
    if not equity.empty:
        for column, label in [("strategy_nav", "Factor portfolio"), ("benchmark_nav", "Benchmark")]:
            axes[0, 0].plot(equity["date"], equity[column], label=label)
        axes[0, 0].legend()
    axes[0, 0].set_title("Net asset value (non-overlapping cohorts)")
    axes[0, 0].tick_params(axis="x", rotation=25)

    ic = stats["ic_daily"]
    if not ic.empty:
        axes[0, 1].plot(ic["date"], ic["rank_ic"], alpha=0.45, label="Daily Rank IC")
        axes[0, 1].plot(
            ic["date"], ic["rank_ic"].rolling(20, min_periods=5).mean(), label="20-session mean"
        )
        axes[0, 1].legend()
    axes[0, 1].axhline(0, color="grey", linewidth=0.7)
    axes[0, 1].set_title("Forward-return prediction")
    axes[0, 1].tick_params(axis="x", rotation=25)

    positive = factors.loc[factors["eligible"] & factors["factor"].gt(0), "factor"]
    if not positive.empty:
        axes[1, 0].hist(np.log1p(positive), bins=35, color="#4877a7")
    axes[1, 0].set_title("Positive scores: log(1 + F)")
    axes[1, 0].set_xlabel("Zeros and unavailable scores excluded")

    quantiles = stats["quantile_returns"]
    if not quantiles.empty:
        grouped = (
            quantiles.groupby("quantile")
            .agg(mean_return=("mean_forward_return", "mean"), dates=("date", "nunique"))
            .reindex(range(1, 6))
        )
        labels = [
            f"Q{number}\n{int(count) if pd.notna(count) else 0} dates"
            for number, count in grouped["dates"].items()
        ]
        axes[1, 1].bar(labels, grouped["mean_return"].to_numpy() * 100, color="#469d89")
    axes[1, 1].axhline(0, color="grey", linewidth=0.7)
    axes[1, 1].set_title("Gross forward return by rank bin (unequal coverage)")
    axes[1, 1].set_ylabel("Mean cohort return (%)")
    figure.suptitle(
        "CFPR — SYNTHETIC DATA / workflow verification only"
        if synthetic
        else "CFPR — user-supplied market data / research diagnostics"
    )
    figure.savefig(output / "diagnostics.png", dpi=160)
    plt.close(figure)


def write_manifest(output: Path, inputs: list[Path]) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "inputs": [file_digest(path) for path in inputs],
        "outputs": [
            file_digest(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        ],
    }
    write_json(output / "manifest.json", manifest)
