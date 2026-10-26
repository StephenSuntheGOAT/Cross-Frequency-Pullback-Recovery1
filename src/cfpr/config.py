"""Explicit, validated research parameters; defaults are hypotheses, not fitted values."""

import math
import re
from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _integer(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class FactorConfig:
    trend_days: int = 20
    pullback_days: int = 4
    gap_days: int = 1
    volatility_days: int = 20
    tail_history_days: int = 20
    tail_minutes: int = 60
    market_timezone: str = "Asia/Shanghai"
    session_close: str = "15:00"
    volume_cap: float = 2.0
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        for name in ("trend_days", "pullback_days", "gap_days", "tail_minutes"):
            _integer(name, getattr(self, name), 1)
        for name in ("volatility_days", "tail_history_days"):
            _integer(name, getattr(self, name), 2)
        if not isinstance(self.session_close, str) or not re.fullmatch(
            r"\d{2}:\d{2}", self.session_close
        ):
            raise ValueError("session_close must be an HH:MM local time")
        try:
            close = time.fromisoformat(self.session_close)
        except (TypeError, ValueError) as exc:
            raise ValueError("session_close must be an HH:MM local time") from exc
        if close.second or close.microsecond or close.tzinfo is not None:
            raise ValueError("session_close must be a naive minute-aligned local time")
        if self.tail_minutes > close.hour * 60 + close.minute:
            raise ValueError("tail window must fit within the same local calendar day")
        try:
            ZoneInfo(self.market_timezone)
        except (ZoneInfoNotFoundError, TypeError) as exc:
            raise ValueError("market_timezone must be a valid IANA timezone") from exc
        for name in ("volume_cap", "epsilon"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")


@dataclass(frozen=True)
class BacktestConfig:
    horizon: int = 5
    top_fraction: float = 0.2
    cost_bps: float = 10.0
    min_cross_section: int = 10
    hac_lags: int | None = None

    def __post_init__(self) -> None:
        _integer("horizon", self.horizon, 1)
        _integer("min_cross_section", self.min_cross_section, 2)
        if not math.isfinite(self.top_fraction) or not 0 < self.top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1]")
        if not math.isfinite(self.cost_bps) or not 0 <= self.cost_bps < 10_000:
            raise ValueError("cost_bps must be finite and in [0, 10000)")
        if self.hac_lags is not None:
            _integer("hac_lags", self.hac_lags, 0)
