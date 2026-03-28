"""
Walk-forward validator — out-of-sample performance verification.

Splits a full date range into rolling In-Sample / Out-of-Sample windows:

  [IS_start ─── IS_end] [OOS_start ─── OOS_end] → advance by oos_months → repeat

For each window:
  1. Run BacktestEngine on IS  → compute IS Sharpe
  2. Run BacktestEngine on OOS → compute OOS Sharpe
  3. efficiency = OOS_Sharpe / IS_Sharpe

A healthy strategy shows efficiency ≥ 0.50 (OOS captures ≥ 50% of IS edge).
Below 0.30 → likely overfit / no durable edge.
"""

import calendar
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .engine import BacktestEngine, BacktestResult, ClosedTrade
from .metrics import compute_metrics
from ..config.settings import TradingConfig
from ..data.loader import _MIN_BARS

logger = logging.getLogger("WalkForward")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WFWindow:
    window_num:  int
    is_start:    str
    is_end:      str
    oos_start:   str
    oos_end:     str
    is_metrics:  Dict[str, Any]
    oos_metrics: Dict[str, Any]
    efficiency:  float   # OOS Sharpe / IS Sharpe  (0 when IS Sharpe ≤ 0)


@dataclass
class WalkForwardResult:
    symbol:         str
    profile:        str
    full_start:     str
    full_end:       str
    is_months:      int
    oos_months:     int
    windows:        List[WFWindow]
    avg_efficiency: float         # mean across all windows
    verdict:        str           # "pass" / "marginal" / "fail"
    all_oos_trades: List[ClosedTrade] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardValidator:
    """
    Rolling walk-forward analysis for a single symbol.

    Args:
        config:     TradingConfig (determines profile, data, risk, timeframes).
        is_months:  Number of in-sample  months per window (default 3).
        oos_months: Number of out-of-sample months per window (default 1).
        initial_equity: Starting equity for each window's backtest.

    Verdict thresholds:
        efficiency ≥ 0.50  → "pass"      (edge generalises well)
        efficiency ≥ 0.30  → "marginal"  (some generalisation; watch closely)
        efficiency <  0.30 → "fail"      (likely overfit / no durable edge)
    """

    PASS_THRESHOLD     = 0.50
    MARGINAL_THRESHOLD = 0.30

    def __init__(
        self,
        config: TradingConfig,
        is_months: int = 6,
        oos_months: int = 2,
        initial_equity: float = 100_000.0,
    ):
        self.config         = config
        self.is_months      = is_months
        self.oos_months     = oos_months
        self.initial_equity = initial_equity
        self.engine         = BacktestEngine(config)

    def run(
        self,
        symbol: str,
        full_start: str,
        full_end: str,
    ) -> WalkForwardResult:
        """Run the full walk-forward analysis and return WalkForwardResult."""
        logger.info(
            f"Walk-forward [{symbol}] {full_start}→{full_end}  "
            f"IS={self.is_months}m OOS={self.oos_months}m"
        )

        windows = _build_windows(full_start, full_end, self.is_months, self.oos_months)
        if not windows:
            logger.error("Date range too short to build any WF windows")
            return WalkForwardResult(
                symbol=symbol, profile=self.config.profile,
                full_start=full_start, full_end=full_end,
                is_months=self.is_months, oos_months=self.oos_months,
                windows=[], avg_efficiency=0.0, verdict="fail",
            )

        # Pre-load bars ONCE for the entire date range (+ 100 day warm-up prepended).
        # This avoids repeated MT5 connect/disconnect between windows, and ensures
        # every window (including short 1-month OOS) has enough warm-up bars.
        _extended_start = (
            datetime.strptime(full_start, "%Y-%m-%d") - timedelta(days=100)
        ).strftime("%Y-%m-%d")
        logger.info(f"  Pre-loading bars {_extended_start}→{full_end} (incl. 100d warm-up)")
        all_bars_full, mid_tf, bar_dates_full = self.engine._preload_bars(
            symbol, _extended_start, full_end
        )
        if not all_bars_full or len(bar_dates_full) < _MIN_BARS + 10:
            logger.error(
                f"Failed to pre-load bars for {symbol} — "
                f"got {len(bar_dates_full)} bars, need {_MIN_BARS + 10}"
            )
            return WalkForwardResult(
                symbol=symbol, profile=self.config.profile,
                full_start=full_start, full_end=full_end,
                is_months=self.is_months, oos_months=self.oos_months,
                windows=[], avg_efficiency=0.0, verdict="fail",
            )

        wf_windows: List[WFWindow] = []
        all_oos_trades: List[ClosedTrade] = []

        for i, (is_s, is_e, oos_s, oos_e) in enumerate(windows, 1):
            logger.info(
                f"  Window {i}/{len(windows)}: "
                f"IS {is_s}→{is_e}  OOS {oos_s}→{oos_e}"
            )
            is_end_idx  = _find_end_bar(bar_dates_full, is_e)
            oos_end_idx = _find_end_bar(bar_dates_full, oos_e)

            is_result  = self.engine.run_with_bars(
                symbol, is_s, is_e,
                _slice_bars(all_bars_full, is_end_idx),
                mid_tf, bar_dates_full[:is_end_idx],
                self.initial_equity,
            )
            oos_result = self.engine.run_with_bars(
                symbol, oos_s, oos_e,
                _slice_bars(all_bars_full, oos_end_idx),
                mid_tf, bar_dates_full[:oos_end_idx],
                self.initial_equity,
            )

            is_m  = compute_metrics(is_result)
            oos_m = compute_metrics(oos_result)

            is_sharpe  = float(is_m.get("sharpe", 0.0))
            oos_sharpe = float(oos_m.get("sharpe", 0.0))

            if is_sharpe > 0.01:
                efficiency = oos_sharpe / is_sharpe
            elif oos_sharpe > 0:
                efficiency = 1.0   # IS was flat but OOS was positive → pass
            else:
                efficiency = 0.0

            wf_windows.append(WFWindow(
                window_num=i,
                is_start=is_s,   is_end=is_e,
                oos_start=oos_s, oos_end=oos_e,
                is_metrics=is_m, oos_metrics=oos_m,
                efficiency=round(efficiency, 3),
            ))
            all_oos_trades.extend(oos_result.trades)

        efficiencies = [w.efficiency for w in wf_windows]
        avg_eff = sum(efficiencies) / len(efficiencies) if efficiencies else 0.0

        if avg_eff >= self.PASS_THRESHOLD:
            verdict = "pass"
        elif avg_eff >= self.MARGINAL_THRESHOLD:
            verdict = "marginal"
        else:
            verdict = "fail"

        logger.info(
            f"Walk-forward done: {len(wf_windows)} windows  "
            f"avg_efficiency={avg_eff:.2f}  verdict={verdict}"
        )
        return WalkForwardResult(
            symbol=symbol, profile=self.config.profile,
            full_start=full_start, full_end=full_end,
            is_months=self.is_months, oos_months=self.oos_months,
            windows=wf_windows,
            avg_efficiency=round(avg_eff, 3),
            verdict=verdict,
            all_oos_trades=all_oos_trades,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_end_bar(bar_dates: List[datetime], date_str: str) -> int:
    """Return the first bar index whose date is strictly after *date_str*."""
    end_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    for i, d in enumerate(bar_dates):
        if d > end_dt:
            return i
    return len(bar_dates)


def _slice_bars(
    all_bars: Dict[str, Dict[str, np.ndarray]], end_idx: int
) -> Dict[str, Dict[str, np.ndarray]]:
    """Slice every timeframe's bar arrays to [:end_idx] (safe for short arrays)."""
    return {
        key: {field: arr[:end_idx] for field, arr in tfs.items()}
        for key, tfs in all_bars.items()
    }


def _add_months(dt: datetime, months: int) -> datetime:
    """Add *months* months to *dt*, clamping the day to the month end."""
    month = dt.month - 1 + months
    year  = dt.year + month // 12
    month = month % 12 + 1
    day   = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def _build_windows(
    start: str, end: str, is_months: int, oos_months: int
) -> List[Tuple[str, str, str, str]]:
    """
    Build list of (is_start, is_end, oos_start, oos_end) date strings.
    Stops when the OOS end would exceed *end*.
    """
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")

    windows: List[Tuple[str, str, str, str]] = []
    cursor = s
    while True:
        is_end  = _add_months(cursor, is_months)
        oos_end = _add_months(is_end,  oos_months)
        if oos_end > e:
            break
        windows.append((
            cursor.strftime("%Y-%m-%d"),
            is_end.strftime("%Y-%m-%d"),
            is_end.strftime("%Y-%m-%d"),
            oos_end.strftime("%Y-%m-%d"),
        ))
        cursor = _add_months(cursor, oos_months)

    return windows
