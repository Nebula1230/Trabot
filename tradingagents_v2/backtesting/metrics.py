"""
Backtest performance metrics — pure functions over BacktestResult.

All functions are stateless: call them with a BacktestResult and get back
a plain dict.  No external dependencies beyond numpy and Python stdlib.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple

from .engine import BacktestResult, ClosedTrade


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(result: BacktestResult) -> Dict[str, Any]:
    """
    Compute the full suite of performance metrics from a BacktestResult.

    Returns a dict with keys:
      symbol, profile, start_date, end_date,
      total_trades, win_rate, profit_factor,
      total_return_pct, max_drawdown_pct, max_drawdown_duration_bars,
      sharpe, sortino, calmar,
      avg_pnl_r, avg_win_r, avg_loss_r, best_trade_r, worst_trade_r,
      avg_bars_held, avg_confidence,
      tp_rate, sl_rate,
      monthly_returns,
      equity_start, equity_end.
    """
    trades: List[ClosedTrade] = result.trades
    equity = np.array(result.equity_curve, dtype=float)
    total  = len(trades)
    wins   = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    win_rate      = len(wins) / total if total > 0 else 0.0
    gross_profit  = sum(t.pnl for t in wins)
    gross_loss    = abs(sum(t.pnl for t in losses)) or 1e-9
    profit_factor = gross_profit / gross_loss

    r_values = [t.pnl_r for t in trades]
    avg_pnl_r  = float(np.mean(r_values))     if r_values else 0.0
    avg_win_r  = float(np.mean([t.pnl_r for t in wins]))   if wins   else 0.0
    avg_loss_r = float(np.mean([t.pnl_r for t in losses])) if losses else 0.0
    best_r     = max(r_values, default=0.0)
    worst_r    = min(r_values, default=0.0)

    avg_bars = float(np.mean([t.close_bar - t.open_bar for t in trades])) if trades else 0.0
    avg_conf = float(np.mean([t.confidence for t in trades]))              if trades else 0.0

    tp_rate = len([t for t in trades if t.exit_reason == "tp"]) / max(total, 1)
    sl_rate = len([t for t in trades if t.exit_reason == "sl"]) / max(total, 1)

    total_return_pct = (equity[-1] / result.initial_equity - 1) * 100 if len(equity) > 0 else 0.0

    _, max_dd_pct, max_dd_bars = compute_drawdown_series(equity)
    sharpe  = _sharpe(equity, result)
    sortino = _sortino(equity, result)
    calmar  = _calmar(total_return_pct, max_dd_pct, result)
    monthly = _monthly_returns(result)

    return {
        "symbol":                     result.symbol,
        "profile":                    result.profile,
        "start_date":                 result.start_date,
        "end_date":                   result.end_date,
        "total_trades":               total,
        "win_rate":                   round(win_rate, 4),
        "profit_factor":              round(profit_factor, 3),
        "total_return_pct":           round(total_return_pct, 3),
        "max_drawdown_pct":           round(max_dd_pct, 3),
        "max_drawdown_duration_bars": max_dd_bars,
        "sharpe":                     round(sharpe, 3),
        "sortino":                    round(sortino, 3),
        "calmar":                     round(calmar, 3),
        "avg_pnl_r":                  round(avg_pnl_r, 3),
        "avg_win_r":                  round(avg_win_r, 3),
        "avg_loss_r":                 round(avg_loss_r, 3),
        "best_trade_r":               round(best_r, 3),
        "worst_trade_r":              round(worst_r, 3),
        "avg_bars_held":              round(avg_bars, 1),
        "avg_confidence":             round(avg_conf, 3),
        "tp_rate":                    round(tp_rate, 3),
        "sl_rate":                    round(sl_rate, 3),
        "equity_start":               round(result.initial_equity, 2),
        "equity_end":                 round(float(equity[-1]) if len(equity) > 0 else result.initial_equity, 2),
        "monthly_returns":            monthly,
    }


def compute_drawdown_series(
    equity: np.ndarray,
) -> Tuple[np.ndarray, float, int]:
    """
    Compute drawdown series from equity curve.

    Returns:
        dd_series:    fraction drawdown at each bar (negative values, e.g. -0.05 = -5%).
        max_dd_pct:   maximum drawdown as a positive percentage.
        max_dd_bars:  longest consecutive drawdown duration in bars.
    """
    if len(equity) == 0:
        return np.array([]), 0.0, 0

    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / np.maximum(running_max, 1e-9)
    max_dd_pct = float(abs(dd.min())) * 100

    # Longest consecutive period in drawdown
    in_dd = dd < -0.001
    max_dur = cur_dur = 0
    for v in in_dd:
        if v:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0

    return dd, max_dd_pct, max_dur


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bars_per_year(result: BacktestResult) -> float:
    """Estimate bars per year from the mid-tier timeframe config."""
    from ..data.loader import _BARS_PER_YEAR as BPY
    try:
        tfs = result.config.get("timeframes", {})
        mid_list = tfs.get("mid", ["1H"]) if tfs else ["1H"]
        mid_tf = mid_list[0] if mid_list else "1H"
    except Exception:
        mid_tf = "1H"
    return float(BPY.get(mid_tf, 6048))


def _sharpe(equity: np.ndarray, result: BacktestResult) -> float:
    """Sharpe ratio computed on DAILY equity returns (not per-bar).

    Using bar-level returns inflates Sharpe because ~23 of every 24 H1 bars
    are flat (no trade activity), driving the volatility denominator toward
    zero and making any positive mean look extraordinary.
    Daily aggregation matches how institutional desks compute the ratio.
    Minimum 2 daily returns so short backtests (e.g. 5 days → 4 returns) work.
    """
    daily = _daily_returns(equity, result)
    if len(daily) < 2:
        return 0.0
    std = daily.std(ddof=1)  # sample std; avoids divide-by-zero on 2-point series
    if std < 1e-10:
        return 0.0
    return float((daily.mean() / std) * np.sqrt(252))


def _daily_returns(equity: np.ndarray, result: BacktestResult) -> np.ndarray:
    """Aggregate bar-level equity to daily closes and compute daily returns."""
    if len(result.bar_dates) != len(equity) or len(equity) < 2:
        # Fallback: sample every N bars (N ≈ bars per day)
        bpy = _bars_per_year(result)
        step = max(1, int(round(bpy / 252)))
        sampled = equity[::step]
        if len(sampled) < 2:
            return np.diff(equity) / np.maximum(equity[:-1], 1e-9)
        return np.diff(sampled) / np.maximum(sampled[:-1], 1e-9)

    # Group equity points by calendar date, take last value per day
    from collections import defaultdict
    day_last: dict = defaultdict(float)
    for i, dt in enumerate(result.bar_dates):
        if i < len(equity):
            day_last[dt.date()] = float(equity[i])
    if not day_last:
        return np.array([])
    sorted_vals = [v for _, v in sorted(day_last.items())]
    arr = np.array(sorted_vals)
    return np.diff(arr) / np.maximum(arr[:-1], 1e-9)


def _sortino(equity: np.ndarray, result: BacktestResult) -> float:
    daily = _daily_returns(equity, result)
    if len(daily) < 2:
        return 0.0
    neg = daily[daily < 0]
    if len(neg) == 0 or neg.std() < 1e-10:
        return 10.0   # no losing days — cap at 10
    return float((daily.mean() / neg.std()) * np.sqrt(252))


def _calmar(total_return_pct: float, max_dd_pct: float,
            result: BacktestResult) -> float:
    """Annualised return divided by max drawdown."""
    if max_dd_pct < 1e-9:
        return 0.0
    try:
        start = datetime.strptime(result.start_date, "%Y-%m-%d")
        end   = datetime.strptime(result.end_date,   "%Y-%m-%d")
        years = max((end - start).days / 365.25, 1e-9)
    except Exception:
        years = 1.0
    return float((total_return_pct / years) / max_dd_pct)


def _monthly_returns(result: BacktestResult) -> Dict[str, float]:
    """Group trade P&L by calendar month, expressed as % of initial equity."""
    if not result.trades or not result.bar_dates:
        return {}
    bar_dates = result.bar_dates
    monthly: Dict[str, float] = {}
    for trade in result.trades:
        if trade.close_bar < len(bar_dates):
            key = bar_dates[trade.close_bar].strftime("%Y-%m")
            monthly[key] = monthly.get(key, 0.0) + trade.pnl
    return {
        k: round(v / max(result.initial_equity, 1.0) * 100, 3)
        for k, v in sorted(monthly.items())
    }
