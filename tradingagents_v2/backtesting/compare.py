"""Backtest vs Live trade comparison.

Loads live trades from ``closed_trades.jsonl``, matches them against
backtest ``ClosedTrade`` entries by (symbol, direction, entry-time
proximity), and produces a structured comparison suitable for console
output and HTML report embedding.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .engine import BacktestResult, ClosedTrade

logger = logging.getLogger("BacktestCompare")

# Maximum time gap (hours) between a backtest trade and a live trade
# for them to be considered "the same intended entry".
_MAX_MATCH_HOURS = 6.0


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradePair:
    """One matched (or unmatched) pair of backtest ↔ live trades."""

    bt: Optional[ClosedTrade] = None   # backtest side (None = live-only)
    live: Optional[ClosedTrade] = None  # live side (None = backtest-only)

    @property
    def symbol(self) -> str:
        return (self.bt or self.live).symbol  # type: ignore[union-attr]

    @property
    def matched(self) -> bool:
        return self.bt is not None and self.live is not None

    @property
    def pnl_delta(self) -> Optional[float]:
        """P&L difference (live − backtest) in account currency."""
        if self.matched:
            return self.live.pnl - self.bt.pnl  # type: ignore[union-attr]
        return None

    @property
    def pnl_r_delta(self) -> Optional[float]:
        if self.matched:
            return self.live.pnl_r - self.bt.pnl_r  # type: ignore[union-attr]
        return None

    @property
    def entry_price_delta(self) -> Optional[float]:
        if self.matched:
            return self.live.entry_price - self.bt.entry_price  # type: ignore[union-attr]
        return None

    @property
    def exit_price_delta(self) -> Optional[float]:
        if self.matched:
            return self.live.exit_price - self.bt.exit_price  # type: ignore[union-attr]
        return None


@dataclass
class ComparisonResult:
    """Full comparison between a backtest run and the live journal."""

    pairs: List[TradePair] = field(default_factory=list)
    bt_total: int = 0
    live_total: int = 0

    @property
    def matched_count(self) -> int:
        return sum(1 for p in self.pairs if p.matched)

    @property
    def bt_only_count(self) -> int:
        return sum(1 for p in self.pairs if p.bt and not p.live)

    @property
    def live_only_count(self) -> int:
        return sum(1 for p in self.pairs if p.live and not p.bt)

    @property
    def match_rate(self) -> float:
        total = max(self.bt_total, self.live_total, 1)
        return self.matched_count / total

    @property
    def avg_pnl_delta(self) -> float:
        deltas = [p.pnl_delta for p in self.pairs if p.pnl_delta is not None]
        return sum(deltas) / len(deltas) if deltas else 0.0

    @property
    def avg_pnl_r_delta(self) -> float:
        deltas = [p.pnl_r_delta for p in self.pairs if p.pnl_r_delta is not None]
        return sum(deltas) / len(deltas) if deltas else 0.0

    @property
    def exit_reason_agreement(self) -> float:
        matched = [p for p in self.pairs if p.matched]
        if not matched:
            return 0.0
        agree = sum(1 for p in matched if p.bt.exit_reason == p.live.exit_reason)  # type: ignore[union-attr]
        return agree / len(matched)

    @property
    def direction_agreement(self) -> float:
        matched = [p for p in self.pairs if p.matched]
        if not matched:
            return 0.0
        agree = sum(1 for p in matched if p.bt.direction == p.live.direction)  # type: ignore[union-attr]
        return agree / len(matched)

    @property
    def bt_pnl(self) -> float:
        return sum(p.bt.pnl for p in self.pairs if p.bt)

    @property
    def live_pnl(self) -> float:
        return sum(p.live.pnl for p in self.pairs if p.live)


# ─────────────────────────────────────────────────────────────────────────────
# Core matching logic
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def load_live_trades(
    log_dir: str,
    start_date: str,
    end_date: str,
) -> List[ClosedTrade]:
    """Load live trades from ``closed_trades.jsonl`` within a date range."""
    path = Path(log_dir) / "closed_trades.jsonl"
    if not path.exists():
        return []

    start = _parse_dt(start_date)
    end = _parse_dt(end_date)
    # Widen the window to catch trades that opened before start but closed during range
    if start:
        start = start - timedelta(days=1)
    if end:
        end = end + timedelta(days=1)

    trades: List[ClosedTrade] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            open_dt = _parse_dt(d.get("open_dt"))
            close_dt = _parse_dt(d.get("close_dt"))

            # Filter to date range
            trade_dt = open_dt or close_dt
            if trade_dt:
                if start and trade_dt < start:
                    continue
                if end and trade_dt > end:
                    continue

            trades.append(ClosedTrade(
                symbol=d["symbol"],
                direction=d["direction"],
                entry_price=d["entry_price"],
                exit_price=d["exit_price"],
                stop_loss=d["stop_loss"],
                take_profit=d["take_profit"],
                quantity=d["quantity"],
                risk_amount=d.get("risk_amount", 0.0),
                pnl=d["pnl"],
                pnl_r=d.get("pnl_r", 0.0),
                open_bar=0,
                close_bar=0,
                exit_reason=d["exit_reason"],
                confidence=d.get("confidence", 0.0),
                win_probability=d.get("win_probability", 0.0),
                agent_votes=d.get("agent_votes", {}),
                open_dt=open_dt,
                close_dt=close_dt,
            ))
    return trades


def compare_trades(
    bt_results: List[BacktestResult],
    live_trades: List[ClosedTrade],
    max_match_hours: float = _MAX_MATCH_HOURS,
) -> ComparisonResult:
    """Match backtest trades against live trades.

    Matching criteria (in order of priority):
      1. Same symbol
      2. Same direction
      3. Closest open_dt within ``max_match_hours``

    Each trade can match at most once (greedy nearest-first).
    """
    bt_trades = [t for r in bt_results for t in r.trades]
    result = ComparisonResult(bt_total=len(bt_trades), live_total=len(live_trades))

    if not bt_trades or not live_trades:
        # All unmatched
        for t in bt_trades:
            result.pairs.append(TradePair(bt=t))
        for t in live_trades:
            result.pairs.append(TradePair(live=t))
        return result

    # Build candidate pairs scored by time proximity
    max_delta = timedelta(hours=max_match_hours)
    used_bt: set = set()
    used_live: set = set()

    # Create (bt_idx, live_idx, time_gap) tuples sorted by gap
    candidates: List[Tuple[int, int, float]] = []
    for bi, bt in enumerate(bt_trades):
        if not bt.open_dt:
            continue
        for li, lv in enumerate(live_trades):
            if not lv.open_dt:
                continue
            if bt.symbol != lv.symbol:
                continue
            if bt.direction != lv.direction:
                continue
            gap = abs((bt.open_dt - lv.open_dt).total_seconds())
            if gap <= max_delta.total_seconds():
                candidates.append((bi, li, gap))

    # Greedy match: smallest gap first
    candidates.sort(key=lambda x: x[2])
    for bi, li, _ in candidates:
        if bi in used_bt or li in used_live:
            continue
        result.pairs.append(TradePair(bt=bt_trades[bi], live=live_trades[li]))
        used_bt.add(bi)
        used_live.add(li)

    # Unmatched
    for bi, bt in enumerate(bt_trades):
        if bi not in used_bt:
            result.pairs.append(TradePair(bt=bt))
    for li, lv in enumerate(live_trades):
        if li not in used_live:
            result.pairs.append(TradePair(live=lv))

    # Sort by time
    def _sort_key(p: TradePair) -> datetime:
        t = p.bt or p.live
        return t.open_dt or datetime.min  # type: ignore[union-attr]
    result.pairs.sort(key=_sort_key)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Console output
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(comp: ComparisonResult) -> None:
    """Print a formatted comparison summary to stdout."""
    sep = "═" * 80
    print(f"\n{sep}")
    print("  BACKTEST vs LIVE TRADE COMPARISON")
    print(sep)
    print(
        f"  Backtest trades: {comp.bt_total:4d}    "
        f"Live trades: {comp.live_total:4d}    "
        f"Matched: {comp.matched_count:4d}  "
        f"({comp.match_rate:.0%})"
    )
    print(
        f"  Backtest-only:   {comp.bt_only_count:4d}    "
        f"Live-only:   {comp.live_only_count:4d}"
    )
    if comp.matched_count > 0:
        print(f"\n  Avg P&L delta (live − BT):  {comp.avg_pnl_delta:+.2f}  "
              f"({comp.avg_pnl_r_delta:+.3f}R)")
        print(f"  Exit reason agreement:     {comp.exit_reason_agreement:.0%}")
        print(f"  Direction agreement:       {comp.direction_agreement:.0%}")
        print(f"  Total P&L — BT: {comp.bt_pnl:+.2f}   Live: {comp.live_pnl:+.2f}   "
              f"Δ: {comp.live_pnl - comp.bt_pnl:+.2f}")

    # Trade-by-trade table
    print(f"\n  {'#':>3}  {'Symbol':<10} {'Dir':<6} {'Open Date':<17} "
          f"{'Match':^7} {'BT P&L':>9} {'Live P&L':>9} {'Δ P&L':>9} "
          f"{'BT Exit':<12} {'Live Exit':<12}")
    print("  " + "─" * 100)
    for i, p in enumerate(comp.pairs, 1):
        sym = p.symbol
        _t = p.bt or p.live
        _dir = _t.direction[:5].upper()  # type: ignore[union-attr]
        _dt = _t.open_dt.strftime("%Y-%m-%d %H:%M") if _t.open_dt else "—"  # type: ignore[union-attr]

        if p.matched:
            _match = "  ✓  "
            _bt_pnl = f"{p.bt.pnl:+.2f}"  # type: ignore[union-attr]
            _lv_pnl = f"{p.live.pnl:+.2f}"  # type: ignore[union-attr]
            _delta = f"{p.pnl_delta:+.2f}"
            _bt_exit = p.bt.exit_reason  # type: ignore[union-attr]
            _lv_exit = p.live.exit_reason  # type: ignore[union-attr]
        elif p.bt:
            _match = "BT   "
            _bt_pnl = f"{p.bt.pnl:+.2f}"
            _lv_pnl = "—"
            _delta = "—"
            _bt_exit = p.bt.exit_reason
            _lv_exit = "—"
        else:
            _match = " LIVE"
            _bt_pnl = "—"
            _lv_pnl = f"{p.live.pnl:+.2f}"  # type: ignore[union-attr]
            _delta = "—"
            _bt_exit = "—"
            _lv_exit = p.live.exit_reason  # type: ignore[union-attr]

        print(f"  {i:3d}  {sym:<10} {_dir:<6} {_dt:<17} "
              f"{_match:^7} {_bt_pnl:>9} {_lv_pnl:>9} {_delta:>9} "
              f"{_bt_exit:<12} {_lv_exit:<12}")

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# HTML section for report embedding
# ─────────────────────────────────────────────────────────────────────────────

def comparison_html_section(comp: ComparisonResult) -> str:
    """Generate an HTML section for the comparison, embeddable in the report."""
    # Summary cards
    html = '<h2>Backtest vs Live Comparison</h2>\n'
    html += '<div class="kpi-row">\n'
    html += (
        f'<div class="kpi"><span class="kpi-val">{comp.matched_count}</span>'
        f'<span class="kpi-label">Matched</span></div>\n'
        f'<div class="kpi"><span class="kpi-val">{comp.bt_only_count}</span>'
        f'<span class="kpi-label">Backtest Only</span></div>\n'
        f'<div class="kpi"><span class="kpi-val">{comp.live_only_count}</span>'
        f'<span class="kpi-label">Live Only</span></div>\n'
        f'<div class="kpi"><span class="kpi-val">{comp.match_rate:.0%}</span>'
        f'<span class="kpi-label">Match Rate</span></div>\n'
    )
    if comp.matched_count > 0:
        html += (
            f'<div class="kpi"><span class="kpi-val">{comp.avg_pnl_delta:+.2f}</span>'
            f'<span class="kpi-label">Avg &Delta; P&amp;L</span></div>\n'
            f'<div class="kpi"><span class="kpi-val">{comp.exit_reason_agreement:.0%}</span>'
            f'<span class="kpi-label">Exit Agreement</span></div>\n'
        )
    html += '</div>\n'

    # Aggregate P&L comparison
    if comp.matched_count > 0:
        html += (
            f'<p style="margin:12px 0;font-size:14px;">'
            f'Total P&amp;L &mdash; '
            f'Backtest: <b>{comp.bt_pnl:+.2f}</b> &nbsp;|&nbsp; '
            f'Live: <b>{comp.live_pnl:+.2f}</b> &nbsp;|&nbsp; '
            f'&Delta;: <b>{comp.live_pnl - comp.bt_pnl:+.2f}</b></p>\n'
        )

    # Trade-by-trade table
    html += (
        '<table><thead><tr>'
        '<th>#</th><th>Symbol</th><th>Dir</th><th>Open</th>'
        '<th>Match</th>'
        '<th>BT Entry</th><th>Live Entry</th><th>&Delta; Entry</th>'
        '<th>BT P&amp;L</th><th>Live P&amp;L</th><th>&Delta; P&amp;L</th>'
        '<th>BT Exit</th><th>Live Exit</th>'
        '</tr></thead><tbody>\n'
    )
    for i, p in enumerate(comp.pairs, 1):
        _t = p.bt or p.live
        sym = _t.symbol  # type: ignore[union-attr]
        _dir = "SHORT" if "short" in str(_t.direction).lower() else "LONG"  # type: ignore[union-attr]
        _dt = _t.open_dt.strftime("%Y-%m-%d %H:%M") if _t.open_dt else "—"  # type: ignore[union-attr]

        if p.matched:
            _match_cls = "color:#4CAF50;font-weight:bold"
            _match = "✓"
            _bt_ep = f"{p.bt.entry_price:.5f}"  # type: ignore[union-attr]
            _lv_ep = f"{p.live.entry_price:.5f}"  # type: ignore[union-attr]
            _d_ep = f"{p.entry_price_delta:+.5f}"
            _clr = "color:#4CAF50" if p.pnl_delta >= 0 else "color:#f44336"  # type: ignore[operator]
            _bt_pnl = f"{p.bt.pnl:+.2f}"  # type: ignore[union-attr]
            _lv_pnl = f"{p.live.pnl:+.2f}"  # type: ignore[union-attr]
            _d_pnl = f"<span style='{_clr}'>{p.pnl_delta:+.2f}</span>"
            _bt_ex = p.bt.exit_reason.upper()  # type: ignore[union-attr]
            _lv_ex = p.live.exit_reason.upper()  # type: ignore[union-attr]
            _ex_style = "" if _bt_ex == _lv_ex else "color:#FF9800"
        elif p.bt:
            _match_cls = "color:#9E9E9E"
            _match = "BT"
            _bt_ep = f"{p.bt.entry_price:.5f}"
            _lv_ep = "—"
            _d_ep = "—"
            _bt_pnl = f"{p.bt.pnl:+.2f}"
            _lv_pnl = "—"
            _d_pnl = "—"
            _bt_ex = p.bt.exit_reason.upper()
            _lv_ex = "—"
            _ex_style = ""
        else:
            _match_cls = "color:#2196F3"
            _match = "LIVE"
            _bt_ep = "—"
            _lv_ep = f"{p.live.entry_price:.5f}"  # type: ignore[union-attr]
            _d_ep = "—"
            _bt_pnl = "—"
            _lv_pnl = f"{p.live.pnl:+.2f}"  # type: ignore[union-attr]
            _d_pnl = "—"
            _bt_ex = "—"
            _lv_ex = p.live.exit_reason.upper()  # type: ignore[union-attr]
            _ex_style = ""

        _lv_ex_td = f"<td style='{_ex_style}'>{_lv_ex}</td>" if _ex_style else f"<td>{_lv_ex}</td>"
        html += (
            f"<tr>"
            f"<td>{i}</td><td>{sym}</td><td>{_dir}</td><td>{_dt}</td>"
            f"<td style='{_match_cls}'>{_match}</td>"
            f"<td>{_bt_ep}</td><td>{_lv_ep}</td><td>{_d_ep}</td>"
            f"<td>{_bt_pnl}</td><td>{_lv_pnl}</td><td>{_d_pnl}</td>"
            f"<td>{_bt_ex}</td>{_lv_ex_td}"
            f"</tr>\n"
        )
    html += '</tbody></table>\n'
    return html


def comparison_json_section(comp: ComparisonResult) -> Dict[str, Any]:
    """Return a dict suitable for embedding in the JSON report."""
    pairs_data = []
    for p in comp.pairs:
        entry: Dict[str, Any] = {"matched": p.matched}
        if p.bt:
            entry["backtest"] = {
                "symbol": p.bt.symbol,
                "direction": p.bt.direction,
                "entry_price": round(p.bt.entry_price, 6),
                "exit_price": round(p.bt.exit_price, 6),
                "pnl": round(p.bt.pnl, 4),
                "pnl_r": round(p.bt.pnl_r, 4),
                "exit_reason": p.bt.exit_reason,
                "open_dt": p.bt.open_dt.isoformat() if p.bt.open_dt else None,
                "close_dt": p.bt.close_dt.isoformat() if p.bt.close_dt else None,
            }
        if p.live:
            entry["live"] = {
                "symbol": p.live.symbol,
                "direction": p.live.direction,
                "entry_price": round(p.live.entry_price, 6),
                "exit_price": round(p.live.exit_price, 6),
                "pnl": round(p.live.pnl, 4),
                "pnl_r": round(p.live.pnl_r, 4),
                "exit_reason": p.live.exit_reason,
                "open_dt": p.live.open_dt.isoformat() if p.live.open_dt else None,
                "close_dt": p.live.close_dt.isoformat() if p.live.close_dt else None,
            }
        if p.pnl_delta is not None:
            entry["pnl_delta"] = round(p.pnl_delta, 4)
            entry["pnl_r_delta"] = round(p.pnl_r_delta, 4)  # type: ignore[arg-type]
            entry["entry_price_delta"] = round(p.entry_price_delta, 6)  # type: ignore[arg-type]
        pairs_data.append(entry)

    return {
        "bt_total": comp.bt_total,
        "live_total": comp.live_total,
        "matched": comp.matched_count,
        "bt_only": comp.bt_only_count,
        "live_only": comp.live_only_count,
        "match_rate": round(comp.match_rate, 4),
        "avg_pnl_delta": round(comp.avg_pnl_delta, 4),
        "avg_pnl_r_delta": round(comp.avg_pnl_r_delta, 4),
        "exit_reason_agreement": round(comp.exit_reason_agreement, 4),
        "bt_total_pnl": round(comp.bt_pnl, 4),
        "live_total_pnl": round(comp.live_pnl, 4),
        "pairs": pairs_data,
    }
