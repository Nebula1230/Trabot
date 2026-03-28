"""
HTML performance report generator for backtest results.

Produces a self-contained single-file HTML with:
  • Summary KPI cards (return, drawdown, Sharpe, Calmar, win rate, PF)
  • Equity curve chart (matplotlib, embedded as base64 PNG)
  • Drawdown chart
  • Monthly returns bar chart
  • Per-symbol breakdown table (multi-symbol runs)
  • Full trade log (last 200)
  • Walk-forward analysis table (when wf_result provided)

No external JS/CDN dependencies — the HTML is fully self-contained.
"""

import base64
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from .engine import BacktestResult
from .metrics import compute_metrics, compute_drawdown_series
from .walk_forward import WalkForwardResult

logger = logging.getLogger("BacktestReport")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    logger.warning("matplotlib not available — charts will be omitted from report")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    results: List[BacktestResult],
    output_path: str = "backtest_report.html",
    wf_result: Optional[WalkForwardResult] = None,
) -> str:
    """
    Generate a self-contained HTML performance report.

    Args:
        results:      List of BacktestResult (one per symbol; multi-symbol supported).
        output_path:  Path to write the HTML file.
        wf_result:    Optional WalkForwardResult to include a WF section.

    Returns:
        Absolute path to the generated HTML file.
    """
    all_metrics = [compute_metrics(r) for r in results]

    # Aggregate equity across symbols
    if len(results) > 1:
        min_len = min(len(r.equity_curve) for r in results)
        combined_equity = [
            sum(r.equity_curve[i] for r in results)
            for i in range(min_len)
        ]
        agg_dates = results[0].bar_dates[:min_len]
        total_initial = sum(r.initial_equity for r in results)
    else:
        combined_equity = results[0].equity_curve
        agg_dates = results[0].bar_dates
        total_initial = results[0].initial_equity

    sections: List[str] = []
    sections.append(_section_summary(all_metrics, combined_equity, total_initial))

    if _HAS_MPL:
        sections.append(_chart_equity(combined_equity, agg_dates, results[0].profile))
        sections.append(_chart_drawdown(combined_equity))
        sections.append(_chart_monthly(all_metrics))

    if len(results) > 1:
        sections.append(_section_per_symbol(all_metrics))

    sections.append(_section_trades(results))

    if wf_result:
        sections.append(_section_walk_forward(wf_result))

    html = _wrap_html(
        "\n".join(sections),
        results[0].profile,
        results[0].start_date,
        results[0].end_date,
    )
    output_path = str(output_path)
    Path(output_path).write_text(html, encoding="utf-8")
    logger.info(f"Report written to {output_path}")
    return output_path


def generate_json_report(
    results: List[BacktestResult],
    output_path: str = "backtest_report.json",
    wf_result: Optional[WalkForwardResult] = None,
) -> str:
    """
    Generate a machine-readable JSON performance report.

    Args:
        results:      List of BacktestResult (one per symbol).
        output_path:  Path to write the JSON file.
        wf_result:    Optional WalkForwardResult to include walk-forward data.

    Returns:
        Absolute path to the generated JSON file.
    """
    all_metrics = [compute_metrics(r) for r in results]

    # Aggregate equity
    if len(results) > 1:
        min_len = min(len(r.equity_curve) for r in results)
        combined_equity = [sum(r.equity_curve[i] for r in results) for i in range(min_len)]
        total_initial = sum(r.initial_equity for r in results)
    else:
        combined_equity = results[0].equity_curve
        total_initial = results[0].initial_equity

    total_return_pct = (combined_equity[-1] / total_initial - 1) * 100 if combined_equity else 0.0

    # Per-symbol detail
    symbols_data = []
    for r, m in zip(results, all_metrics):
        trades_list = []
        for t in r.trades:
            open_dt  = t.open_dt.isoformat()  if t.open_dt  else None
            close_dt = t.close_dt.isoformat() if t.close_dt else None
            trades_list.append({
                "entry_time":  open_dt,
                "exit_time":   close_dt,
                "direction":   t.direction,
                "entry_price": round(t.entry_price, 6),
                "exit_price":  round(t.exit_price,  6),
                "stop_loss":   round(t.stop_loss,   6),
                "take_profit": round(t.take_profit, 6),
                "quantity":    round(t.quantity, 4),
                "pnl":         round(t.pnl,  4),
                "pnl_r":       round(t.pnl_r, 4),
                "exit_reason": t.exit_reason,
                "confidence":  round(t.confidence, 4),
            })
        symbols_data.append({
            "symbol":           r.symbol,
            "metrics": {
                "total_trades":      m["total_trades"],
                "win_rate":          round(m["win_rate"], 4),
                "profit_factor":     round(m["profit_factor"], 4),
                "total_return_pct":  round(m["total_return_pct"], 4),
                "max_drawdown_pct":  round(m["max_drawdown_pct"], 4),
                "sharpe":            round(m["sharpe"], 4),
                "calmar":            round(m["calmar"], 4),
                "avg_win_r":         round(m.get("avg_win_r", 0), 4),
                "avg_loss_r":        round(m.get("avg_loss_r", 0), 4),
                "expectancy_r":      round(m.get("avg_pnl_r", 0), 4),
            },
            "trades": trades_list,
        })

    # Aggregate summary
    total_trades = sum(m["total_trades"] for m in all_metrics)
    avg_win_rate = sum(m["win_rate"] for m in all_metrics) / max(len(all_metrics), 1)
    avg_pf       = sum(m["profit_factor"] for m in all_metrics) / max(len(all_metrics), 1)
    avg_sharpe   = sum(m["sharpe"] for m in all_metrics) / max(len(all_metrics), 1)
    max_dd       = max(m["max_drawdown_pct"] for m in all_metrics)

    report: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "profile":      results[0].profile,
        "start_date":   results[0].start_date,
        "end_date":     results[0].end_date,
        "initial_equity": total_initial,
        "final_equity":   round(combined_equity[-1], 4) if combined_equity else total_initial,
        "summary": {
            "total_return_pct": round(total_return_pct, 4),
            "max_drawdown_pct": round(max_dd, 4),
            "total_trades":     total_trades,
            "avg_win_rate":     round(avg_win_rate, 4),
            "avg_profit_factor": round(avg_pf, 4),
            "avg_sharpe":       round(avg_sharpe, 4),
        },
        "symbols": symbols_data,
    }

    # Walk-forward section
    if wf_result:
        report["walk_forward"] = {
            "verdict":          wf_result.verdict,
            "avg_efficiency":   round(wf_result.avg_efficiency, 4),
            "windows": [
                {
                    "is_start":   w.is_start,
                    "is_end":     w.is_end,
                    "oos_start":  w.oos_start,
                    "oos_end":    w.oos_end,
                    "is_return":  round(w.is_return_pct, 4),
                    "oos_return": round(w.oos_return_pct, 4),
                    "efficiency": round(w.efficiency, 4),
                }
                for w in wf_result.windows
            ],
        }

    output_path = str(output_path)
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"JSON report written to {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────────────────

def _section_summary(
    metrics_list: List[Dict], equity: List[float], initial: float
) -> str:
    total_trades = sum(m["total_trades"]        for m in metrics_list)
    avg_win_rate = sum(m["win_rate"]            for m in metrics_list) / max(len(metrics_list), 1)
    avg_pf       = sum(m["profit_factor"]       for m in metrics_list) / max(len(metrics_list), 1)
    avg_sharpe   = sum(m["sharpe"]              for m in metrics_list) / max(len(metrics_list), 1)
    avg_calmar   = sum(m["calmar"]              for m in metrics_list) / max(len(metrics_list), 1)
    max_dd       = max(m["max_drawdown_pct"]    for m in metrics_list)
    total_return = (equity[-1] / initial - 1) * 100 if equity else 0.0

    def _card(label: str, value: str, sub: str = "", color: str = "#2196F3") -> str:
        return (
            f'<div class="card">'
            f'<div class="card-label">{label}</div>'
            f'<div class="card-value" style="color:{color}">{value}</div>'
            f'<div class="card-sub">{sub}</div>'
            f'</div>'
        )

    ret_color = "#4CAF50" if total_return >= 0 else "#f44336"
    dd_color  = "#f44336" if max_dd > 10 else "#FF9800" if max_dd > 5 else "#4CAF50"
    sh_color  = "#4CAF50" if avg_sharpe >= 1.0 else "#FF9800" if avg_sharpe >= 0.5 else "#f44336"

    cards = "".join([
        _card("Total Return",  f"{total_return:+.2f}%",    "",           ret_color),
        _card("Max Drawdown",  f"{max_dd:.2f}%",           "",           dd_color),
        _card("Sharpe Ratio",  f"{avg_sharpe:.2f}",       "annualised",  sh_color),
        _card("Calmar Ratio",  f"{avg_calmar:.2f}",       "",           "#9C27B0"),
        _card("Total Trades",  str(total_trades),          "",           "#607D8B"),
        _card("Win Rate",      f"{avg_win_rate:.1%}",      "",           "#2196F3"),
        _card("Profit Factor", f"{avg_pf:.2f}",            "",           "#00BCD4"),
    ])
    return f'<div class="cards-row">{cards}</div>'


def _chart_equity(equity: List[float], dates: List, profile: str) -> str:
    if not _HAS_MPL or not equity:
        return ""
    fig, ax = plt.subplots(figsize=(12, 4))
    xs = dates[:len(equity)] if dates else list(range(len(equity)))
    ax.plot(xs, equity, color="#2196F3", linewidth=1.5)
    ax.fill_between(xs, equity, min(equity), alpha=0.10, color="#2196F3")
    ax.set_title(f"Equity Curve — {profile.upper()} profile", fontsize=12)
    ax.set_ylabel("Equity ($)")
    if dates:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _fig_to_html(fig, "Equity Curve")


def _chart_drawdown(equity: List[float]) -> str:
    if not _HAS_MPL or len(equity) < 2:
        return ""
    dd, _, _ = compute_drawdown_series(np.array(equity))
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.fill_between(range(len(dd)), dd * 100, 0, color="#f44336", alpha=0.5)
    ax.set_title("Drawdown (%)", fontsize=11)
    ax.set_ylabel("DD %")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _fig_to_html(fig, "Drawdown")


def _chart_monthly(metrics_list: List[Dict]) -> str:
    if not _HAS_MPL:
        return ""
    combined: Dict[str, float] = {}
    for m in metrics_list:
        for k, v in m.get("monthly_returns", {}).items():
            combined[k] = combined.get(k, 0.0) + v / max(len(metrics_list), 1)
    if not combined:
        return ""

    keys   = sorted(combined)
    values = [combined[k] for k in keys]
    colors = ["#4CAF50" if v >= 0 else "#f44336" for v in values]

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.65), 3))
    ax.bar(keys, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Monthly Returns (%)", fontsize=11)
    ax.set_ylabel("%")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    return _fig_to_html(fig, "Monthly Returns")


def _section_per_symbol(metrics_list: List[Dict]) -> str:
    rows = ""
    for m in sorted(metrics_list, key=lambda x: x["total_return_pct"], reverse=True):
        ret = m["total_return_pct"]
        clr = "color:#4CAF50" if ret >= 0 else "color:#f44336"
        rows += (
            f"<tr>"
            f"<td>{m['symbol']}</td>"
            f"<td>{m['total_trades']}</td>"
            f"<td>{m['win_rate']:.1%}</td>"
            f"<td>{m['profit_factor']:.2f}</td>"
            f"<td style='{clr}'>{ret:+.2f}%</td>"
            f"<td>{m['max_drawdown_pct']:.2f}%</td>"
            f"<td>{m['sharpe']:.2f}</td>"
            f"<td>{m['avg_pnl_r']:+.3f}R</td>"
            f"<td>{m['tp_rate']:.1%}</td>"
            f"</tr>"
        )
    return (
        "<h2>Per-Symbol Breakdown</h2>"
        "<table><thead><tr>"
        "<th>Symbol</th><th>Trades</th><th>Win%</th><th>PF</th>"
        "<th>Return</th><th>MaxDD</th><th>Sharpe</th><th>Avg R</th><th>TP%</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _section_trades(results: List[BacktestResult]) -> str:
    all_trades = [t for r in results for t in r.trades]
    trades = sorted(all_trades, key=lambda t: t.close_bar, reverse=True)[:200]
    rows = ""
    for t in trades:
        clr = "color:#4CAF50" if t.pnl >= 0 else "color:#f44336"
        rows += (
            f"<tr>"
            f"<td>{t.symbol}</td>"
            f"<td>{'SHORT' if 'short' in str(t.direction).lower() else 'LONG'}</td>"
            f"<td>{t.entry_price:.5f}</td>"
            f"<td>{t.exit_price:.5f}</td>"
            f"<td>{t.stop_loss:.5f}</td>"
            f"<td>{t.take_profit:.5f}</td>"
            f"<td style='{clr}'>{t.pnl:+.2f}</td>"
            f"<td style='{clr}'>{t.pnl_r:+.2f}R</td>"
            f"<td>{t.exit_reason.upper()}</td>"
            f"<td>{t.win_probability:.0%}</td>"
            f"<td>{t.confidence:.2f}</td>"
            f"</tr>"
        )
    return (
        "<h2>Trade Log (last 200)</h2>"
        "<table><thead><tr>"
        "<th>Symbol</th><th>Dir</th><th>Entry</th><th>Exit</th>"
        "<th>SL</th><th>TP</th><th>P&amp;L</th><th>R</th>"
        "<th>Reason</th><th>Win%</th><th>Conf</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _section_walk_forward(wf: WalkForwardResult) -> str:
    verdict_color = {
        "pass":     "#4CAF50",
        "marginal": "#FF9800",
        "fail":     "#f44336",
    }.get(wf.verdict, "#9E9E9E")

    rows = ""
    for w in wf.windows:
        eff_clr = (
            "#4CAF50" if w.efficiency >= 0.5
            else "#FF9800" if w.efficiency >= 0.3
            else "#f44336"
        )
        oos_ret = w.oos_metrics.get("total_return_pct", 0.0)
        ret_clr = "color:#4CAF50" if oos_ret >= 0 else "color:#f44336"
        rows += (
            f"<tr>"
            f"<td>{w.window_num}</td>"
            f"<td>{w.is_start} → {w.is_end}</td>"
            f"<td>{w.oos_start} → {w.oos_end}</td>"
            f"<td>{w.is_metrics['sharpe']:.2f}</td>"
            f"<td>{w.oos_metrics['sharpe']:.2f}</td>"
            f"<td style='color:{eff_clr};font-weight:bold'>{w.efficiency:.2f}</td>"
            f"<td>{w.is_metrics['total_trades']}</td>"
            f"<td>{w.oos_metrics['total_trades']}</td>"
            f"<td style='{ret_clr}'>{oos_ret:+.2f}%</td>"
            f"</tr>"
        )

    return (
        f"<h2>Walk-Forward Analysis "
        f"<span style='color:{verdict_color};font-size:0.9em;margin-left:12px'>"
        f"{wf.verdict.upper()} (avg efficiency: {wf.avg_efficiency:.2f})"
        f"</span></h2>"
        f"<p style='color:#888;margin-top:0'>"
        f"IS={wf.is_months} months, OOS={wf.oos_months} month(s).  "
        f"Efficiency = OOS Sharpe ÷ IS Sharpe.  "
        f"≥0.50 = pass, ≥0.30 = marginal, &lt;0.30 = fail.</p>"
        "<table><thead><tr>"
        "<th>#</th><th>In-Sample</th><th>Out-of-Sample</th>"
        "<th>IS Sharpe</th><th>OOS Sharpe</th><th>Efficiency</th>"
        "<th>IS Trades</th><th>OOS Trades</th><th>OOS Return</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_html(fig, title: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return (
        f'<div class="chart">'
        f'<img src="data:image/png;base64,{b64}" alt="{title}" style="max-width:100%"/>'
        f'</div>'
    )


def _wrap_html(body: str, profile: str, start: str, end: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    css = """
    * { box-sizing:border-box; font-family:'Segoe UI',Arial,sans-serif; }
    body { background:#1a1a2e; color:#e0e0e0; margin:0; padding:20px; }
    h1 { color:#90CAF9; border-bottom:1px solid #333; padding-bottom:8px; }
    h2 { color:#90CAF9; margin-top:28px; }
    .cards-row { display:flex; flex-wrap:wrap; gap:12px; margin:20px 0; }
    .card { background:#16213e; border:1px solid #0f3460; border-radius:8px;
            padding:16px 20px; min-width:130px; }
    .card-label { font-size:.75em; color:#888; text-transform:uppercase; letter-spacing:1px; }
    .card-value { font-size:1.8em; font-weight:bold; margin:4px 0; }
    .card-sub { font-size:.7em; color:#888; }
    .chart { background:#16213e; border-radius:8px; padding:12px; margin:16px 0; }
    table { width:100%; border-collapse:collapse; margin:12px 0; font-size:.84em; }
    th { background:#0f3460; color:#90CAF9; padding:8px 10px; text-align:left; }
    td { padding:6px 10px; border-bottom:1px solid #1e2a45; }
    tr:hover td { background:#1e2a45; }
    """
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="UTF-8">\n'
        f'  <title>Backtest — {profile.upper()} — {start} to {end}</title>\n'
        f"  <style>{css}</style>\n"
        "</head>\n<body>\n"
        "  <h1>TradingAgents-v2 Backtest Report</h1>\n"
        f'  <p style="color:#888">Profile: <b style="color:#90CAF9">{profile.upper()}</b>'
        f" &nbsp;|&nbsp; Period: {start} → {end} &nbsp;|&nbsp; Generated: {ts}</p>\n"
        f"  {body}\n"
        "</body></html>"
    )
