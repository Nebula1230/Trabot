"""
Structured debug tracer for the backtest pipeline.

Captures one record per pipeline invocation (per bar that fires the agent loop)
and one record per exit event.  When the backtest finishes the tracer prints a
human-readable summary to stderr and optionally writes a JSON trace file.

Activation
----------
CLI:   python -m tradingagents_v2.backtesting --debug ...
Code:  tracer = DebugTracer(enabled=True)  # pass to BacktestEngine
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DecisionTrace:
    """One pipeline pass (one bar where agents ran)."""

    bar_idx: int = 0
    bar_time: str = ""
    symbol: str = ""

    # Fusion
    dir_long: float = 0.0
    dir_mid: float = 0.0
    dir_short: float = 0.0
    csns_long: float = 0.0
    csns_mid: float = 0.0
    csns_short: float = 0.0
    align_str: float = 0.0
    breadth: float = 0.0
    regime_trendiness: float = 0.0

    # Alignment
    aligned: bool = False
    alignment_type: str = ""       # "full", "pullback-entry", ""
    alignment_direction: str = ""  # "bullish", "bearish"
    alignment_block_reason: str = ""

    # Recipe
    recipe_name: str = ""
    win_prob: float = 0.0
    expected_value: float = 0.0
    risk_reward: float = 0.0

    # Risk check
    risk_passed: bool = False
    risk_block_reason: str = ""

    # Consensus
    votes_for: int = 0
    votes_total: int = 0

    # Plan (SL/TP sizing)
    atr_for_sizing: float = 0.0
    atr_long: float = 0.0
    atr_mid: float = 0.0
    atr_short: float = 0.0
    sl_atr_mult: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    stop_distance_pips: float = 0.0
    tp_distance_pips: float = 0.0
    actual_rr: float = 0.0
    pivot_nearest_support: float = 0.0
    pivot_nearest_resist: float = 0.0
    pivot_s1: float = 0.0
    pivot_s2: float = 0.0
    pivot_r1: float = 0.0
    pivot_r2: float = 0.0
    swing_highs: str = ""
    swing_lows: str = ""
    quantity: float = 0.0
    risk_amount: float = 0.0

    # Execution
    executed: bool = False

    # Agent votes (compact)
    agent_votes: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExitTrace:
    """One position exit."""

    bar_idx: int = 0
    bar_time: str = ""
    symbol: str = ""
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_r: float = 0.0
    bars_held: int = 0


class DebugTracer:
    """Collects decision and exit traces during a backtest run."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.decisions: List[DecisionTrace] = []
        self.exits: List[ExitTrace] = []
        self._current: Optional[DecisionTrace] = None

    # ── Recording API (called from graph.py and engine.py) ──────────────

    def begin_bar(self, bar_idx: int, bar_time: str, symbol: str) -> None:
        if not self.enabled:
            return
        self._current = DecisionTrace(
            bar_idx=bar_idx, bar_time=bar_time, symbol=symbol,
        )

    def record_fusion(self, fusion) -> None:
        """Record TimeframeFusion dataclass fields."""
        if not self.enabled or self._current is None:
            return
        c = self._current
        c.dir_long = round(fusion.dir_long, 5)
        c.dir_mid = round(fusion.dir_mid, 5)
        c.dir_short = round(fusion.dir_short, 5)
        c.csns_long = round(fusion.conf_long, 3)
        c.csns_mid = round(fusion.conf_mid, 3)
        c.csns_short = round(fusion.conf_short, 3)
        c.align_str = round(fusion.alignment_strength, 3)
        c.breadth = round(fusion.breadth_score, 4)
        c.regime_trendiness = round(fusion.regime_trendiness, 3)

    def record_alignment(self, aligned: bool, atype: str = "",
                         direction: str = "", block_reason: str = "") -> None:
        if not self.enabled or self._current is None:
            return
        c = self._current
        c.aligned = aligned
        c.alignment_type = atype
        c.alignment_direction = direction
        c.alignment_block_reason = block_reason

    def record_recipe(self, recipe) -> None:
        if not self.enabled or self._current is None:
            return
        c = self._current
        c.recipe_name = recipe.name
        c.win_prob = round(recipe.win_probability, 4)
        c.expected_value = round(recipe.expected_value, 4)
        c.risk_reward = round(recipe.risk_reward_ratio, 3)

    def record_risk(self, passed: bool, block_reason: str = "") -> None:
        if not self.enabled or self._current is None:
            return
        c = self._current
        c.risk_passed = passed
        c.risk_block_reason = block_reason

    def record_consensus(self, votes_for: int, votes_total: int) -> None:
        if not self.enabled or self._current is None:
            return
        self._current.votes_for = votes_for
        self._current.votes_total = votes_total

    def record_plan(self, *, entry: float, sl: float, tp: float,
                    atr_sizing: float, atr_long: float, atr_mid: float,
                    atr_short: float, sl_mult: float,
                    pivots: Optional[Dict] = None,
                    swing_highs: Optional[List] = None,
                    swing_lows: Optional[List] = None,
                    quantity: float = 0.0, risk_amount: float = 0.0,
                    pip_size: float = 0.0001) -> None:
        if not self.enabled or self._current is None:
            return
        c = self._current
        c.entry_price = round(entry, 5)
        c.stop_loss = round(sl, 5)
        c.take_profit = round(tp, 5)
        c.atr_for_sizing = round(atr_sizing, 6)
        c.atr_long = round(atr_long, 6)
        c.atr_mid = round(atr_mid, 6)
        c.atr_short = round(atr_short, 6)
        c.sl_atr_mult = round(sl_mult, 2)
        c.quantity = round(quantity, 4)
        c.risk_amount = round(risk_amount, 2)

        sd = abs(entry - sl)
        td = abs(tp - entry)
        c.stop_distance_pips = round(sd / pip_size, 1) if pip_size else 0.0
        c.tp_distance_pips = round(td / pip_size, 1) if pip_size else 0.0
        c.actual_rr = round(td / sd, 2) if sd > 0 else 0.0

        if pivots:
            c.pivot_nearest_support = round(pivots.get("nearest_support", 0), 5)
            c.pivot_nearest_resist = round(pivots.get("nearest_resist", 0), 5)
            c.pivot_s1 = round(pivots.get("s1", 0), 5)
            c.pivot_s2 = round(pivots.get("s2", 0), 5)
            c.pivot_r1 = round(pivots.get("r1", 0), 5)
            c.pivot_r2 = round(pivots.get("r2", 0), 5)
        if swing_highs:
            c.swing_highs = ",".join(f"{h:.5f}" for h in swing_highs[:5])
        if swing_lows:
            c.swing_lows = ",".join(f"{l:.5f}" for l in swing_lows[:5])

    def record_agent_votes(self, outputs: Dict[str, Any]) -> None:
        if not self.enabled or self._current is None:
            return
        for name, out in outputs.items():
            if hasattr(out, "dir_score"):
                self._current.agent_votes[name] = round(float(out.dir_score), 4)

    def record_executed(self, executed: bool) -> None:
        if not self.enabled or self._current is None:
            return
        self._current.executed = executed

    def commit_bar(self) -> None:
        """Finalize the current bar's trace and add it to the list."""
        if not self.enabled or self._current is None:
            return
        self.decisions.append(self._current)
        self._current = None

    def record_exit(self, *, bar_idx: int, bar_time: str, symbol: str,
                    direction: str, entry: float, exit_price: float,
                    sl: float, tp: float, reason: str,
                    pnl: float, pnl_r: float, bars_held: int = 0) -> None:
        if not self.enabled:
            return
        self.exits.append(ExitTrace(
            bar_idx=bar_idx, bar_time=bar_time, symbol=symbol,
            direction=direction, entry_price=round(entry, 5),
            exit_price=round(exit_price, 5),
            stop_loss=round(sl, 5), take_profit=round(tp, 5),
            exit_reason=reason, pnl=round(pnl, 2),
            pnl_r=round(pnl_r, 3), bars_held=bars_held,
        ))

    # ── Output ───────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Write a human-readable trace to stderr."""
        if not self.enabled:
            return
        out = sys.stderr
        out.write("\n" + "=" * 90 + "\n")
        out.write("  DEBUG TRACE SUMMARY\n")
        out.write("=" * 90 + "\n\n")

        # Decisions
        n_aligned = sum(1 for d in self.decisions if d.aligned)
        n_recipe = sum(1 for d in self.decisions if d.recipe_name)
        n_risk = sum(1 for d in self.decisions if d.risk_passed)
        n_exec = sum(1 for d in self.decisions if d.executed)
        out.write(f"Pipeline: {len(self.decisions)} agent runs → "
                  f"{n_aligned} aligned → {n_recipe} recipes → "
                  f"{n_risk} risk-passed → {n_exec} executed\n")

        # Block reasons
        block_reasons: Dict[str, int] = {}
        for d in self.decisions:
            if d.alignment_block_reason:
                block_reasons[d.alignment_block_reason] = block_reasons.get(
                    d.alignment_block_reason, 0) + 1
            if d.risk_block_reason:
                block_reasons[d.risk_block_reason] = block_reasons.get(
                    d.risk_block_reason, 0) + 1
        if block_reasons:
            out.write("\nBlock reasons:\n")
            for reason, count in sorted(block_reasons.items(), key=lambda x: -x[1]):
                out.write(f"  {count:3d}× {reason}\n")

        # Decision detail table
        out.write("\n--- Decision Detail ---\n")
        out.write(f"{'Bar':>6} {'Time':>12} {'L→':>7} {'M→':>7} {'S→':>7} "
                  f"{'AlStr':>5} {'Algn':>15} {'Recipe':>8} "
                  f"{'WinP':>5} {'EV':>6} {'Risk':>6} "
                  f"{'Entry':>9} {'SL':>9} {'TP':>9} {'SLpip':>6} {'R:R':>5} "
                  f"{'ATRsz':>8} {'Exec':>4}\n")
        for d in self.decisions:
            algn = (d.alignment_type[:12] if d.aligned
                    else d.alignment_block_reason[:12] if d.alignment_block_reason
                    else "—")
            recipe = d.recipe_name[-8:] if d.recipe_name else "—"
            risk = "PASS" if d.risk_passed else (d.risk_block_reason[:6] if d.risk_block_reason else "—")
            entry = f"{d.entry_price:.5f}" if d.entry_price else "—"
            sl = f"{d.stop_loss:.5f}" if d.stop_loss else "—"
            tp = f"{d.take_profit:.5f}" if d.take_profit else "—"
            out.write(
                f"{d.bar_idx:6d} {d.bar_time:>12} "
                f"{d.dir_long:+.4f} {d.dir_mid:+.4f} {d.dir_short:+.4f} "
                f"{d.align_str:5.3f} {algn:>15} {recipe:>8} "
                f"{d.win_prob:5.3f} {d.expected_value:+.3f} {risk:>6} "
                f"{entry:>9} {sl:>9} {tp:>9} {d.stop_distance_pips:6.1f} "
                f"{d.actual_rr:5.2f} {d.atr_for_sizing:8.6f} "
                f"{'✓' if d.executed else '—':>4}\n"
            )

        # SL/TP sizing detail for executed trades
        executed = [d for d in self.decisions if d.executed]
        if executed:
            out.write("\n--- SL/TP Sizing Detail (executed trades) ---\n")
            for d in executed:
                out.write(f"\n  Bar {d.bar_idx} [{d.bar_time}] {d.symbol} "
                          f"{d.alignment_direction.upper()}\n")
                out.write(f"    ATR: long={d.atr_long:.6f}  mid={d.atr_mid:.6f}  "
                          f"short={d.atr_short:.6f}  → sizing={d.atr_for_sizing:.6f}\n")
                out.write(f"    Entry={d.entry_price:.5f}  SL={d.stop_loss:.5f}  "
                          f"TP={d.take_profit:.5f}\n")
                out.write(f"    SL dist={d.stop_distance_pips:.1f} pips  "
                          f"TP dist={d.tp_distance_pips:.1f} pips  "
                          f"R:R={d.actual_rr:.2f}\n")
                out.write(f"    Pivots: S2={d.pivot_s2:.5f} S1={d.pivot_s1:.5f} "
                          f"R1={d.pivot_r1:.5f} R2={d.pivot_r2:.5f}\n")
                out.write(f"    Swing H: {d.swing_highs or '—'}\n")
                out.write(f"    Swing L: {d.swing_lows or '—'}\n")
                out.write(f"    Qty={d.quantity:.4f}  Risk=${d.risk_amount:.2f}  "
                          f"WinP={d.win_prob:.3f}  EV={d.expected_value:+.3f}\n")
                if d.agent_votes:
                    votes = "  ".join(f"{k}:{v:+.3f}" for k, v in
                                      sorted(d.agent_votes.items(),
                                             key=lambda x: abs(x[1]), reverse=True))
                    out.write(f"    Votes: {votes}\n")

        # Exit detail
        if self.exits:
            out.write("\n--- Exit Detail ---\n")
            out.write(f"{'Bar':>6} {'Time':>12} {'Dir':>5} {'Entry':>9} "
                      f"{'SL':>9} {'TP':>9} {'Exit':>9} {'Reason':>10} "
                      f"{'PnL':>8} {'R':>6} {'Held':>5}\n")
            for e in self.exits:
                out.write(
                    f"{e.bar_idx:6d} {e.bar_time:>12} {e.direction:>5} "
                    f"{e.entry_price:9.5f} {e.stop_loss:9.5f} {e.take_profit:9.5f} "
                    f"{e.exit_price:9.5f} {e.exit_reason:>10} "
                    f"{e.pnl:+8.2f} {e.pnl_r:+6.3f} {e.bars_held:5d}\n"
                )

        out.write("\n" + "=" * 90 + "\n")
        out.flush()

    def save_json(self, path: str) -> None:
        """Write the full trace to a JSON file."""
        if not self.enabled:
            return
        data = {
            "decisions": [asdict(d) for d in self.decisions],
            "exits": [asdict(e) for e in self.exits],
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str))


# Singleton for easy access from both graph.py and engine.py.
# Replaced with a real instance by the CLI when --debug is passed.
_global_tracer = DebugTracer(enabled=False)


def get_tracer() -> DebugTracer:
    return _global_tracer


def set_tracer(tracer: DebugTracer) -> None:
    global _global_tracer
    _global_tracer = tracer
