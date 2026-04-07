"""
Trade journal — records every signal cycle and executed trade to disk.

Files written under `log_dir/`:
  decisions_YYYY-MM-DD.jsonl   — one JSON line per symbol per cycle
  trades_YYYY-MM-DD.csv        — one row per executed order  closed_trades.jsonl           — one JSON line per closed trade (ClosedTrade-compatible)  summary.json                 — running P&L / win-rate snapshot
"""

import csv
import fcntl
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class TradeJournal:
    """Persistent logger for signal decisions and executed trades."""

    TRADES_CSV_HEADER = [
        "timestamp", "symbol", "direction", "quantity",
        "entry_price", "stop_loss", "take_profit",
        "risk_amount", "confidence", "win_probability",
        "expected_value", "order_ticket", "recipe_name",
    ]

    def __init__(self, log_dir: str = "logs",
                 log_decisions: bool = True,
                 log_trades: bool = True,
                 debug_decisions: bool = False):
        self.log_dir = Path(log_dir)
        self.log_decisions = log_decisions
        self.log_trades = log_trades
        self.debug_decisions = debug_decisions
        self.logger = logging.getLogger("TradeJournal")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def record_cycle(self, symbol: str, cycle_result: Dict[str, Any],
                     features_summary: Optional[Dict] = None):
        """
        Called once per symbol per analysis cycle.
        cycle_result: the dict returned by TradingRunner._run_cycle for one symbol.
        """
        if not self.log_decisions:
            return
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "symbol": symbol,
            "decision": cycle_result.get("decision", "stop"),
            "executed": cycle_result.get("executed", False),
            "order_ticket": cycle_result.get("order_ticket"),
            "errors": cycle_result.get("errors", []),
        }
        if features_summary:
            entry["features"] = features_summary
        self._append_jsonl(self._decisions_path(), entry)

    def record_trade(self, symbol: str, trade_plan: Any,
                     order_result: Optional[Dict] = None):
        """
        Called whenever an order is executed.
        trade_plan: TradePlan Pydantic model.
        order_result: dict returned by MT5Executor.place_bracket_order.
        """
        if not self.log_trades:
            return
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "symbol": symbol,
            "direction": trade_plan.recipe.direction,
            "quantity": trade_plan.quantity,
            "entry_price": trade_plan.entry_price,
            "stop_loss": trade_plan.stop_loss,
            "take_profit": trade_plan.take_profit,
            "risk_amount": trade_plan.risk_amount,
            "confidence": round(trade_plan.confidence, 4),
            "win_probability": round(trade_plan.recipe.win_probability, 4),
            "expected_value": round(trade_plan.recipe.expected_value, 4),
            "order_ticket": order_result.get("order_ticket") if order_result else None,
            "recipe_name": trade_plan.recipe.name,
        }
        self._append_csv(self._trades_path(), row)
        self.logger.info(
            f"TRADE  {symbol:15s}  {trade_plan.recipe.direction:5s}  "
            f"qty={trade_plan.quantity:.2f}  entry={trade_plan.entry_price:.5f}  "
            f"sl={trade_plan.stop_loss:.5f}  tp={trade_plan.take_profit:.5f}  "
            f"ticket={row['order_ticket']}"
        )
        self._update_summary()

    def record_decision_debug(
        self,
        symbol: str,
        debug_snapshot: Dict[str, Any],
        bar_time: Optional[str] = None,
    ) -> None:
        """Write a rich per-decision debug record to ``debug_decisions_YYYY-MM-DD.jsonl``.

        Called after every graph pipeline run (trade taken OR rejected) when
        ``debug_decisions`` is enabled.  The snapshot is produced by
        ``TradingGraph._build_debug_snapshot()`` and contains the full
        pipeline data: agent votes, fusion, alignment, recipe, plan, execution.

        Parameters
        ----------
        symbol : str
            The symbol this decision refers to.
        debug_snapshot : dict
            The debug payload assembled by TradingGraph._build_debug_snapshot().
        bar_time : str, optional
            Bar timestamp string (backtest) or omitted (live uses wall clock).
        """
        if not self.debug_decisions:
            return
        entry = {
            "ts": bar_time or datetime.now().isoformat(timespec="seconds"),
            **debug_snapshot,
        }
        self._append_jsonl(self._debug_decisions_path(), entry)

    def _debug_decisions_path(self) -> Path:
        date = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"debug_decisions_{date}.jsonl"

    def record_pnl_snapshot(self, executor, cycle_num: int) -> dict:
        """
        Record open + realized PnL for this cycle to pnl log.
        Returns the snapshot dict so the caller can log/display it.
        """
        open_positions = executor.get_open_positions()
        closed_trades  = executor.get_closed_trades(days=1)

        unrealized    = sum(p.get("profit", 0.0) for p in open_positions
                            if p.get("magic") == executor.magic_number)
        realized_today = sum(t.get("profit", 0.0) for t in closed_trades)

        snapshot = {
            "ts":              datetime.now().isoformat(timespec="seconds"),
            "cycle":           cycle_num,
            "open_positions":  len([p for p in open_positions
                                    if p.get("magic") == executor.magic_number]),
            "unrealized_pnl":  round(unrealized, 2),
            "realized_today":  round(realized_today, 2),
            "total_today":     round(unrealized + realized_today, 2),
        }
        self._append_jsonl(self._pnl_path(), snapshot)
        self._update_summary(snapshot)
        return snapshot

    def get_todays_pnl(self) -> dict:
        """Return the most recent PnL snapshot from today's log."""
        p = self._pnl_path()
        if not p.exists():
            return {"unrealized_pnl": 0.0, "realized_today": 0.0, "total_today": 0.0}
        last_line = None
        with open(p) as f:
            for line in f:
                last_line = line.strip()
        if last_line:
            try:
                return json.loads(last_line)
            except json.JSONDecodeError:
                pass
        return {"unrealized_pnl": 0.0, "realized_today": 0.0, "total_today": 0.0}

    def print_cycle_banner(self, cycle_num: int, results: dict,
                           portfolio_equity: float = None,
                           pnl_snapshot: dict = None):
        """Print a formatted summary banner after each cycle to stdout."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "─" * 72
        trades_this_cycle = sum(1 for r in results.values() if r.get("executed"))

        equity_str = f"  •  equity: ${portfolio_equity:,.2f}" if portfolio_equity else ""
        print(f"\n{sep}")
        print(f"  Cycle #{cycle_num:04d}  •  {now}{equity_str}")
        if pnl_snapshot:
            u = pnl_snapshot.get("unrealized_pnl", 0.0)
            r = pnl_snapshot.get("realized_today", 0.0)
            t = pnl_snapshot.get("total_today", 0.0)
            print(f"  PnL today   unrealized: {u:+.2f}  |  "
                  f"realized: {r:+.2f}  |  total: {t:+.2f}")
        print(sep)
        for sym, r in results.items():
            decision = r.get("decision", "stop")
            executed = r.get("executed", False)
            ticket   = r.get("order_ticket", "")
            errors   = r.get("errors", [])
            flag     = "✓ TRADE" if executed else f"  {decision}"
            ticket_str = f"  #{ticket}" if ticket else ""
            err_str    = f"  ERR: {errors[0]}" if errors else ""
            print(f"  {sym:<20s}  {flag}{ticket_str}{err_str}")
        print(f"{sep}")
        print(f"  Trades this cycle: {trades_this_cycle} | "
              f"Total today: {self._count_trades_today()}")
        print(sep)

    def get_summary(self) -> Dict[str, Any]:
        """Return the current running summary from disk."""
        p = self.log_dir / "summary.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return {"total_trades": 0, "last_updated": None}

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _decisions_path(self) -> Path:
        date = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"decisions_{date}.jsonl"

    def _trades_path(self) -> Path:
        date = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"trades_{date}.csv"

    def _pnl_path(self) -> Path:
        date = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"pnl_{date}.jsonl"

    def _append_jsonl(self, path: Path, entry: dict):
        with open(path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(entry, default=str) + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _append_csv(self, path: Path, row: dict):
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                writer = csv.DictWriter(f, fieldnames=self.TRADES_CSV_HEADER)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _count_trades_today(self) -> int:
        p = self._trades_path()
        if not p.exists():
            return 0
        with open(p) as f:
            # subtract 1 for header row
            return max(0, sum(1 for _ in f) - 1)

    def _update_summary(self, pnl_snapshot: dict = None):
        """Recount all trades and update summary.json with latest PnL."""
        total = 0
        for csv_path in self.log_dir.glob("trades_*.csv"):
            with open(csv_path) as f:
                total += max(0, sum(1 for _ in f) - 1)
        summary = {
            "total_trades":    total,
            "last_updated":    datetime.now().isoformat(timespec="seconds"),
        }
        if pnl_snapshot:
            summary["unrealized_pnl"]  = pnl_snapshot.get("unrealized_pnl", 0.0)
            summary["realized_today"]  = pnl_snapshot.get("realized_today", 0.0)
            summary["total_today_pnl"] = pnl_snapshot.get("total_today", 0.0)
        with open(self.log_dir / "summary.json", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(summary, f, indent=2)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    # ----------------------------------------------------------------
    # Closed-trade log (ClosedTrade-compatible format)
    # ----------------------------------------------------------------

    def record_closed_trade(
        self,
        *,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        stop_loss: float,
        take_profit: float,
        quantity: float,
        risk_amount: float,
        pnl: float,
        pnl_r: float,
        exit_reason: str,
        confidence: float = 0.0,
        win_probability: float = 0.0,
        agent_votes: Optional[Dict[str, float]] = None,
        open_dt: Optional[str] = None,
        close_dt: Optional[str] = None,
        ticket: Optional[int] = None,
        entry_type: str = "full-alignment",
    ) -> None:
        """Append a closed trade in the same schema as backtest ClosedTrade.

        Written to ``closed_trades.jsonl`` (all-time, not daily-rotated) so
        that ``load_live_result()`` can reconstruct a BacktestResult for
        direct comparison with backtest reports.
        """
        entry: Dict[str, Any] = {
            "symbol":          symbol,
            "direction":       direction,
            "entry_price":     round(entry_price, 6),
            "exit_price":      round(exit_price, 6),
            "stop_loss":       round(stop_loss, 6),
            "take_profit":     round(take_profit, 6),
            "quantity":        round(quantity, 4),
            "risk_amount":     round(risk_amount, 4),
            "pnl":             round(pnl, 4),
            "pnl_r":           round(pnl_r, 4),
            "exit_reason":     exit_reason,
            "confidence":      round(confidence, 4),
            "win_probability": round(win_probability, 4),
            "agent_votes":     {k: round(v, 4) for k, v in (agent_votes or {}).items()},
            "open_dt":         open_dt,
            "close_dt":        close_dt or datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "ticket":          ticket,
            "entry_type":      entry_type,
        }
        self._append_jsonl(self.log_dir / "closed_trades.jsonl", entry)
        self.logger.info(
            f"CLOSED {symbol:8s}  {direction:5s}  {exit_reason:16s}  "
            f"pnl={pnl:+.2f}  ({pnl_r:+.2f}R)  qty={quantity:.2f}  "
            f"#{ticket or '-'}"
        )

    def load_live_result(
        self,
        profile: str = "live",
        initial_equity: float = 100_000.0,
        config: Optional[Dict] = None,
    ) -> "BacktestResult":
        """Reconstruct a BacktestResult from the closed_trades.jsonl log.

        This allows ``compute_metrics()`` and ``generate_report()`` to work
        on live trades exactly as they do on backtest results.
        """
        from ..backtesting.engine import BacktestResult, ClosedTrade

        path = self.log_dir / "closed_trades.jsonl"
        trades: list = []
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    open_dt = None
                    close_dt = None
                    if d.get("open_dt"):
                        try:
                            open_dt = datetime.fromisoformat(d["open_dt"].replace("Z", "+00:00"))
                        except (ValueError, TypeError):
                            pass
                    if d.get("close_dt"):
                        try:
                            close_dt = datetime.fromisoformat(d["close_dt"].replace("Z", "+00:00"))
                        except (ValueError, TypeError):
                            pass
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

        # Build a synthetic equity curve from cumulative P&L.
        # bar_dates and equity_curve must have the same length for the
        # report charting code.  We prepend the open_dt of the first trade
        # (or a placeholder) so both lists are len(trades) + 1.
        equity_curve = [initial_equity]
        for t in trades:
            equity_curve.append(equity_curve[-1] + t.pnl)

        _first_dt = (trades[0].open_dt if trades and trades[0].open_dt
                      else datetime.min)
        bar_dates = [_first_dt] + [t.close_dt or datetime.min for t in trades]

        # Derive date range from trades
        start_date = trades[0].open_dt.isoformat() if trades and trades[0].open_dt else ""
        end_date = trades[-1].close_dt.isoformat() if trades and trades[-1].close_dt else ""

        # Symbol — "PORTFOLIO" if mixed, else the single symbol
        symbols = list({t.symbol for t in trades})
        symbol = symbols[0] if len(symbols) == 1 else "PORTFOLIO"

        return BacktestResult(
            profile=profile,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            trades=trades,
            equity_curve=equity_curve,
            bar_dates=bar_dates,
            initial_equity=initial_equity,
            config=config or {},
        )
