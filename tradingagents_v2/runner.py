"""
TradingRunner — the main event loop for automated trading.

Usage:
    import asyncio
    from tradingagents_v2.runner import TradingRunner

    runner = TradingRunner()
    asyncio.run(runner.run_forever())

Or from the CLI:
    python -m tradingagents_v2.runner
"""

import asyncio
import json
import logging
import signal
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from .core.graph import TradingGraph
from .core.agent_base import AgentRegistry
from .core.types import PortfolioState, RiskLimits
from .agents import (
    RegimeAgent, TrendAgent, MomentumAgent,
    MeanReversionAgent, VolatilityAgent, BreadthAgent, PatternAgent,
    IntermarketAgent, SessionBreakoutAgent, DivergenceAgent,
    ScalpingAgent, VwapScalpAgent, SqueezeBreakoutAgent, OrderFlowAgent,
)
from .config.settings import TradingConfig
from .data.loader import DataLoader
from .execution.mt5_executor import MT5Executor
from .execution.order_manager import OrderManager
from .execution.trailing_stop import TrailingStopManager
from .monitoring.journal import TradeJournal


# ──────────────────────────────────────────────────────────────────────────────
# News blackout helpers
# ──────────────────────────────────────────────────────────────────────────────

def _symbol_currencies(symbol: str) -> frozenset:
    """
    Return the set of ISO currency codes a symbol is sensitive to.
    Used by the news blackout guard to match economic events to a symbol.
    """
    sym = symbol.upper()
    # Strip common broker suffixes (EURUSDm, EURUSD.)
    bare = sym.rstrip(".").rstrip("M") if sym.endswith(("M", ".")) else sym
    # Standard 6-character forex pair (EURUSD, GBPJPY, …)
    if len(bare) == 6 and bare.isalpha():
        return frozenset([bare[:3], bare[3:]])
    # 6-char prefix with broker suffix (EURUSDm, EURUSDx)
    if len(bare) > 6 and bare[:6].isalpha():
        return frozenset([bare[:3], bare[3:6]])
    # US equity indices — USD-denominated
    if any(x in sym for x in ("US30", "US500", "USTEC", "NAS", "DJ30", "SP500")):
        return frozenset(["USD"])
    # European indices
    if any(x in sym for x in ("DAX", "GER", "CAC", "FRA")):
        return frozenset(["EUR"])
    if any(x in sym for x in ("UK100", "FTSE", "UK")):
        return frozenset(["GBP"])
    if any(x in sym for x in ("JP225", "JPN", "NIK")):
        return frozenset(["JPY"])
    if any(x in sym for x in ("AUS200", "ASX", "AUS")):
        return frozenset(["AUD"])
    # Commodities / metals
    if "XAU" in sym or "GOLD" in sym:
        return frozenset(["USD"])
    if any(x in sym for x in ("OIL", "WTI", "BRENT", "XTI", "NGAS")):
        return frozenset(["USD"])
    return frozenset(["USD"])   # safe fallback


class _NewsCalendar:
    """
    Lightweight economic calendar cache using the ForexFactory public JSON feed.
    Fetches high-impact events once and caches them for 4 hours.

    If the feed is unavailable or cannot be parsed the guard silently disables
    itself for that cycle — no trades are ever blocked due to an API failure.
    """

    _TTL = 4 * 3600   # refresh every 4 hours
    _URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

    def __init__(self):
        self._events: list = []       # [{"currency": str, "ts": float}, …]
        self._fetched_at: float = 0.0
        self._logger = logging.getLogger("NewsCalendar")

    def _parse_event_ts(self, date_str: str) -> "Optional[float]":
        """Parse a date string (ISO 8601 or MM-DD-YYYY) to a UTC Unix timestamp."""
        from datetime import timezone as _tz
        # Python 3.11 fromisoformat handles full ISO 8601 including timezone offsets
        try:
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_tz.utc)
            return dt.timestamp()
        except Exception:
            pass
        # ForexFactory legacy: "MM-DD-YYYY" or bare "YYYY-MM-DD"
        for fmt in ("%m-%d-%Y", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(date_str[:10], fmt).replace(tzinfo=_tz.utc)
                return dt.timestamp()
            except Exception:
                pass
        return None

    def _refresh(self) -> None:
        import requests as _req
        try:
            resp = _req.get(self._URL, timeout=5)
            resp.raise_for_status()
            raw = resp.json()
            events = []
            for ev in raw:
                impact   = (ev.get("impact") or "").strip().lower()
                if impact not in ("high", "3"):   # FF uses "High" or numeric "3"
                    continue
                currency = (ev.get("currency") or ev.get("country") or "").upper().strip()
                date_str  = (ev.get("date") or "").strip()
                if not date_str or not currency:
                    continue
                ts = self._parse_event_ts(date_str)
                if ts is not None:
                    events.append({"currency": currency, "ts": ts})
            self._events     = events
            self._fetched_at = time.time()
            self._logger.info(
                f"[NEWS] Loaded {len(events)} high-impact events (source: ForexFactory)"
            )
        except Exception as exc:
            # Set _fetched_at to TTL/2 in the past so the next retry is in
            # ~2 hours instead of immediately.  This prevents a 429 retry
            # storm when 4 profile bots share the same source IP.
            # _events is kept unchanged (last successful fetch) so any
            # previously-loaded events still protect the next cycles.
            self._fetched_at = time.time() - self._TTL / 2
            self._logger.warning(
                f"[NEWS] Calendar fetch failed ({exc}) — "
                "news blackout disabled for this refresh cycle; "
                "next retry in ~2 hours"
            )

    def is_blackout(self, symbol: str, blackout_minutes: int) -> bool:
        """Return True if a high-impact event is within ±blackout_minutes for this symbol."""
        if blackout_minutes <= 0:
            return False
        now = time.time()
        if now - self._fetched_at > self._TTL:
            self._refresh()
        if not self._events:
            return False
        window     = blackout_minutes * 60.0
        currencies = _symbol_currencies(symbol)
        for ev in self._events:
            if ev["currency"] in currencies and abs(ev["ts"] - now) <= window:
                mins_away = (ev["ts"] - now) / 60.0
                self._logger.info(
                    f"[NEWS] Blackout: {symbol} ({ev['currency']} event "
                    f"in {mins_away:+.0f} min)"
                )
                return True
        return False


_news_calendar = _NewsCalendar()


def _build_registry(config: TradingConfig) -> AgentRegistry:
    """Instantiate and register all enabled agents."""
    registry = AgentRegistry()
    all_agents = [
        RegimeAgent(),
        TrendAgent(),
        MomentumAgent(),
        MeanReversionAgent(),
        VolatilityAgent(),
        BreadthAgent(),
        PatternAgent(),
        IntermarketAgent(),
        SessionBreakoutAgent(),
        DivergenceAgent(),
        ScalpingAgent(),         # active in scalp profile (3× weight)
        VwapScalpAgent(),        # active in scalp profile (2.5× weight)
        SqueezeBreakoutAgent(),  # active in scalp profile (2.5× weight)
        OrderFlowAgent(),        # active in scalp profile (2.0× weight)
    ]
    for agent in all_agents:
        if config.is_agent_enabled(agent.name):
            agent.weight = config.get_agent_weight(agent.name)
            registry.register(agent)
    return registry


def _portfolio_state_from_executor(executor: MT5Executor,
                                   order_manager: OrderManager) -> Optional[PortfolioState]:
    """Build a PortfolioState from live MT5 account data."""
    account = executor.get_account_info()
    if account is None:
        return None
    open_pos = [p["symbol"] for p in executor.get_open_positions()
                if p.get("magic") == executor.magic_number]
    # Build per-symbol detail map for scale-in decisions
    open_positions_map: dict = {}
    for p in executor.get_open_positions():
        if p.get("magic") != executor.magic_number:
            continue
        sym = p["symbol"]
        open_positions_map.setdefault(sym, []).append({
            "ticket":     p["ticket"],
            "type":       p["type"],    # "BUY" or "SELL"
            "profit":     p["profit"],
            "price_open": p["price_open"],
        })
    balance = account["balance"]
    equity  = account["equity"]
    daily_pnl = equity - balance    # open floating P&L (sign: + = profit, - = loss)
    # daily_drawdown is populated by _update_daily_drawdown on TradingRunner;
    # use floating P&L as a safe initialiser until the runner fills it in.
    daily_drawdown = daily_pnl / max(balance, 1.0)
    return PortfolioState(
        equity=equity,
        margin_used=account["margin"],
        free_margin=account["free_margin"],
        daily_pnl=daily_pnl,
        daily_drawdown=daily_drawdown,
        open_positions=open_pos,
        open_positions_map=open_positions_map,
        max_daily_drawdown=daily_drawdown,
        leverage_used=account["leverage"],
    )


class TradingRunner:
    """
    Orchestrates the full automated trading cycle:
        1. Load features from MT5 (or simulation)
        2. Run TradingGraph for each symbol
        3. Log results / update OrderManager
        4. Sleep until next bar
    """

    def __init__(self, config: TradingConfig = None, simulation: bool = False):
        self.config = config or TradingConfig()
        self.logger = logging.getLogger("TradingRunner")
        self._running = False

        # Build components
        executor_cfg = self.config.mt5.model_dump()
        if simulation:
            executor_cfg["simulation"] = True    # force simulation mode regardless of mt5linux
        # Stamp each order's MT5 comment with the active profile so trades are
        # identifiable in the MT5 History / Trades tab of the UI.
        # Format: "b_{abbrev}" e.g. "b_safe", "b_bal", "b_rsk", "b_scp"
        _p_abbrev = {"balanced": "bal", "risky": "rsk", "scalp": "scp"}.get(
            self.config.profile, self.config.profile
        )
        executor_cfg["order_comment"] = f"b_{_p_abbrev}"
        self.executor = MT5Executor(executor_cfg)
        self.order_manager = OrderManager(self.config.mt5.model_dump())

        # Build tier→timeframe map from profile config so scalp's 1m actually loads
        _tf_cfg = self.config.timeframes  # TimeframeConfig model
        _tier_tf_map = {
            "long":  _tf_cfg.long[0]  if _tf_cfg.long  else "1D",
            "mid":   _tf_cfg.mid[0]   if _tf_cfg.mid   else "1H",
            "short": _tf_cfg.short[0] if _tf_cfg.short else "15m",
        }
        self.data_loader = DataLoader(simulation=simulation, executor=self.executor,
                                      timeframe_map=_tier_tf_map)
        registry = _build_registry(self.config)
        self.graph = TradingGraph(registry, self.config.model_dump(), self.executor,
                                  data_loader=self.data_loader)

        # Journal
        j = self.config.journal
        self.journal = TradeJournal(
            log_dir=j.log_dir,
            log_decisions=j.log_decisions,
            log_trades=j.log_trades,
        )

        # Trailing stop manager — read tuning from optional trailing: config block
        trailing_cfg = self.config.model_dump().get("trailing", {})
        self.trailing_stop_mgr = TrailingStopManager(
            executor=self.executor,
            atr_multiplier=float(trailing_cfg.get("atr_multiplier", 1.5)),
            min_profit_atr=1.0,
            tp_extend_enabled=bool(trailing_cfg.get("tp_extend_enabled", True)),
            tp_extend_atr_mult=float(trailing_cfg.get("tp_extend_atr_mult", 2.0)),
            partial_tp_enabled=bool(trailing_cfg.get("partial_tp_enabled", False)),
            partial_tp_fraction=float(trailing_cfg.get("partial_tp_fraction", 0.50)),
            partial_tp2_enabled=bool(trailing_cfg.get("partial_tp2_enabled", False)),
            partial_tp2_r_mult=float(trailing_cfg.get("partial_tp2_r_mult", 2.0)),
            partial_tp2_fraction=float(trailing_cfg.get("partial_tp2_fraction", 0.50)),
            windfall_exit_enabled=bool(trailing_cfg.get("windfall_exit_enabled", False)),
            windfall_r_mult=float(trailing_cfg.get("windfall_r_mult", 3.0)),
            profile=self.config.profile,
        )

        # Risk limits from config
        self.risk_limits = RiskLimits(
            base_risk_pct=self.config.risk.base_risk_pct,
            max_daily_drawdown_pct=self.config.risk.max_daily_drawdown_pct,
            max_concurrent_trades=self.config.risk.max_concurrent_trades,
            per_symbol_leverage_cap=self.config.risk.per_symbol_leverage_cap,
            portfolio_leverage_cap=self.config.risk.portfolio_leverage_cap,
            max_correlated_positions=self.config.risk.max_correlated_positions,
        )

        # Daily drawdown tracking: measure from the *start-of-day* balance
        # (first snapshot each UTC calendar day) so realized losses are included.
        # `equity - balance` only measures open floating P&L, which resets to 0
        # whenever positions are flat — useless as a daily loss limit.
        self._start_of_day_balance: float = 0.0
        self._start_of_day_date: str = ""

        # Per-symbol entry cooldown: tracks the last time (epoch seconds) a
        # trade was successfully placed on each symbol.  Prevents continuous
        # re-entry on the same symbol every 60-second cycle.
        self._entry_cooldown: dict = {}   # symbol → float (epoch seconds)

        # Daily trade count: resets each UTC calendar day.  Enforces
        # max_daily_trades from the risk config to cap total daily exposure.
        self._daily_trade_count: int = 0

        # Weekly drawdown tracking — measures equity from start of UTC week
        # (ISO Monday).  max_weekly_drawdown_pct > 0 trips a weekly halt circuit
        # breaker that blocks all new entries until the next Monday.
        self._start_of_week_balance: float = 0.0
        self._start_of_week_date:    str   = ""   # ISO week e.g. "2026-W12"

        # Weekend gap protection flag — set Friday after market close, cleared
        # on Monday when markets reopen.  While True, the signal loop skips
        # all new entries so no positions remain at the Sunday gap open.
        self._weekend_close_active: bool = False

        # ── Persist / restore daily guards across restarts ────────────────────
        # Without this: every restart resets _daily_trade_count=0 and
        # _entry_cooldown={}, completely bypassing the overtrading guards.
        # Root cause of 1556 scalp trades in one session (guard always read 0).
        self._state_file = Path(
            self.journal.log_dir / f"_bot_state_{self.config.profile}.json"
        )
        self._restore_daily_state()

        self._setup_signal_handlers()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def run_forever(self, interval_seconds: int = None):
        """
        Run two concurrent loops:

        Fast loop  (surveillance_interval_seconds, default 60s):
          — Checks every open position for SL lock-in and agent-based exit signals.
          — Uses only live bid/ask price to evaluate SL staging (no bar loading).
          — Agent exit check still uses bars, but is lightweight (exit_check_only).

        Slow loop  (interval_seconds, default 300s):
          — Full multi-timeframe agent analysis for every symbol.
          — Places new entry orders when alignment conditions are met.

        A watchdog task runs every 90s and reconnects MT5 if the connection
        drops (network blip, container restart, etc.).
        """
        if interval_seconds is None:
            interval_seconds = self.config.interval_seconds
        rt_cfg = self.config.model_dump().get("realtime", {})
        surveillance_interval = int(rt_cfg.get("surveillance_interval_seconds", 60))

        self._running = True
        self._cycle_num = 0
        self.logger.info(
            f"TradingRunner started — symbols: {self.config.symbols} | "
            f"profile: {self.config.profile} | "
            f"signal: {interval_seconds}s | surveillance: {surveillance_interval}s"
        )
        # Run all three loops concurrently; if any crashes fatally it raises
        # and asyncio.gather re-raises, letting the outer try/except log it.
        await asyncio.gather(
            self._surveillance_loop(surveillance_interval),
            self._signal_loop(interval_seconds),
            self._watchdog_loop(check_interval=90),
        )

    # ------------------------------------------------------------------
    # Watchdog — MT5 connection health + autonomous recovery
    # ------------------------------------------------------------------

    async def _watchdog_loop(self, check_interval: int = 90) -> None:
        """
        Runs every *check_interval* seconds.
        - Verifies MT5 connection is alive (account_info ping).
        - If disconnected, attempts up to 5 reconnects with exponential back-off.
        - Logs a heartbeat every 10 minutes so external monitors can detect silence.
        """
        _MAX_RECONNECT  = 5
        _BACKOFF_BASE   = 10   # seconds — doubles each attempt
        _HEARTBEAT_SECS = 600  # log "alive" every 10 min
        last_heartbeat  = 0.0

        while self._running:
            await asyncio.sleep(check_interval)
            try:
                import time as _time
                now = _time.time()

                # ── Heartbeat ─────────────────────────────────────────────────
                if now - last_heartbeat >= _HEARTBEAT_SECS:
                    acct = self.executor.get_account_info()
                    equity = acct.get("equity", "?") if acct else "?"
                    self.logger.info(
                        f"[WATCHDOG] alive | equity={equity} | "
                        f"cycle={self._cycle_num} | profile={self.config.profile}"
                    )
                    last_heartbeat = now

                # ── MT5 connection check ───────────────────────────────────────
                if self.executor.simulation_mode:
                    continue   # nothing to reconnect in simulation

                acct = self.executor.get_account_info()
                if acct is not None:
                    continue   # all good

                # Connection lost — attempt reconnect
                self.logger.warning("[WATCHDOG] MT5 connection lost — attempting reconnect")
                delay = _BACKOFF_BASE
                reconnected = False
                for attempt in range(1, _MAX_RECONNECT + 1):
                    self.logger.info(
                        f"[WATCHDOG] Reconnect attempt {attempt}/{_MAX_RECONNECT} "
                        f"(waiting {delay}s)"
                    )
                    await asyncio.sleep(delay)
                    try:
                        self.executor.shutdown()
                    except Exception:
                        pass
                    try:
                        cfg = self.config.mt5.model_dump()
                        from .execution.mt5_executor import MT5Executor
                        self.executor = MT5Executor(cfg)
                        # Propagate new executor to dependent components
                        self.data_loader._executor = self.executor
                        self.data_loader.simulation = self.executor.simulation_mode
                        self.data_loader._tf_map    = self.executor.get_tf_map()
                        self.trailing_stop_mgr.executor = self.executor
                        self.graph.executor = self.executor
                        # Verify
                        if self.executor.get_account_info() is not None:
                            self.logger.info("[WATCHDOG] MT5 reconnected successfully")
                            reconnected = True
                            break
                    except Exception as exc:
                        self.logger.error(f"[WATCHDOG] Reconnect attempt {attempt} failed: {exc}")
                    delay = min(delay * 2, 120)   # cap at 2 min

                if not reconnected:
                    self.logger.critical(
                        "[WATCHDOG] Could not reconnect to MT5 after "
                        f"{_MAX_RECONNECT} attempts — bot will keep retrying next cycle"
                    )

            except Exception as exc:
                self.logger.error(f"[WATCHDOG] Unexpected error: {exc}")

    async def _surveillance_loop(self, interval: int) -> None:
        """Fast loop: SL management + exit checks for open positions."""
        _skip_ratchet = bool(
            self.config.model_dump().get("exit_rules", {})
            .get("skip_structural_sl_tp_ratchet", False)
        )
        while self._running:
            start = datetime.now()
            try:
                # ── Friday weekend gap protection ──────────────────────────────────
                # Close all positions before the weekend to prevent Sunday-open gaps
                # from blowing through SLs that were sized for normal market hours.
                _rt_cfg_sv = self.config.model_dump().get("realtime", {})
                if bool(_rt_cfg_sv.get("weekend_close_enabled", False)):
                    _now_utc  = datetime.utcnow()
                    _cutoff   = int(_rt_cfg_sv.get("weekend_close_utc_hour", 20))
                    if _now_utc.weekday() == 4 and _now_utc.hour >= _cutoff:
                        # Friday past market close — close everything once
                        if not self._weekend_close_active:
                            self._weekend_close_active = True
                            await self._close_all_positions(
                                f"Friday weekend gap protection (past {_cutoff}:00 UTC)"
                            )
                    elif _now_utc.weekday() not in (4, 5):
                        # Monday–Thursday or Sunday: markets open → allow entries again
                        self._weekend_close_active = False

                # 1. Staged SL lock-in (uses live price from positions, no bar load)
                self.trailing_stop_mgr.update_trails(self.data_loader)
                # 2. Structural SL/TP ratchet (pivot-based, every surveillance tick)
                #    Skipped when `exit_rules.skip_structural_sl_tp_ratchet` is True
                #    (e.g. scalp profile — ratchet would overwrite tight 1m-ATR targets)
                if not _skip_ratchet:
                    self.trailing_stop_mgr.update_structural_sl_tp(self.data_loader)

                # ── Drawdown circuit breaker: close ALL open positions when the
                # daily drawdown ceiling is breached.  Unlike the entry-halt in
                # _run_cycle (which only stops new entries), this actively closes
                # existing losers so a bad morning can't silently compound.
                # The check runs every surveillance tick (60s) so the response is fast.
                _dd_cfg = self.config.model_dump().get("risk", {})
                _max_dd_pct = float(_dd_cfg.get("max_daily_drawdown_pct", 0))
                if _max_dd_pct > 0 and self._start_of_day_balance > 0:
                    _acct = self.executor.get_account_info()
                    if _acct is not None:
                        _equity_now = _acct.get("equity", self._start_of_day_balance)
                        _real_dd = (_equity_now - self._start_of_day_balance) / self._start_of_day_balance
                        if _real_dd < -(_max_dd_pct / 100):
                            self.logger.warning(
                                f"[DD CLOSE-ALL] Daily drawdown {_real_dd*100:.2f}% "
                                f"breached -{_max_dd_pct:.2f}% ceiling — "
                                "closing all open positions"
                            )
                            await self._close_all_positions(
                                f"drawdown circuit breaker: {_real_dd*100:.2f}% < -{_max_dd_pct:.2f}%"
                            )

                # 3. Agent-based exit check (exit_check_only — no entry evaluation)
                await self._check_and_close_positions()
            except Exception as e:
                self.logger.error(f"Surveillance loop error: {e}")
            elapsed = (datetime.now() - start).total_seconds()
            await asyncio.sleep(max(0.0, interval - elapsed))

    async def _signal_loop(self, interval: int) -> None:
        """Slow loop: full agent analysis + new entry decisions."""
        while self._running:
            cycle_start = datetime.now()
            self._cycle_num += 1
            try:
                results = await self._run_cycle()
                elapsed = (datetime.now() - cycle_start).total_seconds()
                account = self.executor.get_account_info()
                equity = account.get("equity") if account else None
                pnl = self.journal.get_todays_pnl()
                self.journal.print_cycle_banner(self._cycle_num, results, equity, pnl)
                self.logger.info(f"Signal cycle done in {elapsed:.1f}s — sleeping {max(0, interval - elapsed):.0f}s")
                await asyncio.sleep(max(0.0, interval - elapsed))
            except Exception as e:
                self.logger.error(f"Signal loop error: {e}")
                await asyncio.sleep(interval)

    async def run_once(self, symbols: List[str] = None) -> dict:
        """Run a single analysis cycle and return results. Useful for testing."""
        return await self._run_cycle(symbols)

    async def _close_all_positions(self, reason: str) -> None:
        """
        Close every open position managed by this bot instance.
        Used for end-of-week gap protection and weekly drawdown circuit breaker.
        """
        open_positions = [
            p for p in self.executor.get_open_positions()
            if p.get("magic") == self.executor.magic_number
        ]
        if not open_positions:
            self.logger.info(f"[CLOSE ALL] No open positions ({reason})")
            return
        self.logger.warning(
            f"[CLOSE ALL] Closing {len(open_positions)} position(s) — {reason}"
        )
        for pos in open_positions:
            symbol = pos["symbol"]
            ticket = pos["ticket"]
            ok = self.executor.close_position(ticket)
            if ok:
                self.logger.info(
                    f"  Closed {symbol} #{ticket}  pnl={pos['profit']:+.2f}"
                )
                self.journal.record_cycle(symbol, {
                    "decision":     "closed",
                    "executed":     True,
                    "order_ticket": ticket,
                    "errors":       [],
                })
            else:
                self.logger.error(f"  Failed to close {symbol} #{ticket}")

    async def _check_and_close_positions(self) -> None:
        """
        For every open bot position, re-compute the fused directional signal
        and evaluate three exit conditions:

        1. D1 FLIP        — D1 trend has reversed against the trade direction.
        2. CONVICTION FADE — D1 signal has gone flat/neutral (trend evaporated).
        3. MID+SHORT OPPOSITION — both 1H and 15m strongly oppose the trade
                                   while D1 conviction is weak (stalled trend).
        """
        open_positions = [
            p for p in self.executor.get_open_positions()
            if p.get("magic") == self.executor.magic_number
        ]
        if not open_positions:
            return

        cfg_dict   = self.config.model_dump()
        al_cfg     = cfg_dict.get("alignment", {})
        exit_cfg   = cfg_dict.get("exit_rules", {})

        flip_threshold      = float(al_cfg.get("long_min_score", 0.25))
        # Conviction fade: close when |D1| drops below this (trend gone flat)
        fade_threshold      = float(exit_cfg.get("conviction_fade_threshold", 0.10))
        # Mid+Short opposition: close when BOTH mid & short oppose by more than this
        opposition_threshold = float(exit_cfg.get("mid_short_opposition_threshold", 0.35))
        # Whether each condition is enabled
        fade_enabled        = bool(exit_cfg.get("conviction_fade_enabled", True))
        opposition_enabled  = bool(exit_cfg.get("mid_short_opposition_enabled", True))
        # Scalp mode: exit on SHORT-tier reversal instead of D1 conviction
        short_tier_exits    = bool(exit_cfg.get("use_short_tier_exits", False))
        short_flip_thresh   = float(exit_cfg.get("short_flip_threshold", 0.35))

        for pos in open_positions:
            symbol     = pos["symbol"]
            trade_type = pos["type"]  # "BUY" or "SELL"
            try:
                features_by_tf = self.data_loader.get_multi_features(symbol)
                features = (features_by_tf.get("mid")
                            or features_by_tf.get("long")
                            or features_by_tf.get("short"))
                if features is None:
                    continue

                state = await self.graph.run(
                    symbol=symbol,
                    features=features,
                    portfolio_state=None,
                    risk_limits=self.risk_limits,
                    features_by_tf=features_by_tf,
                    exit_check_only=True,
                )

                fusion = state.get("timeframe_fusion")
                if fusion is None:
                    continue

                dir_long  = fusion.dir_long
                dir_mid   = fusion.dir_mid
                dir_short = fusion.dir_short

                should_close = False
                reason = ""

                if short_tier_exits:
                    # ── SCALP exits: driven by the 1m SHORT-tier signal ────────
                    # dir_long ≈ 0 in scalp (no long-tier agents) — using it for
                    # exit decisions would close every trade within 20 seconds.
                    # Instead we exit when the 1m momentum has clearly reversed.

                    # Condition A: SHORT-tier flip against trade
                    if trade_type == "BUY" and dir_short < -short_flip_thresh:
                        should_close = True
                        reason = f"1m signal flipped bearish (dir_short={dir_short:.3f})"
                    elif trade_type == "SELL" and dir_short > short_flip_thresh:
                        should_close = True
                        reason = f"1m signal flipped bullish (dir_short={dir_short:.3f})"

                    # Condition B: Mid+Short both strongly oppose (trend exhaustion)
                    if not should_close and opposition_enabled:
                        if trade_type == "BUY" and dir_mid < -opposition_threshold and dir_short < -opposition_threshold:
                            should_close = True
                            reason = (
                                f"Mid+Short both reversed bearish "
                                f"(M={dir_mid:.3f} S={dir_short:.3f})"
                            )
                        elif trade_type == "SELL" and dir_mid > opposition_threshold and dir_short > opposition_threshold:
                            should_close = True
                            reason = (
                                f"Mid+Short both reversed bullish "
                                f"(M={dir_mid:.3f} S={dir_short:.3f})"
                            )

                else:
                    # ── SWING exits: driven by D1 long-tier signal ─────────────

                    # ── Condition 1: D1 flip ───────────────────────────────────
                    if trade_type == "BUY" and dir_long < -flip_threshold:
                        should_close = True
                        reason = f"D1 flipped bearish ({dir_long:.3f})"
                    elif trade_type == "SELL" and dir_long > flip_threshold:
                        should_close = True
                        reason = f"D1 flipped bullish ({dir_long:.3f})"

                    # ── Condition 2: Conviction fade ───────────────────────────
                    if not should_close and fade_enabled:
                        if abs(dir_long) < fade_threshold:
                            should_close = True
                            reason = (
                                f"D1 conviction faded (|{dir_long:.3f}| < {fade_threshold}) "
                                f"— trend evaporated"
                            )

                    # ── Condition 3: Mid+Short strong opposition ───────────────
                    if not should_close and opposition_enabled:
                        if trade_type == "BUY":
                            if dir_mid < -opposition_threshold and dir_short < -opposition_threshold:
                                should_close = True
                                reason = (
                                    f"Mid+Short strongly bearish "
                                    f"(M={dir_mid:.3f} S={dir_short:.3f}) while D1 weak ({dir_long:.3f})"
                                )
                        elif trade_type == "SELL":
                            if dir_mid > opposition_threshold and dir_short > opposition_threshold:
                                should_close = True
                                reason = (
                                    f"Mid+Short strongly bullish "
                                    f"(M={dir_mid:.3f} S={dir_short:.3f}) while D1 weak ({dir_long:.3f})"
                                )

                # ── Time-stop ────────────────────────────────────────────────
                # If neither signal condition fired, check whether the trade has
                # been open too long without reaching its TP.
                if not should_close:
                    max_hours = float(exit_cfg.get("max_trade_duration_hours", 0))
                    if max_hours > 0:
                        open_ts = pos.get("open_time", 0)  # UNIX seconds
                        if open_ts and open_ts > 0:
                            import time as _t
                            hours_open = (_t.time() - open_ts) / 3600.0
                            if hours_open >= max_hours:
                                should_close = True
                                reason = (
                                    f"time-stop: open {hours_open:.1f}h "
                                    f">= {max_hours:.0f}h limit  "
                                    f"pnl={pos['profit']:+.2f}"
                                )

                if should_close:
                    self.logger.info(
                        f"EXIT [{symbol} #{pos['ticket']} {trade_type}  "
                        f"pnl={pos['profit']:+.2f}] — {reason}"
                    )
                    ok = self.executor.close_position(pos["ticket"])
                    if ok:
                        self.journal.record_cycle(symbol, {
                            "decision": "closed",
                            "executed": True,
                            "order_ticket": pos["ticket"],
                            "errors": [],
                        })
                    else:
                        self.logger.warning(f"Failed to close {symbol} #{pos['ticket']}")
                else:
                    # ── Conviction-fade tighten (SL→BE + TP→price±N×ATR) ──────────
                    # When conviction is weakening (fading but not yet at close
                    # threshold), lock the SL at break-even and tighten the TP so
                    # the trade captures nearby profit before conditions worsen.
                    tighten_enabled   = bool(exit_cfg.get("tighten_on_fade_enabled", True))
                    tighten_threshold = float(exit_cfg.get("tighten_fade_threshold", 0.20))
                    tighten_tp_mult   = float(exit_cfg.get("tighten_tp_atr_mult", 0.5))
                    pos_entry   = pos["price_open"]
                    pos_sl      = pos.get("sl", 0.0)
                    pos_tp      = pos.get("tp", 0.0)
                    price_now   = pos["price_current"]
                    pos_profit  = pos.get("profit", 0.0)
                    atr = features.atr_14 if features and features.atr_14 else 0.0

                    if (tighten_enabled
                            and atr > 0
                            and pos_profit > 0
                            and fade_threshold < abs(dir_long) < tighten_threshold):
                        if trade_type == "BUY":
                            tight_sl = max(pos_sl, pos_entry)            # at least BE
                            tight_tp = price_now + tighten_tp_mult * atr  # nearby TP
                            # Only apply if TP would meaningfully tighten
                            # (must be at least 0.2×ATR more conservative than current)
                            tp_improves = (pos_tp <= 0 or tight_tp < pos_tp - 0.2 * atr)
                            sl_improves = (tight_sl > pos_sl)
                        else:                                              # SELL
                            tight_sl = min(pos_sl, pos_entry) if pos_sl > 0 else pos_entry
                            tight_tp = price_now - tighten_tp_mult * atr
                            tp_improves = (pos_tp <= 0 or tight_tp > pos_tp + 0.2 * atr)
                            sl_improves = (pos_sl <= 0 or tight_sl < pos_sl)

                        if tp_improves or sl_improves:
                            new_sl = tight_sl if sl_improves else pos_sl
                            new_tp = tight_tp if tp_improves else pos_tp
                            self.logger.info(
                                f"TIGHTEN [{symbol} #{pos['ticket']} {trade_type}  "
                                f"pnl={pos_profit:+.2f}] conviction fading D1={dir_long:.3f} — "
                                f"SL: {pos_sl:.5f}→{new_sl:.5f}  "
                                f"TP: {pos_tp:.5f}→{new_tp:.5f}"
                            )
                            self.executor.modify_sl_tp(pos["ticket"], new_sl, new_tp)

                    self.logger.debug(
                        f"HOLD {symbol} #{pos['ticket']} {trade_type}  "
                        f"pnl={pos['profit']:+.2f}  "
                        f"D1={dir_long:.3f} M={dir_mid:.3f} S={dir_short:.3f}"
                    )

            except Exception as e:
                self.logger.error(f"Error checking exit for {symbol}: {e}")

    # ------------------------------------------------------------------
    # Internal cycle
    # ------------------------------------------------------------------

    async def _run_cycle(self, symbols: List[str] = None) -> dict:
        symbols = symbols or self.config.symbols

        portfolio = _portfolio_state_from_executor(self.executor, self.order_manager)
        results = {}

        # Update start-of-day balance once per UTC calendar day so that
        # daily_drawdown correctly counts ALL realized losses, not just
        # the current unrealized floating P&L.
        if portfolio is not None:
            today_utc = datetime.utcnow().strftime("%Y-%m-%d")
            if self._start_of_day_date != today_utc:
                self._start_of_day_balance = portfolio.equity
                self._start_of_day_date = today_utc
                self._daily_trade_count = 0  # reset counter each new UTC day
                self.logger.info(
                    f"New trading day {today_utc} — "
                    f"start-of-day equity: {self._start_of_day_balance:.2f}"
                )
                # Persist the fresh daily state (counter = 0, new SOD balance)
                self._persist_daily_state()
            if self._start_of_day_balance > 0:
                real_dd = (portfolio.equity - self._start_of_day_balance) / self._start_of_day_balance
                portfolio = portfolio.model_copy(
                    update={"daily_drawdown": real_dd, "max_daily_drawdown": real_dd}
                )

            # ── Weekly drawdown tracking ───────────────────────────────────────
            # Use ISO 8601 week number (%G-W%V) so the counter resets on Monday.
            _this_week = datetime.utcnow().strftime("%G-W%V")
            if self._start_of_week_date != _this_week:
                self._start_of_week_balance = portfolio.equity
                self._start_of_week_date    = _this_week
                self.logger.info(
                    f"New trading week {_this_week} — "
                    f"start-of-week equity: {self._start_of_week_balance:.2f}"
                )

        # ── Circuit breakers (checked before ANY new entry this cycle) ─────────
        # 1. Weekend close mode: surveillance loop has already closed positions.
        if self._weekend_close_active:
            self.logger.debug("Weekend close active — no new entries until Monday")
            return {}

        # 2. Weekly drawdown halt: too much equity lost this week.
        _risk_cfg_cb = self.config.model_dump().get("risk", {})
        _max_weekly_pct = float(_risk_cfg_cb.get("max_weekly_drawdown_pct", 0))
        if _max_weekly_pct > 0 and self._start_of_week_balance > 0 and portfolio is not None:
            _weekly_dd = (
                (portfolio.equity - self._start_of_week_balance)
                / self._start_of_week_balance
            )
            if _weekly_dd < -(_max_weekly_pct / 100):
                self.logger.warning(
                    f"[WEEKLY HALT] Weekly drawdown {_weekly_dd * 100:.2f}% breached "
                    f"-{_max_weekly_pct:.2f}% ceiling — "
                    "no new entries until Monday"
                )
                return {}

        # Phase 1: Fetch bar data for all symbols concurrently, capped at 5 simultaneous
        # requests to avoid overwhelming the MT5 bridge with a request storm.
        _sem = asyncio.Semaphore(5)
        loop = asyncio.get_event_loop()

        async def _fetch(sym: str):
            async with _sem:
                return await loop.run_in_executor(
                    None, self.data_loader.get_multi_features, sym
                )

        fetch_results = await asyncio.gather(
            *[_fetch(s) for s in symbols], return_exceptions=True
        )
        features_map = {
            sym: ft
            for sym, ft in zip(symbols, fetch_results)
            if not isinstance(ft, BaseException)
        }
        # Log any fetch-level failures
        for sym, ft in zip(symbols, fetch_results):
            if isinstance(ft, BaseException):
                self.logger.error(f"Data fetch failed for {sym}: {ft}")

        # Phase 2: Process each symbol sequentially — graph analysis and order
        # execution must remain serial to avoid race conditions in order_manager.
        _risk_cfg = self.config.model_dump().get("risk", {})
        _max_daily = int(_risk_cfg.get("max_daily_trades", 0))
        _cooldown_secs = float(_risk_cfg.get("entry_cooldown_minutes", 0)) * 60.0
        for symbol in symbols:
            # Daily trade cap: stop analysing further symbols once limit reached
            if _max_daily > 0 and self._daily_trade_count >= _max_daily:
                self.logger.warning(
                    f"Max daily trades ({_max_daily}) reached — no further entries today"
                )
                break

            # Per-symbol cooldown: skip symbol if not enough time has elapsed since
            # the last trade on that symbol.
            if _cooldown_secs > 0 and symbol in self._entry_cooldown:
                elapsed = time.time() - self._entry_cooldown[symbol]
                if elapsed < _cooldown_secs:
                    self.logger.debug(
                        f"{symbol}: cooldown active "
                        f"({elapsed:.0f}s / {_cooldown_secs:.0f}s) — skipping"
                    )
                    results[symbol] = {"decision": "cooldown", "executed": False,
                                       "order_ticket": None, "errors": []}
                    continue

            self.logger.info(f"Analysing {symbol} …")
            try:
                if symbol not in features_map:
                    self.logger.warning(f"Could not load features for {symbol} — skipping")
                    results[symbol] = {"decision": "stop", "executed": False,
                                       "order_ticket": None, "errors": ["fetch failed"]}
                    continue

                features_by_tf = features_map[symbol]
                features = (features_by_tf.get("mid")
                            or features_by_tf.get("long")
                            or features_by_tf.get("short"))
                if features is None:
                    self.logger.warning(f"Could not load features for {symbol} — skipping")
                    continue

                # ── News blackout ──────────────────────────────────────────────
                # Block new entries within ±N minutes of a high-impact economic
                # event for any currency in this symbol (NFP, Fed, CPI, …).
                _news_mins = int(
                    self.config.model_dump().get("execution", {})
                    .get("news_blackout_minutes", 0)
                )
                if _news_mins > 0 and _news_calendar.is_blackout(symbol, _news_mins):
                    self.logger.info(
                        f"{symbol}: high-impact event within {_news_mins} min — "
                        "skipping entry (news blackout)"
                    )
                    results[symbol] = {
                        "decision":     "news_blackout",
                        "executed":     False,
                        "order_ticket": None,
                        "errors":       [],
                    }
                    continue

                state = await self.graph.run(
                    symbol=symbol,
                    features=features,
                    portfolio_state=portfolio,
                    risk_limits=self.risk_limits,
                    features_by_tf=features_by_tf,
                )

                results[symbol] = {
                    "decision":     state.get("decision", "stop"),
                    "executed":     state.get("metadata", {}).get("executed", False),
                    "order_ticket": state.get("metadata", {}).get("order_ticket"),
                    "errors":       state.get("errors", []),
                    "block_reason": state.get("metadata", {}).get("block_reason", ""),
                }

                # Journal: record this cycle
                self.journal.record_cycle(symbol, results[symbol])

                # Register in order manager if a trade was placed
                if state.get("trade_plan") and state.get("metadata", {}).get("executed"):
                    mt5_ticket = state["metadata"].get("order_ticket")
                    self.order_manager.create_order(state["trade_plan"], mt5_ticket=mt5_ticket)
                    self.journal.record_trade(
                        symbol,
                        state["trade_plan"],
                        {"order_ticket": mt5_ticket},
                    )
                    # Update entry cooldown and daily trade counter
                    self._entry_cooldown[symbol] = time.time()
                    self._daily_trade_count += 1
                    self.logger.info(
                        f"{symbol}: trade placed — daily count {self._daily_trade_count}/{_max_daily or '∞'}, "
                        f"next entry in {_cooldown_secs/60:.0f}min"
                    )
                    # Persist state immediately so a restart won't bypass guards
                    self._persist_daily_state()
                    # Refresh portfolio so the next symbol in this cycle sees the
                    # updated open-position count and daily drawdown.
                    _refreshed = _portfolio_state_from_executor(self.executor, self.order_manager)
                    if _refreshed is not None:
                        if self._start_of_day_balance > 0:
                            real_dd = (_refreshed.equity - self._start_of_day_balance) / self._start_of_day_balance
                            _refreshed = _refreshed.model_copy(
                                update={"daily_drawdown": real_dd, "max_daily_drawdown": real_dd}
                            )
                        portfolio = _refreshed

            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                results[symbol] = {"decision": "stop", "executed": False,
                                   "order_ticket": None, "errors": [str(e)]}

        # Update trailing stops for all open positions — now owned by surveillance loop
        # (kept here as fallback when run_once is called directly in tests)
        if not self._running:
            self.trailing_stop_mgr.update_trails(self.data_loader)
            self.trailing_stop_mgr.update_structural_sl_tp(self.data_loader)

        # Record PnL snapshot
        pnl = self.journal.record_pnl_snapshot(
            self.executor, getattr(self, "_cycle_num", 0)
        )
        self.logger.info(
            f"PnL — unrealized: {pnl['unrealized_pnl']:+.2f}  "
            f"realized today: {pnl['realized_today']:+.2f}  "
            f"total: {pnl['total_today']:+.2f}"
        )

        return results

    # ------------------------------------------------------------------
    # Daily state persistence — survives restarts
    # ------------------------------------------------------------------

    def _restore_daily_state(self) -> None:
        """
        Load persisted daily state from disk.  Guards against the #1 production
        bug: every restart previously reset _daily_trade_count=0 and
        _entry_cooldown={}, silently bypassing all overtrading guards.

        State is keyed by UTC calendar date.  If the saved state is from a
        previous day it is discarded (daily reset is intentional).

        Skipped in simulation mode — tests and back-tests always start fresh
        so that real-world production state from logs/ doesn't pollute test runs.
        """
        if self.executor.simulation_mode:
            return   # simulation / test: never inherit production state
        today = datetime.utcnow().strftime("%Y-%m-%d")
        try:
            raw = json.loads(self._state_file.read_text())
            if raw.get("date") != today:
                self.logger.info(
                    f"[State] Stale state from {raw.get('date')} — starting fresh for {today}"
                )
                return
            self._daily_trade_count      = int(raw.get("daily_trade_count", 0))
            self._entry_cooldown         = {k: float(v) for k, v in raw.get("entry_cooldown", {}).items()}
            self._start_of_day_balance   = float(raw.get("start_of_day_balance", 0.0))
            self._start_of_day_date      = raw.get("start_of_day_date", "")
            self._start_of_week_balance  = float(raw.get("start_of_week_balance", 0.0))
            self._start_of_week_date     = raw.get("start_of_week_date", "")
            self._weekend_close_active   = bool(raw.get("weekend_close_active", False))
            self.logger.info(
                f"[State] Restored: daily_trades={self._daily_trade_count}, "
                f"cooldowns={len(self._entry_cooldown)}, "
                f"sod_balance={self._start_of_day_balance:.2f}"
            )
        except FileNotFoundError:
            pass   # first start of the day — no state file yet
        except Exception as exc:
            self.logger.warning(f"[State] Could not restore state ({exc}) — starting fresh")

    def _persist_daily_state(self) -> None:
        """Write current daily state to disk so it survives a restart."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        try:
            payload = {
                "date":                  today,
                "daily_trade_count":     self._daily_trade_count,
                "entry_cooldown":        self._entry_cooldown,
                "start_of_day_balance":  self._start_of_day_balance,
                "start_of_day_date":     self._start_of_day_date,
                "start_of_week_balance": self._start_of_week_balance,
                "start_of_week_date":    self._start_of_week_date,
                "weekend_close_active":  self._weekend_close_active,
            }
            self._state_file.write_text(json.dumps(payload))
        except Exception as exc:
            self.logger.warning(f"[State] Could not persist state: {exc}")

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    def _setup_signal_handlers(self):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return  # No running loop (e.g. in tests) — skip signal handler setup
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_shutdown)
            except NotImplementedError:
                pass  # Windows

    def _handle_shutdown(self):
        self.logger.info("Shutdown signal received — stopping after current cycle")
        self._running = False
        self.executor.shutdown()


# ---------------------------------------------------------------------------
# CLI entry point:  python -m tradingagents_v2.runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    runner = TradingRunner()
    asyncio.run(runner.run_forever())
