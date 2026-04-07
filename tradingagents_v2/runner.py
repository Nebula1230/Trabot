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
from .core.ranking import RankCandidate, RankConfig, rank_and_select
from .agents import (
    RegimeAgent, TrendAgent, MomentumAgent,
    MeanReversionAgent, VolatilityAgent, BreadthAgent, PatternAgent,
    IntermarketAgent, SessionBreakoutAgent, DivergenceAgent,
    ScalpingAgent, VwapScalpAgent, SqueezeBreakoutAgent, OrderFlowAgent,
    CorrelationAgent, LLMSentimentAgent,
)
from .config.settings import TradingConfig
from .data.loader import DataLoader
from .execution.mt5_executor import MT5Executor
from .execution.order_manager import OrderManager
from .execution.trailing_stop import TrailingStopManager
from .monitoring.journal import TradeJournal
from .monitoring.agent_tracker import AgentCalibrationTracker
from .monitoring.adaptive_weights import AdaptiveWeightManager


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

    def maybe_refresh(self) -> None:
        """Refresh the calendar if TTL has expired (called from sync or async context)."""
        if time.time() - self._fetched_at > self._TTL:
            self._refresh()

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
        CorrelationAgent(),      # cross-symbol correlation divergences
    ]
    for agent in all_agents:
        if config.is_agent_enabled(agent.name):
            agent.weight = config.get_agent_weight(agent.name)
            registry.register(agent)

    # Optional LLM agent bridge (tradingagents package)
    llm_cfg = config.llm_agents
    if llm_cfg.enabled:
        llm_agent = LLMSentimentAgent(
            weight=llm_cfg.weight,
            analysts=list(llm_cfg.analysts),
            throttle_hours=llm_cfg.throttle_hours,
            timeout_seconds=llm_cfg.timeout_seconds,
            max_retries=llm_cfg.max_retries,
            llm_config={
                "llm_provider":        llm_cfg.llm_provider,
                "deep_think_llm":      llm_cfg.deep_think_llm,
                "quick_think_llm":     llm_cfg.quick_think_llm,
                "backend_url":         llm_cfg.backend_url,
                "upstream_path":       llm_cfg.upstream_path,
                # CB thresholds forwarded so _ensure_graph can sync them
                "cb_fail_threshold":   llm_cfg.cb_fail_threshold,
                "cb_cooldown_minutes": llm_cfg.cb_cooldown_minutes,
            },
        )
        registry.register(llm_agent)
        _rlog = logging.getLogger("runner")
        _rlog.info(
            "LLMSentimentAgent enabled — "
            f"analysts={llm_cfg.analysts}, throttle={llm_cfg.throttle_hours}h, "
            f"timeout={llm_cfg.timeout_seconds}s, retries={llm_cfg.max_retries}, "
            f"provider={llm_cfg.llm_provider}, "
            f"cb_threshold={llm_cfg.cb_fail_threshold}, "
            f"cb_cooldown={llm_cfg.cb_cooldown_minutes}min"
        )

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
            "sl":         p.get("sl", 0.0),
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

        # ── Concurrency guards ────────────────────────────────────────────────
        # All three loops (signal, surveillance, watchdog) share `self.executor`
        # and related state.  These asyncio.Lock instances prevent:
        #   • Watchdog replacing executor mid-call (use-after-free)
        #   • Surveillance + signal racing on position modifications
        #   • Shared flag/dict corruption across loops
        self._executor_lock = asyncio.Lock()   # guards all executor operations
        self._state_lock    = asyncio.Lock()   # guards mutable flags/dicts

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
            debug_decisions=j.debug_decisions,
        )

        # Agent calibration tracker — records per-agent vote outcomes
        self.agent_tracker = AgentCalibrationTracker(log_dir=j.log_dir)

        # Adaptive weight manager — periodically updates agent.weight based on
        # closed-trade hit-rate statistics collected by agent_tracker.
        self.adaptive_weight_mgr = AdaptiveWeightManager(
            registry=registry,
            cal_tracker=self.agent_tracker,
            config=self.config.adaptive_weights,
        )

        # Trailing stop manager — read tuning from optional trailing: config block
        trailing_cfg = self.config.model_dump().get("trailing", {})
        # Collect per-symbol trailing overrides for live mode (e.g. forex early_be_r)
        _sym_trail_ov = {}
        for _sym, _ov in self.config.model_dump().get("symbol_overrides", {}).items():
            if "trailing" in _ov:
                _sym_trail_ov[_sym] = _ov["trailing"]
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
            early_be_r=float(trailing_cfg.get("early_be_r", 0.0)),
            be_buffer_r=float(trailing_cfg.get("be_buffer_r", 0.0)),
            stale_sl_hours=float(trailing_cfg.get("stale_sl_hours", 0.0)),
            stale_sl_r_mult=float(trailing_cfg.get("stale_sl_r_mult", 0.75)),
            profile=self.config.profile,
            sym_trailing_overrides=_sym_trail_ov,
        )

        # RL exit policy — learned or heuristic-based exit decisions
        from .execution.rl_exit_policy import RLExitPolicy
        _rl_cfg = self.config.model_dump().get("rl_exit_policy", {})
        self.rl_exit_policy = RLExitPolicy(_rl_cfg)

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

        # ── Same-direction losing streak guard ────────────────────────────────
        # Mirrors the backtest engine logic: after 2 consecutive same-direction
        # losses on a symbol, block that direction for an escalating cooldown
        # (4h → 8h → 16h → 24h, doubles each re-trigger, capped at 24h).
        self._streak_history: dict = {}        # symbol → [(dir, pnl, epoch), ...]
        self._streak_block_until: dict = {}    # symbol → epoch seconds when block expires
        self._streak_block_dir: dict = {}      # symbol → blocked direction ("long"/"short")
        self._streak_block_count: dict = {}    # symbol → consecutive streak count

        # Per-symbol daily SL counter — mirrors backtest engine's day_sl_count.
        # Reset each UTC calendar day in _run_cycle; incremented by the
        # surveillance loop when a position hits its stop-loss.
        self._daily_sl_count: dict = {}        # symbol → int

        # Counter-trend scalp tickets — used for shorter time-stop (default 2h)
        self._ct_tickets: set = set()

        # ── Ticket reconciliation: detect MT5-native SL/TP closes ─────────
        # MT5 silently removes positions hit by SL/TP.  Without reconciliation,
        # _daily_sl_count and _streak_history are never updated for those exits.
        self._known_tickets: dict = {}   # ticket → {"symbol": str, "type": str}

        # ── Persist / restore daily guards across restarts ────────────────────
        # Without this: every restart resets _daily_trade_count=0 and
        # _entry_cooldown={}, completely bypassing the overtrading guards.
        # Root cause of 1556 scalp trades in one session (guard always read 0).
        self._state_file = Path(
            self.journal.log_dir / f"_bot_state_{self.config.profile}.json"
        )
        self._restore_daily_state()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def run_forever(self, interval_seconds: int = None,
                          surveillance_interval: int = None,
                          watchdog_interval: int = None):
        """
        Run two concurrent loops:

        Fast loop  (surveillance_interval, default from config or 60s):
          — Checks every open position for SL lock-in and agent-based exit signals.
          — Uses only live bid/ask price to evaluate SL staging (no bar loading).
          — Agent exit check still uses bars, but is lightweight (exit_check_only).

        Slow loop  (interval_seconds, default from config or 300s):
          — Full multi-timeframe agent analysis for every symbol.
          — Places new entry orders when alignment conditions are met.

        Watchdog   (watchdog_interval, default 90s):
          — Reconnects MT5 if the connection drops.
        """
        if interval_seconds is None:
            interval_seconds = self.config.interval_seconds
        rt_cfg = self.config.model_dump().get("realtime", {})
        if surveillance_interval is None:
            _explicit = rt_cfg.get("surveillance_interval_seconds")
            if _explicit is not None:
                surveillance_interval = int(_explicit)
            else:
                # Auto-derive from mid timeframe: faster mid TF → faster surveillance
                _tf_cfg = self.config.timeframes
                _mid_tf = _tf_cfg.mid[0] if _tf_cfg.mid else "1H"
                _surv_by_tf = {
                    "1m": 10, "5m": 15, "15m": 30, "30m": 45,
                    "1H": 60, "4H": 90, "1D": 120,
                }
                surveillance_interval = _surv_by_tf.get(_mid_tf, 60)
        # Store mid TF for cooldown TF-scaling in _run_cycle
        _tf_cfg_cd = self.config.timeframes
        self._mid_tf = _tf_cfg_cd.mid[0] if _tf_cfg_cd.mid else "1H"
        if watchdog_interval is None:
            watchdog_interval = int(rt_cfg.get("watchdog_interval_seconds", 90))

        self._running = True
        self._cycle_num = 0
        self._setup_signal_handlers()
        self.logger.info(
            f"TradingRunner started — symbols: {self.config.symbols} | "
            f"profile: {self.config.profile} | "
            f"signal: {interval_seconds}s | surveillance: {surveillance_interval}s | "
            f"watchdog: {watchdog_interval}s"
        )
        # Run all three loops concurrently; if any crashes fatally it raises
        # and asyncio.gather re-raises, letting the outer try/except log it.
        try:
            await asyncio.gather(
                self._surveillance_loop(surveillance_interval),
                self._signal_loop(interval_seconds),
                self._watchdog_loop(check_interval=watchdog_interval),
            )
        finally:
            self.executor.shutdown()

    # ------------------------------------------------------------------
    # Watchdog — MT5 connection health + autonomous recovery
    # ------------------------------------------------------------------

    async def _watchdog_loop(self, check_interval: int = 90) -> None:
        """
        Runs every *check_interval* seconds.
        - Verifies MT5 connection is alive (account_info ping).
        - If disconnected, attempts up to 5 reconnects with exponential back-off.
        - Evaluates open trades: stale-SL tightening, per-position health.
        - Logs a heartbeat every 10 minutes so external monitors can detect silence.
        """
        _MAX_RECONNECT  = 5
        _BACKOFF_BASE   = 10   # seconds — doubles each attempt
        _HEARTBEAT_SECS = 600  # log "alive" every 10 min
        last_heartbeat  = 0.0

        # Trade evaluation config (read once)
        _trailing_cfg  = self.config.model_dump().get("trailing", {})
        _stale_hours   = float(_trailing_cfg.get("stale_sl_hours", 0))
        _stale_r_mult  = float(_trailing_cfg.get("stale_sl_r_mult", 0.75))

        while self._running:
            await asyncio.sleep(check_interval)
            try:
                import time as _time
                now = _time.time()

                # ── Heartbeat ─────────────────────────────────────────────────
                if now - last_heartbeat >= _HEARTBEAT_SECS:
                    async with self._executor_lock:
                        acct = self.executor.get_account_info()
                    equity = acct.get("equity", "?") if acct else "?"
                    self.logger.info(
                        f"[WATCHDOG] alive | equity={equity} | "
                        f"cycle={self._cycle_num} | profile={self.config.profile}"
                    )
                    last_heartbeat = now

                # ── MT5 connection check ───────────────────────────────────────
                if self.executor.simulation_mode:
                    pass   # nothing to reconnect in simulation
                else:
                    async with self._executor_lock:
                        acct = self.executor.get_account_info()
                    if acct is None:
                        # Connection lost — attempt reconnect
                        await self._reconnect_mt5(_BACKOFF_BASE, _MAX_RECONNECT)

                # ── Trade health evaluation ────────────────────────────────────
                # Lightweight per-position checks that don't require agent re-runs.
                # Complementary to surveillance (which checks signal conditions).
                if _stale_hours > 0:
                    async with self._executor_lock:
                        _positions = self.executor.get_open_positions()
                    _our = [p for p in _positions
                            if p.get("magic") == self.executor.magic_number]
                    import time as _tw
                    _now_ts = _tw.time()
                    for _wp in _our:
                        _ticket = _wp.get("ticket")
                        _sym    = _wp.get("symbol", "?")
                        _type   = _wp.get("type", "").upper()
                        _entry  = _wp.get("price_open", 0.0)
                        _price  = _wp.get("price_current", 0.0)
                        _sl     = _wp.get("sl", 0.0)
                        _open_t = _wp.get("open_time", 0)
                        if not (_entry and _sl and _open_t and _ticket):
                            continue
                        # Use original SL from _known_tickets (immutable, set at
                        # entry) so that prior trailing-SL moves don't shrink 1R.
                        # Falls back to current SL for positions opened before
                        # this runner session (unknown original SL).
                        _kt_info = self._known_tickets.get(_ticket)
                        _orig_sl = (_kt_info.get("stop_loss", 0.0)
                                    if _kt_info and "stop_loss" in _kt_info
                                    else 0.0)
                        _one_r = abs(_entry - _orig_sl) if _orig_sl else abs(_entry - _sl)
                        if _one_r < 1e-9:
                            continue
                        _profit = (_price - _entry) if _type == "BUY" else (_entry - _price)
                        _hours  = (_now_ts - _open_t) / 3600.0

                        # Stale-SL tightening: trade open > N hours and still losing.
                        # Cap SL at -stale_r_mult × 1R instead of full -1R.
                        if _hours >= _stale_hours and _profit < 0:
                            if _type == "BUY":
                                _tight_sl = _entry - _stale_r_mult * _one_r
                                if _sl < _tight_sl - 1e-6:
                                    async with self._executor_lock:
                                        self.executor.modify_stop_loss(_ticket, _tight_sl)
                                    self.logger.info(
                                        f"[WATCHDOG] STALE-SL BUY {_sym} #{_ticket}: "
                                        f"open {_hours:.1f}h, SL {_sl:.5f}→{_tight_sl:.5f} "
                                        f"(-{_stale_r_mult:.2f}R cap)"
                                    )
                            elif _type == "SELL":
                                _tight_sl = _entry + _stale_r_mult * _one_r
                                if _sl == 0.0 or _sl > _tight_sl + 1e-6:
                                    async with self._executor_lock:
                                        self.executor.modify_stop_loss(_ticket, _tight_sl)
                                    self.logger.info(
                                        f"[WATCHDOG] STALE-SL SELL {_sym} #{_ticket}: "
                                        f"open {_hours:.1f}h, SL {_sl:.5f}→{_tight_sl:.5f} "
                                        f"(-{_stale_r_mult:.2f}R cap)"
                                    )

                        # Per-position health log: warn when close to SL hit
                        _r_mult = _profit / _one_r
                        if _r_mult < -0.8:
                            self.logger.warning(
                                f"[WATCHDOG] DANGER {_sym} #{_ticket} {_type}: "
                                f"{_r_mult:+.2f}R ({_hours:.1f}h open) — approaching SL"
                            )

            except Exception as exc:
                self.logger.error(f"[WATCHDOG] Unexpected error: {exc}")

    async def _reconnect_mt5(self, backoff_base: float, max_attempts: int) -> None:
        """Attempt to reconnect to MT5 with exponential back-off."""
        self.logger.warning("[WATCHDOG] MT5 connection lost — attempting reconnect")
        delay = backoff_base
        reconnected = False
        for attempt in range(1, max_attempts + 1):
            self.logger.info(
                f"[WATCHDOG] Reconnect attempt {attempt}/{max_attempts} "
                f"(waiting {delay}s)"
            )
            await asyncio.sleep(delay)
            try:
                async with self._executor_lock:
                    self.executor.shutdown()
            except Exception:
                pass
            try:
                cfg = self.config.mt5.model_dump()
                from .execution.mt5_executor import MT5Executor
                async with self._executor_lock:
                    self.executor = MT5Executor(cfg)
                    self.data_loader._executor = self.executor
                    self.data_loader.simulation = self.executor.simulation_mode
                    self.data_loader._tf_map    = self.executor.get_tf_map()
                    self.trailing_stop_mgr.executor = self.executor
                    self.graph.executor = self.executor
                    if self.executor.get_account_info() is not None:
                        self.logger.info("[WATCHDOG] MT5 reconnected successfully")
                        reconnected = True
                if reconnected:
                    break
            except Exception as exc:
                self.logger.error(f"[WATCHDOG] Reconnect attempt {attempt} failed: {exc}")
            delay = min(delay * 2, 120)
        if not reconnected:
            self.logger.critical(
                "[WATCHDOG] Could not reconnect to MT5 after "
                f"{max_attempts} attempts — bot will keep retrying next cycle"
            )

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
                        # Friday past market close — close everything, retry each
                        # surveillance tick until no bot positions remain.
                        async with self._state_lock:
                            self._weekend_close_active = True
                        async with self._executor_lock:
                            _bot_open = [
                                p for p in self.executor.get_open_positions()
                                if p.get("magic") == self.executor.magic_number
                            ]
                        if _bot_open:
                            async with self._executor_lock:
                                await self._close_all_positions(
                                    f"Friday weekend gap protection (past {_cutoff}:00 UTC)"
                                )
                    elif _now_utc.weekday() in (0, 1, 2, 3):
                        # Monday–Thursday: markets open → allow entries again
                        async with self._state_lock:
                            self._weekend_close_active = False

                # Acquire executor lock for all position management operations
                # so watchdog cannot swap the executor mid-call.
                async with self._executor_lock:
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
                            # Auto-correct a stale SOD balance (e.g. from a previous
                            # backtest/demo session with a different account size).
                            if (
                                _equity_now > 0
                                and (
                                    self._start_of_day_balance / _equity_now > 5.0
                                    or self._start_of_day_balance / _equity_now < 0.2
                                )
                            ):
                                self.logger.warning(
                                    f"[State] SOD balance {self._start_of_day_balance:.2f} "
                                    f"is implausible vs live equity {_equity_now:.2f} — "
                                    "re-anchoring automatically"
                                )
                                self._start_of_day_balance  = _equity_now
                                self._start_of_week_balance = _equity_now
                                self._persist_daily_state()
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

                    # ── Weekly drawdown circuit breaker: close ALL open positions
                    # when the weekly drawdown ceiling is breached.  Mirrors the
                    # backtest engine's weekly_dd_close_all behaviour.
                    _max_wk_pct = float(_dd_cfg.get("max_weekly_drawdown_pct", 0))
                    if _max_wk_pct > 0 and self._start_of_week_balance > 0:
                        _acct_wk = self.executor.get_account_info()
                        if _acct_wk is not None:
                            _eq_wk = _acct_wk.get("equity", self._start_of_week_balance)
                            _wk_dd = (_eq_wk - self._start_of_week_balance) / self._start_of_week_balance
                            if _wk_dd < -(_max_wk_pct / 100):
                                self.logger.warning(
                                    f"[WEEKLY DD CLOSE-ALL] Weekly drawdown {_wk_dd*100:.2f}% "
                                    f"breached -{_max_wk_pct:.2f}% ceiling — "
                                    "closing all open positions"
                                )
                                await self._close_all_positions(
                                    f"weekly drawdown circuit breaker: {_wk_dd*100:.2f}% < -{_max_wk_pct:.2f}%"
                                )

                    # 2b. Reconcile broker-closed positions (SL/TP hit by MT5)
                    #     Updates streak history, daily SL cap, CT tickets.
                    async with self._state_lock:
                        self._reconcile_closed_tickets()

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
                async with self._executor_lock:
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
        Retries failed closes up to 3 times to prevent weekend gap exposure.
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
        # Derive a clean exit_reason tag for the closed-trade journal
        _reason_lc = reason.lower()
        if "weekend" in _reason_lc:
            _exit_tag = "weekend"
        elif "weekly" in _reason_lc:
            _exit_tag = "weekly_dd_close_all"
        elif "drawdown" in _reason_lc:
            _exit_tag = "dd_close_all"
        else:
            _exit_tag = "close_all"
        failed: list = []
        for pos in open_positions:
            symbol = pos["symbol"]
            ticket = pos["ticket"]
            ok = self.executor.close_position(ticket)
            if ok:
                self.logger.info(
                    f"  Closed {symbol} #{ticket}  pnl={pos['profit']:+.2f}"
                )
                _dir_ca = "long" if pos.get("type") == "BUY" else "short"
                self._record_exit(
                    ticket=ticket,
                    symbol=symbol,
                    direction=_dir_ca,
                    exit_price=pos.get("price_current", 0.0),
                    pnl=pos.get("profit", 0.0),
                    exit_reason=_exit_tag,
                )
                async with self._state_lock:
                    self._known_tickets.pop(ticket, None)
                    self._ct_tickets.discard(ticket)
                self.agent_tracker.score_closed_trade(ticket, pos.get("profit", 0.0))
                self.journal.record_cycle(symbol, {
                    "decision":     "closed",
                    "executed":     True,
                    "order_ticket": ticket,
                    "errors":       [],
                })
            else:
                self.logger.error(f"  Failed to close {symbol} #{ticket}")
                failed.append(pos)
        # Retry failed closes (network blip, requote, etc.)
        for attempt in range(1, 3):
            if not failed:
                break
            await asyncio.sleep(2)
            still_failed: list = []
            for pos in failed:
                ok = self.executor.close_position(pos["ticket"])
                if ok:
                    self.logger.info(
                        f"  Retry #{attempt}: closed {pos['symbol']} #{pos['ticket']}"
                    )
                    _dir_rt = "long" if pos.get("type") == "BUY" else "short"
                    self._record_exit(
                        ticket=pos["ticket"],
                        symbol=pos["symbol"],
                        direction=_dir_rt,
                        exit_price=pos.get("price_current", 0.0),
                        pnl=pos.get("profit", 0.0),
                        exit_reason=_exit_tag,
                    )
                    async with self._state_lock:
                        self._known_tickets.pop(pos["ticket"], None)
                        self._ct_tickets.discard(pos["ticket"])
                    self.agent_tracker.score_closed_trade(pos["ticket"], pos.get("profit", 0.0))
                else:
                    still_failed.append(pos)
            failed = still_failed
        if failed:
            tags = [f"{p['symbol']}#{p['ticket']}" for p in failed]
            self.logger.error(
                f"[CLOSE ALL] {len(failed)} position(s) STILL OPEN after retries: {tags}"
            )

    def _record_exit(
        self,
        *,
        ticket: int,
        symbol: str,
        direction: str,
        exit_price: float,
        pnl: float,
        exit_reason: str,
    ) -> None:
        """Record a closed trade in ClosedTrade-compatible format.

        Pulls entry-time metadata from ``_known_tickets`` (enriched at order
        placement) and falls back to MT5 position data when unavailable.
        """
        info = self._known_tickets.get(ticket, {})
        entry_price = info.get("entry_price", 0.0)
        stop_loss = info.get("stop_loss", 0.0)
        take_profit = info.get("take_profit", 0.0)
        quantity = info.get("quantity", 0.0)
        risk_amount = info.get("risk_amount", 0.0)
        confidence = info.get("confidence", 0.0)
        win_probability = info.get("win_probability", 0.0)
        agent_votes = info.get("agent_votes", {})
        open_dt = info.get("open_dt")
        entry_type = info.get("entry_type", "full-alignment")

        # Compute pnl_r (P&L in risk units)
        pnl_r = pnl / risk_amount if risk_amount and risk_amount > 1e-9 else 0.0

        self.journal.record_closed_trade(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            risk_amount=risk_amount,
            pnl=pnl,
            pnl_r=pnl_r,
            exit_reason=exit_reason,
            confidence=confidence,
            win_probability=win_probability,
            agent_votes=agent_votes,
            open_dt=open_dt,
            ticket=ticket,
            entry_type=entry_type,
        )

    def _reconcile_closed_tickets(self) -> None:
        """Detect positions closed natively by MT5 (SL/TP hit) and update
        streak history, daily SL counter, and CT ticket tracking.

        Called every surveillance tick inside the executor lock.
        Compares _known_tickets against current open positions; any ticket
        that vanished was closed by the broker.  We then query
        get_closed_trades() to get the P&L and reason.
        """
        current_positions = self.executor.get_open_positions()
        current_tickets = {
            p["ticket"] for p in current_positions
            if p.get("magic") == self.executor.magic_number
        }

        # Snapshot current open tickets into _known_tickets for new ones
        for p in current_positions:
            _t = p["ticket"]
            if p.get("magic") == self.executor.magic_number and _t not in self._known_tickets:
                self._known_tickets[_t] = {
                    "symbol": p["symbol"],
                    "type": p["type"],
                }

        # Find vanished tickets
        # Snapshot to avoid RuntimeError if dict changes during iteration
        vanished = {t: info for t, info in list(self._known_tickets.items())
                    if t not in current_tickets}
        if not vanished:
            return

        # Query recent closed deals to get P&L for vanished tickets
        closed_deals = self.executor.get_closed_trades(days=1)
        if closed_deals is None:
            self.logger.warning("[RECONCILE] Could not fetch closed trades — skipping reconciliation")
            return
        # Build position_id → deal lookup.  MT5 deal tickets differ from
        # position tickets; position_id links back to the original position.
        deal_by_pos_id: dict = {}
        for d in closed_deals:
            _pid = d.get("position_id")
            if _pid:
                deal_by_pos_id[_pid] = d

        for ticket, info in vanished.items():
            symbol = info["symbol"]
            trade_type = info["type"]
            _dir = "long" if trade_type == "BUY" else "short"

            # Try to find the deal for this position ticket
            deal = deal_by_pos_id.get(ticket)
            pnl = deal["profit"] if deal else 0.0

            self.logger.info(
                f"[RECONCILE] {symbol} #{ticket} {trade_type} vanished from MT5 "
                f"(broker SL/TP hit) — pnl={pnl:+.2f}"
            )

            # Determine exit reason from deal details
            _exit_reason = "broker_closed"
            if deal:
                _deal_comment = str(deal.get("comment", "")).lower()
                if "sl" in _deal_comment or "stop" in _deal_comment:
                    _exit_reason = "sl"
                elif "tp" in _deal_comment or "take" in _deal_comment:
                    _exit_reason = "tp"
            _exit_px = deal["price"] if deal and "price" in deal else 0.0

            # Record in ClosedTrade-compatible journal
            self._record_exit(
                ticket=ticket,
                symbol=symbol,
                direction=_dir,
                exit_price=_exit_px,
                pnl=pnl,
                exit_reason=_exit_reason,
            )

            # Update streak history — append ALL trades (not just losses)
            # so that recent_pnl_r reflects the full win/loss sequence for
            # streak-momentum sizing (parity with engine.py).
            if _exit_reason == "sl":
                self._daily_sl_count[symbol] = self._daily_sl_count.get(symbol, 0) + 1
            self._streak_history.setdefault(symbol, []).append(
                (_dir, pnl, time.time())
            )
            _sh = self._streak_history[symbol]
            if len(_sh) > 50:
                self._streak_history[symbol] = _sh[-50:]
                _sh = self._streak_history[symbol]
            if pnl < 0:
                if len(_sh) >= 2 and _sh[-1][1] < 0 and _sh[-2][1] < 0:
                    _prev_cnt = self._streak_block_count.get(symbol, 0)
                    self._streak_block_count[symbol] = _prev_cnt + 1
                    _streak_base_h = float(getattr(self.config.risk, "streak_cooldown_base_hours", 4.0))
                    _streak_max_h = float(getattr(self.config.risk, "streak_cooldown_max_hours", 24.0))
                    _cd_secs = min(int(_streak_base_h * 3600) * (2 ** _prev_cnt), int(_streak_max_h * 3600))
                    self._streak_block_dir[symbol] = "both"
                    self._streak_block_until[symbol] = time.time() + _cd_secs
                    self.logger.warning(
                        f"STREAK [{symbol}]: 2 consecutive losses (broker SL) "
                        f"— blocking ALL directions for {_cd_secs/3600:.0f}h "
                        f"(streak #{_prev_cnt + 1})"
                    )
            elif pnl >= 0:
                # Win resets the block state but keeps history intact
                # (engine.py retains full history for recent_pnl_r sizing).
                self._streak_block_count.pop(symbol, None)
                self._streak_block_dir.pop(symbol, None)
                self._streak_block_until.pop(symbol, None)

            # Clean up CT ticket tracking
            self._ct_tickets.discard(ticket)

            # Score agent votes
            self.agent_tracker.score_closed_trade(ticket, pnl)

            # Journal
            self.journal.record_cycle(symbol, {
                "decision": "broker_closed",
                "executed": True,
                "order_ticket": ticket,
                "errors": [],
            })

            # Remove from known set
            del self._known_tickets[ticket]

        # Persist state changes from reconciliation (SL counts, streak, etc.)
        if vanished:
            self._persist_daily_state()

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

        # d1_flip_threshold: separate exit key so profiles can require a STRONGER
        # D1 reversal before exiting.  Using long_min_score (entry gate) as the
        # exit threshold caused premature exits when the score crossed the entry
        # bar by a few ticks — the actual trend hadn't reversed yet.
        flip_threshold      = float(exit_cfg.get("d1_flip_threshold",
                                                  al_cfg.get("long_min_score", 0.25)))
        # Conviction fade: close when |D1| drops below this (trend gone flat)
        fade_threshold      = float(exit_cfg.get("conviction_fade_threshold", 0.10))
        # Mid+Short opposition: close when BOTH mid & short oppose by more than this
        opposition_threshold = float(exit_cfg.get("mid_short_opposition_threshold", 0.35))
        ct_mid_flip_threshold = float(exit_cfg.get("ct_mid_flip_threshold", opposition_threshold))
        # Per-symbol ct_mid_flip_threshold overrides (from tuning layer)
        _sym_overrides_cfg = self.config.model_dump().get("symbol_overrides", {})
        # Whether each condition is enabled
        # Default False: conviction_fade is opt-in (must be explicitly enabled in config).
        # Defaulting True was silently force-closing positions on every bar where |dir_long|
        # dipped below 0.10, killing trades that were entered correctly.
        # Matches the backtest engine default (engine.py) to ensure live/backtest parity.
        fade_enabled        = bool(exit_cfg.get("conviction_fade_enabled", False))
        opposition_enabled  = bool(exit_cfg.get("mid_short_opposition_enabled", True))
        # Scalp mode: exit on SHORT-tier reversal instead of D1 conviction
        short_tier_exits    = bool(exit_cfg.get("use_short_tier_exits", False))
        short_flip_thresh   = float(exit_cfg.get("short_flip_threshold", 0.35))

        for pos in open_positions:
            symbol     = pos["symbol"]
            trade_type = pos["type"]  # "BUY" or "SELL"
            try:
                features_by_tf = self.data_loader.get_multi_features(symbol)
                if features_by_tf is None:
                    self.logger.warning(f"Could not load features for {symbol} — skipping exit check")
                    continue
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
                _exit_tag = "signal"   # default; overridden for time-stop

                # ── RL Exit Policy check ──────────────────────────────────────
                # The RL policy gets first say: if it recommends EXIT, skip the
                # hand-tuned rules entirely.  TIGHTEN is handled as a SL move.
                if self.rl_exit_policy.enabled:
                    from .execution.rl_exit_policy import ExitObservation, ExitAction
                    _trade_sign_rl = 1.0 if trade_type == "BUY" else -1.0
                    _open_ts_rl = pos.get("open_time", 0)
                    _hours_rl = (time.time() - _open_ts_rl) / 3600.0 if _open_ts_rl else 0
                    _max_h_rl = float(exit_cfg.get("max_trade_duration_hours", 16))
                    _entry_rl = pos.get("price_open", 0)
                    _sl_rl = pos.get("sl", 0)
                    _one_r_rl = abs(_entry_rl - _sl_rl) if _sl_rl else 1.0
                    _profit_in_r = (pos.get("price_current", _entry_rl) - _entry_rl) * _trade_sign_rl / max(_one_r_rl, 1e-9)
                    _atr_now_rl = features.atr_14 if features and features.atr_14 else 1.0
                    _sig_conf = float(fusion.confidence) if hasattr(fusion, 'confidence') else 0.5

                    _obs = ExitObservation(
                        dir_long=dir_long * _trade_sign_rl,
                        dir_mid=dir_mid * _trade_sign_rl,
                        dir_short=dir_short * _trade_sign_rl,
                        profit_r=_profit_in_r,
                        hours_held=_hours_rl,
                        max_hours=_max_h_rl,
                        atr_ratio=1.0,  # ratio vs entry ATR (simplified)
                        signal_conf=_sig_conf,
                        is_counter_trend=pos.get("ticket") in self._ct_tickets,
                    )
                    _rl_action = self.rl_exit_policy.predict(_obs)
                    if _rl_action == ExitAction.EXIT:
                        should_close = True
                        _exit_tag = "rl-exit"
                        reason = (
                            f"RL exit policy: score triggered "
                            f"(D1={dir_long:.3f} M={dir_mid:.3f} "
                            f"profit={_profit_in_r:.2f}R held={_hours_rl:.1f}h)"
                        )
                    elif _rl_action == ExitAction.TIGHTEN:
                        # Tighten SL to breakeven + buffer
                        _atr_rl = features.atr_14 if features else 0
                        if _atr_rl > 0 and pos.get("sl"):
                            _be_buf_rl = 0.10 * _one_r_rl
                            if trade_type == "BUY":
                                _new_sl_rl = max(pos["sl"], _entry_rl - _be_buf_rl)
                                if _new_sl_rl > pos["sl"]:
                                    self.executor.modify_stop_loss(pos["ticket"], _new_sl_rl)
                                    self.logger.info(
                                        f"RL-TIGHTEN [{symbol} #{pos['ticket']}] "
                                        f"SL: {pos['sl']:.5f}→{_new_sl_rl:.5f}"
                                    )
                            else:
                                _new_sl_rl = min(pos["sl"], _entry_rl + _be_buf_rl) if pos["sl"] > 0 else _entry_rl + _be_buf_rl
                                if pos["sl"] == 0 or _new_sl_rl < pos["sl"]:
                                    self.executor.modify_stop_loss(pos["ticket"], _new_sl_rl)
                                    self.logger.info(
                                        f"RL-TIGHTEN [{symbol} #{pos['ticket']}] "
                                        f"SL: {pos['sl']:.5f}→{_new_sl_rl:.5f}"
                                    )

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
                    _ticket_exit = pos.get("ticket")
                    _is_ct = _ticket_exit in self._ct_tickets

                    # ── Condition 1: D1 flip (CT trades immune — they oppose D1) ──
                    if not _is_ct:
                        if trade_type == "BUY" and dir_long < -flip_threshold:
                            should_close = True
                            reason = f"D1 flipped bearish ({dir_long:.3f})"
                        elif trade_type == "SELL" and dir_long > flip_threshold:
                            should_close = True
                            reason = f"D1 flipped bullish ({dir_long:.3f})"

                    # ── Condition 2: Conviction fade (CT trades immune) ────────
                    if not should_close and not _is_ct and fade_enabled:
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

                    # ── Condition 4: CT mid-flip — dedicated CT exit ───────────
                    # CT trades oppose D1, so D1-flip/fade don't apply.  Instead,
                    # close when the mid-TF flips against the CT trade direction.
                    if not should_close and _is_ct:
                        _trade_sign = 1.0 if trade_type == "BUY" else -1.0
                        _ct_thresh_sym = float(
                            _sym_overrides_cfg.get(symbol, {}).get("exit_rules", {}).get(
                                "ct_mid_flip_threshold", ct_mid_flip_threshold))
                        if dir_mid * _trade_sign < -_ct_thresh_sym:
                            should_close = True
                            reason = f"CT mid flipped against (dir_mid={dir_mid:.3f})"

                # ── Time-stop (adaptive) ─────────────────────────────────────
                # Base time-stop is scaled by volatility and session:
                #   - High ATR (> 1.5× 14-period ATR) → extend by 30% (let volatile moves develop)
                #   - Low ATR  (< 0.7× baseline)      → shorten by 20% (exit stale trades faster)
                #   - Asian session (21:00-06:00 UTC)  → shorten by 25% (low liquidity, signals degrade)
                # Counter-trend scalps use a shorter time-stop (default 2h).
                if not should_close:
                    max_hours = float(exit_cfg.get("max_trade_duration_hours", 0))
                    _ct_max_hours = float(exit_cfg.get("ct_max_hours", 2.0))
                    _ticket = pos.get("ticket")
                    _is_ct = _ticket in self._ct_tickets
                    _eff_hours = _ct_max_hours if _is_ct else max_hours

                    # ── Adaptive time-stop scaling ────────────────────────────
                    _adaptive_ts = bool(exit_cfg.get("adaptive_time_stop", False))
                    if _adaptive_ts and _eff_hours > 0 and features:
                        _atr_now = features.atr_14 if features.atr_14 else 0.0
                        # Use EMA20/close ratio as a baseline volatility proxy
                        _atr_baseline = _atr_now  # simplification: compare to self
                        if hasattr(features, 'atr_ratio') and features.atr_ratio:
                            _atr_ratio = features.atr_ratio
                        else:
                            _atr_ratio = 1.0  # neutral

                        # Volatility scaling
                        if _atr_ratio > 1.5:
                            _eff_hours *= 1.30  # high vol → give more room
                        elif _atr_ratio < 0.7:
                            _eff_hours *= 0.80  # low vol → exit sooner

                        # Session scaling: Asian session = shorter time-stop
                        from datetime import datetime as _dt_cls
                        _utc_hour = _dt_cls.utcnow().hour
                        if _utc_hour >= 21 or _utc_hour < 6:
                            _eff_hours *= 0.75  # Asian session: low liquidity

                    if _eff_hours > 0:
                        open_ts = pos.get("open_time", 0)  # UNIX seconds
                        if open_ts and open_ts > 0:
                            import time as _t
                            hours_open = (_t.time() - open_ts) / 3600.0
                            if hours_open >= _eff_hours:
                                should_close = True
                                _exit_tag = "time-stop"
                                reason = (
                                    f"time-stop{'(CT)' if _is_ct else ''}{'(adaptive)' if _adaptive_ts else ''}: "
                                    f"open {hours_open:.1f}h "
                                    f">= {_eff_hours:.1f}h limit  "
                                    f"pnl={pos['profit']:+.2f}"
                                )

                if should_close:
                    self.logger.info(
                        f"EXIT [{symbol} #{pos['ticket']} {trade_type}  "
                        f"pnl={pos['profit']:+.2f}] — {reason}"
                    )
                    ok = self.executor.close_position(pos["ticket"])
                    if ok:
                        # Record closed trade in ClosedTrade-compatible journal
                        _dir_exit = "long" if trade_type == "BUY" else "short"
                        self._record_exit(
                            ticket=pos["ticket"],
                            symbol=symbol,
                            direction=_dir_exit,
                            exit_price=pos.get("price_current", 0.0),
                            pnl=pos.get("profit", 0.0),
                            exit_reason=_exit_tag,
                        )
                        async with self._state_lock:
                            self._known_tickets.pop(pos["ticket"], None)
                            self._ct_tickets.discard(pos["ticket"])
                        # Score agent votes against the outcome of this trade
                        self.agent_tracker.score_closed_trade(
                            pos["ticket"], pos.get("profit", 0.0)
                        )
                        self.journal.record_cycle(symbol, {
                            "decision": "closed",
                            "executed": True,
                            "order_ticket": pos["ticket"],
                            "errors": [],
                        })
                        # ── Streak guard: record trade (parity with engine.py) ──
                        _pnl = pos.get("profit", 0.0)
                        _dir = "long" if trade_type == "BUY" else "short"
                        self._streak_history.setdefault(symbol, []).append(
                            (_dir, _pnl, time.time())
                        )
                        _sh = self._streak_history[symbol]
                        if len(_sh) > 50:
                            self._streak_history[symbol] = _sh[-50:]
                            _sh = self._streak_history[symbol]
                        if _pnl < 0:
                            # Direction-agnostic: any 2 consecutive losses block
                            # the symbol (CT trades can't bypass the guard)
                            if (len(_sh) >= 2
                                    and _sh[-1][1] < 0 and _sh[-2][1] < 0):
                                _prev_cnt = self._streak_block_count.get(symbol, 0)
                                self._streak_block_count[symbol] = _prev_cnt + 1
                                _streak_base_h = float(getattr(self.config.risk, "streak_cooldown_base_hours", 4.0))
                                _streak_max_h = float(getattr(self.config.risk, "streak_cooldown_max_hours", 24.0))
                                _cd_secs = min(
                                    int(_streak_base_h * 3600) * (2 ** _prev_cnt),
                                    int(_streak_max_h * 3600),
                                )
                                self._streak_block_dir[symbol] = "both"
                                self._streak_block_until[symbol] = time.time() + _cd_secs
                                self.logger.warning(
                                    f"STREAK [{symbol}]: 2 consecutive losses "
                                    f"— blocking ALL directions for {_cd_secs/3600:.0f}h "
                                    f"(streak #{_prev_cnt + 1})"
                                )
                        elif _pnl >= 0:
                            # Win resets block state but keeps history for
                            # recent_pnl_r sizing (engine.py retains full history).
                            self._streak_block_count.pop(symbol, None)
                            self._streak_block_dir.pop(symbol, None)
                            self._streak_block_until.pop(symbol, None)
                        self._persist_daily_state()
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
                        # SL → exact entry price (break-even), matching engine.py
                        if trade_type == "BUY":
                            tight_sl = max(pos_sl, pos_entry)             # exact BE
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

                    # ── In-trade signal re-evaluation ──────────────────────────
                    # Re-check signal strength while trade is open:
                    #   - Signal STRENGTHENED (all 3 tiers agree by > 1.5× entry gate):
                    #     → loosen SL by 0.25×ATR to give the winner more room
                    #   - Signal WEAKENED (D1 < 50% of entry gate but not at flip):
                    #     → tighten SL toward entry by 0.15×ATR to protect profits
                    _reeval_enabled = bool(exit_cfg.get("signal_reeval_enabled", False))
                    if _reeval_enabled and atr > 0:
                        _trade_sign = 1.0 if trade_type == "BUY" else -1.0
                        _d1_aligned = dir_long * _trade_sign
                        _mid_aligned = dir_mid * _trade_sign
                        _short_aligned = dir_short * _trade_sign
                        _entry_gate = float(al_cfg.get("long_min_score", 0.15))

                        # Strong signal: all 3 tiers agree at > 1.5× entry threshold
                        if (_d1_aligned > _entry_gate * 1.5
                                and _mid_aligned > 0.05
                                and _short_aligned > 0.0
                                and pos_profit > 0):
                            # Loosen SL by 0.25×ATR (give winner more room)
                            if trade_type == "BUY":
                                _loose_sl = pos_sl - 0.25 * atr
                                if _loose_sl < pos_sl and pos_sl > 0:
                                    self.logger.info(
                                        f"REEVAL-LOOSEN [{symbol} #{pos['ticket']}] "
                                        f"signal strong (D1={dir_long:.3f} M={dir_mid:.3f} S={dir_short:.3f}) "
                                        f"SL: {pos_sl:.5f}→{_loose_sl:.5f}"
                                    )
                                    self.executor.modify_stop_loss(pos["ticket"], _loose_sl)
                            else:
                                _loose_sl = pos_sl + 0.25 * atr
                                if pos_sl > 0 and _loose_sl > pos_sl:
                                    self.logger.info(
                                        f"REEVAL-LOOSEN [{symbol} #{pos['ticket']}] "
                                        f"signal strong (D1={dir_long:.3f} M={dir_mid:.3f} S={dir_short:.3f}) "
                                        f"SL: {pos_sl:.5f}→{_loose_sl:.5f}"
                                    )
                                    self.executor.modify_stop_loss(pos["ticket"], _loose_sl)

                        # Weakened signal: D1 still in trade direction but below 50% of entry gate
                        elif (0 < _d1_aligned < _entry_gate * 0.5
                              and pos_profit > 0):
                            # Tighten SL by 0.15×ATR to protect profits
                            if trade_type == "BUY":
                                _tight_sl = pos_sl + 0.15 * atr
                                _max_tight = pos_entry  # don't tighten past entry
                                _tight_sl = min(_tight_sl, _max_tight)
                                if _tight_sl > pos_sl:
                                    self.logger.info(
                                        f"REEVAL-TIGHTEN [{symbol} #{pos['ticket']}] "
                                        f"signal weakened (D1={dir_long:.3f}) "
                                        f"SL: {pos_sl:.5f}→{_tight_sl:.5f}"
                                    )
                                    self.executor.modify_stop_loss(pos["ticket"], _tight_sl)
                            else:
                                _tight_sl = pos_sl - 0.15 * atr
                                _min_tight = pos_entry
                                _tight_sl = max(_tight_sl, _min_tight)
                                if pos_sl > 0 and _tight_sl < pos_sl:
                                    self.logger.info(
                                        f"REEVAL-TIGHTEN [{symbol} #{pos['ticket']}] "
                                        f"signal weakened (D1={dir_long:.3f}) "
                                        f"SL: {pos_sl:.5f}→{_tight_sl:.5f}"
                                    )
                                    self.executor.modify_stop_loss(pos["ticket"], _tight_sl)

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

        # Refresh adaptive weights before the symbol loop so all agents in this
        # cycle use the latest calibration-adjusted weights.  The manager's own
        # TTL guard limits actual recomputation to at most every N hours.
        self.adaptive_weight_mgr.update()

        # Pre-refresh news calendar in a thread pool so the blocking HTTP
        # call never freezes the event loop during the per-symbol loop.
        _news_cfg_pre = self.config.model_dump().get("execution", {})
        if int(_news_cfg_pre.get("news_blackout_minutes", 0)) > 0:
            await asyncio.get_event_loop().run_in_executor(
                None, _news_calendar.maybe_refresh
            )

        async with self._executor_lock:
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
                self._daily_sl_count.clear()  # reset per-symbol SL counter
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

        # 3. Daily drawdown entry halt: block new entries when the daily
        #    drawdown ceiling is breached.  Mirrors the engine's _day_halted
        #    flag.  (The surveillance loop independently closes positions via
        #    the DD close-all logic, but this gate prevents *new* entries.)
        _max_dd_pct_cb = float(_risk_cfg_cb.get("max_daily_drawdown_pct", 0))
        if _max_dd_pct_cb > 0 and self._start_of_day_balance > 0 and portfolio is not None:
            _daily_dd = (
                (portfolio.equity - self._start_of_day_balance)
                / self._start_of_day_balance
            )
            if _daily_dd < -(_max_dd_pct_cb / 100):
                self.logger.warning(
                    f"[DAILY HALT] Daily drawdown {_daily_dd * 100:.2f}% breached "
                    f"-{_max_dd_pct_cb:.2f}% ceiling — "
                    "no new entries until tomorrow"
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

        # Phase 2: Process each symbol — graph analysis and order execution.
        # When rank_signals is enabled, we run dry (analysis only) then sort
        # candidates by expected value and execute from best to worst.
        _risk_cfg = self.config.model_dump().get("risk", {})
        _max_daily = int(_risk_cfg.get("max_daily_trades", 0))
        # Scale cooldown by mid-TF granularity (mirrors engine.py):
        # the configured value assumes 1H bars; faster TFs get shorter cooldowns.
        _cooldown_base = float(_risk_cfg.get("entry_cooldown_minutes", 0))
        _cd_scale = {"1m": 0.25, "5m": 0.50, "15m": 0.75, "30m": 0.85}.get(
            getattr(self, "_mid_tf", "1H"), 1.0
        )
        _cooldown_min = max(5.0, _cooldown_base * _cd_scale) if _cooldown_base > 0 else 0
        _cooldown_secs = _cooldown_min * 60.0
        _max_daily_sl = int(_risk_cfg.get("max_daily_sl_per_symbol", 0))
        _rank_signals = bool(getattr(self.config, "rank_signals", False))
        _pending_candidates: list = []   # used only when _rank_signals is True

        for symbol in symbols:
            # Daily trade cap: stop analysing further symbols once limit reached
            if _max_daily > 0 and self._daily_trade_count >= _max_daily:
                self.logger.warning(
                    f"Max daily trades ({_max_daily}) reached — no further entries today"
                )
                break

            # Margin guard: reject entry if free margin < 20% of equity
            # (mirrors engine.py: _free_margin < 0.20 * equity)
            _acct_mg = self.executor.get_account_info()
            if _acct_mg is None:
                self.logger.error(
                    f"{symbol}: cannot fetch account info — blocking entry"
                )
                results[symbol] = {"decision": "error", "executed": False,
                                   "order_ticket": None,
                                   "errors": ["account_info_unavailable"]}
                continue
            _eq_mg = _acct_mg.get("equity", 0.0)
            _fm_mg = _acct_mg.get("free_margin", _eq_mg)
            if _eq_mg > 0 and _fm_mg < 0.20 * _eq_mg:
                self.logger.debug(
                    f"{symbol}: margin guard — free_margin={_fm_mg:.0f} "
                    f"< 20% equity={_eq_mg:.0f} — skipping"
                )
                results[symbol] = {"decision": "margin_guard", "executed": False,
                                   "order_ticket": None, "errors": []}
                continue

            # Per-symbol daily SL cap: halt symbol after N SL exits today
            _sym_sl_cnt = self._daily_sl_count.get(symbol, 0)
            if _max_daily_sl > 0 and _sym_sl_cnt >= _max_daily_sl:
                self.logger.info(
                    f"{symbol}: daily SL cap reached "
                    f"(SL_today={_sym_sl_cnt} >= max={_max_daily_sl}) "
                    f"— skipping entry"
                )
                results[symbol] = {"decision": "daily_sl_cap", "executed": False,
                                   "order_ticket": None, "errors": []}
                continue

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

                # ── Streak guard: compute blocked direction for this symbol ────
                _streak_blk = None
                _blk_dir   = self._streak_block_dir.get(symbol)
                _blk_until = self._streak_block_until.get(symbol, 0)
                if _blk_dir and time.time() < _blk_until:
                    _streak_blk = _blk_dir
                elif _blk_dir:
                    # Cooldown expired — clear block
                    self._streak_block_dir.pop(symbol, None)
                    self._streak_block_until.pop(symbol, None)

                # ── Recent P&L signs for streak-momentum sizing ───────────
                _sym_hist = self._streak_history.get(symbol, [])
                _recent_pnl_r = [1.0 if t[1] > 0 else -1.0
                                 for t in _sym_hist[-6:]]

                state = await self.graph.run(
                    symbol=symbol,
                    features=features,
                    portfolio_state=portfolio,
                    risk_limits=self.risk_limits,
                    features_by_tf=features_by_tf,
                    streak_blocked_dir=_streak_blk,
                    executor_lock=self._executor_lock,
                    recent_pnl_r=_recent_pnl_r,
                    dry_run=_rank_signals,
                )

                results[symbol] = {
                    "decision":     state.get("decision", "stop"),
                    "executed":     state.get("metadata", {}).get("executed", False),
                    "order_ticket": state.get("metadata", {}).get("order_ticket"),
                    "errors":       state.get("errors", []),
                    "block_reason": state.get("metadata", {}).get("block_reason", ""),
                }

                # ── Debug decision log (rich per-pipeline snapshot) ──
                _dbg = state.get("metadata", {}).get("debug_snapshot")
                if _dbg:
                    self.journal.record_decision_debug(symbol, _dbg)

                if _rank_signals:
                    # Collect candidate for deferred ranking; skip symbols
                    # whose pipeline didn't produce a trade plan.
                    tp = state.get("trade_plan")
                    if tp and state.get("decision") != "rejected":
                        _sl_dist = abs(tp.entry_price - tp.stop_loss)
                        _cd_elapsed = None
                        if symbol in self._entry_cooldown:
                            _cd_elapsed = (time.time() - self._entry_cooldown[symbol]) / 60.0
                        _streak_blk_active = bool(
                            self._streak_block_dir.get(symbol)
                            and time.time() < self._streak_block_until.get(symbol, 0)
                        )
                        _pending_candidates.append(RankCandidate(
                            symbol=symbol,
                            ev=tp.recipe.expected_value,
                            confidence=tp.confidence,
                            sl_distance=_sl_dist,
                            day_sl_count=self._daily_sl_count.get(symbol, 0),
                            streak_blocked=_streak_blk_active,
                            last_entry_elapsed_min=_cd_elapsed,
                            payload={"state": state},
                        ))
                    # Journal even if not executed yet
                    self.journal.record_cycle(symbol, results[symbol])
                    continue   # defer execution to ranked fill phase below

                # ── Original FCFS mode: immediate post-execution bookkeeping ──

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
                    # Enrich _known_tickets with full entry data for closed-trade journal
                    if mt5_ticket:
                        _tp = state["trade_plan"]
                        _agent_out = state.get("agent_outputs", {})
                        _votes = {k: float(v.dir_score) for k, v in _agent_out.items() if hasattr(v, "dir_score")} if _agent_out else {}
                        async with self._state_lock:
                            self._known_tickets[mt5_ticket] = {
                                "symbol":          symbol,
                                "type":            "BUY" if _tp.recipe.direction == "long" else "SELL",
                                "entry_price":     _tp.entry_price,
                                "stop_loss":       _tp.stop_loss,
                                "take_profit":     _tp.take_profit,
                                "quantity":        _tp.quantity,
                                "risk_amount":     _tp.risk_amount,
                                "confidence":      _tp.confidence,
                                "win_probability": _tp.recipe.win_probability,
                                "agent_votes":     _votes,
                                "open_dt":         datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                "entry_type":      state.get("metadata", {}).get("entry_type", "full-alignment"),
                            }
                    # Track counter-trend scalp tickets for shorter time-stop
                    if mt5_ticket and state.get("metadata", {}).get("entry_type") == "counter-trend-scalp":
                        async with self._state_lock:
                            self._ct_tickets.add(mt5_ticket)
                    # Record agent votes for calibration tracking
                    if mt5_ticket and state.get("agent_outputs"):
                        direction = str(state["trade_plan"].recipe.direction)
                        self.agent_tracker.record_trade_votes(
                            symbol, mt5_ticket, direction, state["agent_outputs"]
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
                    async with self._executor_lock:
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

        # ── Ranked signal fill: shared ranking logic ────────────────────
        if _rank_signals and _pending_candidates:
            # Refresh portfolio so slot count reflects positions closed by
            # the surveillance loop during the candidate-collection phase.
            async with self._executor_lock:
                _fresh_pf = _portfolio_state_from_executor(self.executor, self.order_manager)
            if _fresh_pf is not None:
                portfolio = _fresh_pf
            _rank_cfg = RankConfig(
                max_concurrent=self.risk_limits.max_concurrent_trades,
                open_position_count=len(portfolio.open_positions) if portfolio else 0,
                max_daily=_max_daily,
                daily_trade_count=self._daily_trade_count,
                cooldown_minutes=_cooldown_min,
                max_daily_sl=_max_daily_sl,
            )
            _accepted = rank_and_select(_pending_candidates, _rank_cfg)
            self.logger.info(
                f"[RANK] {len(_pending_candidates)} candidate(s), "
                f"{len(_accepted)} accepted — "
                + ", ".join(f"{c.symbol}(EV={c.ev:.3f})" for c in _accepted)
            )
            for _cand in _accepted:
              try:
                # Re-check daily limit before executing each candidate.
                # rank_and_select budgets slots, but prior candidates in
                # this loop may have incremented the counter already.
                if _max_daily > 0 and self._daily_trade_count >= _max_daily:
                    self.logger.info(
                        f"[RANK] Daily limit {_max_daily} reached — skipping remaining candidates"
                    )
                    break
                _csym = _cand.symbol
                _cstate = _cand.payload["state"]
                _cstate = await self.graph.execute_deferred(
                    _cstate, executor_lock=self._executor_lock
                )
                # Post-execution bookkeeping (same as FCFS path)
                results[_csym] = {
                    "decision":     _cstate.get("decision", "stop"),
                    "executed":     _cstate.get("metadata", {}).get("executed", False),
                    "order_ticket": _cstate.get("metadata", {}).get("order_ticket"),
                    "errors":       _cstate.get("errors", []),
                    "block_reason": _cstate.get("metadata", {}).get("block_reason", ""),
                }
                self.journal.record_cycle(_csym, results[_csym])
                if _cstate.get("trade_plan") and _cstate.get("metadata", {}).get("executed"):
                    mt5_ticket = _cstate["metadata"].get("order_ticket")
                    self.order_manager.create_order(_cstate["trade_plan"], mt5_ticket=mt5_ticket)
                    self.journal.record_trade(
                        _csym, _cstate["trade_plan"], {"order_ticket": mt5_ticket}
                    )
                    # Enrich _known_tickets with full entry data for closed-trade journal
                    if mt5_ticket:
                        _tp_r = _cstate["trade_plan"]
                        _ao_r = _cstate.get("agent_outputs", {})
                        _votes_r = {k: float(v.dir_score) for k, v in _ao_r.items() if hasattr(v, "dir_score")} if _ao_r else {}
                        async with self._state_lock:
                            self._known_tickets[mt5_ticket] = {
                                "symbol":          _csym,
                                "type":            "BUY" if _tp_r.recipe.direction == "long" else "SELL",
                                "entry_price":     _tp_r.entry_price,
                                "stop_loss":       _tp_r.stop_loss,
                                "take_profit":     _tp_r.take_profit,
                                "quantity":        _tp_r.quantity,
                                "risk_amount":     _tp_r.risk_amount,
                                "confidence":      _tp_r.confidence,
                                "win_probability": _tp_r.recipe.win_probability,
                                "agent_votes":     _votes_r,
                                "open_dt":         datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                "entry_type":      _cstate.get("metadata", {}).get("entry_type", "full-alignment"),
                            }
                    if mt5_ticket and _cstate.get("metadata", {}).get("entry_type") == "counter-trend-scalp":
                        async with self._state_lock:
                            self._ct_tickets.add(mt5_ticket)
                    if mt5_ticket and _cstate.get("agent_outputs"):
                        direction = str(_cstate["trade_plan"].recipe.direction)
                        self.agent_tracker.record_trade_votes(
                            _csym, mt5_ticket, direction, _cstate["agent_outputs"]
                        )
                    self._entry_cooldown[_csym] = time.time()
                    self._daily_trade_count += 1
                    self.logger.info(
                        f"[RANK] {_csym}: trade executed (EV={_cand.ev:.3f}) — "
                        f"daily count {self._daily_trade_count}/{_max_daily or '∞'}"
                    )
                    self._persist_daily_state()
                    # Refresh portfolio for next candidate
                    async with self._executor_lock:
                        _refreshed = _portfolio_state_from_executor(self.executor, self.order_manager)
                    if _refreshed is not None:
                        if self._start_of_day_balance > 0:
                            real_dd = (_refreshed.equity - self._start_of_day_balance) / self._start_of_day_balance
                            _refreshed = _refreshed.model_copy(
                                update={"daily_drawdown": real_dd, "max_daily_drawdown": real_dd}
                            )
                        portfolio = _refreshed
              except Exception as _rank_err:
                self.logger.error(f"Error filling ranked candidate {_cand.symbol}: {_rank_err}")
                results[_cand.symbol] = {"decision": "stop", "executed": False, "errors": [str(_rank_err)], "block_reason": "rank_fill_error"}

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
                # Restore cross-day state that shouldn't reset daily
                self._weekend_close_active  = bool(raw.get("weekend_close_active", False))
                self._start_of_week_balance = float(raw.get("start_of_week_balance", 0.0))
                self._start_of_week_date    = raw.get("start_of_week_date", "")
                self._streak_block_dir   = raw.get("streak_block_dir", {})
                self._streak_block_until = {k: float(v) for k, v in raw.get("streak_block_until", {}).items()}
                self._streak_block_count = {k: int(v) for k, v in raw.get("streak_block_count", {}).items()}
                self._ct_tickets         = set(raw.get("ct_tickets", []))
                self._streak_history = {
                    k: [tuple(e) for e in v]
                    for k, v in raw.get("streak_history", {}).items()
                }
                self._known_tickets = {
                    int(k): v for k, v in raw.get("known_tickets", {}).items()
                }
                return
            self._daily_trade_count      = int(raw.get("daily_trade_count", 0))
            self._entry_cooldown         = {k: float(v) for k, v in raw.get("entry_cooldown", {}).items()}
            self._start_of_day_balance   = float(raw.get("start_of_day_balance", 0.0))
            self._start_of_day_date      = raw.get("start_of_day_date", "")
            self._start_of_week_balance  = float(raw.get("start_of_week_balance", 0.0))
            self._start_of_week_date     = raw.get("start_of_week_date", "")
            self._weekend_close_active   = bool(raw.get("weekend_close_active", False))
            self._daily_sl_count         = {k: int(v) for k, v in raw.get("daily_sl_count", {}).items()}
            self._streak_block_dir       = raw.get("streak_block_dir", {})
            self._streak_block_until     = {k: float(v) for k, v in raw.get("streak_block_until", {}).items()}
            self._streak_block_count     = {k: int(v) for k, v in raw.get("streak_block_count", {}).items()}
            self._ct_tickets             = set(raw.get("ct_tickets", []))
            self._streak_history = {
                k: [tuple(e) for e in v]
                for k, v in raw.get("streak_history", {}).items()
            }
            self._known_tickets = {
                int(k): v for k, v in raw.get("known_tickets", {}).items()
            }
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
                "daily_sl_count":        self._daily_sl_count,
                "streak_block_dir":      self._streak_block_dir,
                "streak_block_until":    self._streak_block_until,
                "streak_block_count":    self._streak_block_count,
                "ct_tickets":            list(self._ct_tickets),
                "streak_history":        {k: v[-50:] for k, v in self._streak_history.items()},
                "known_tickets":         {str(k): v for k, v in self._known_tickets.items()},
            }
            # Atomic write: write to temp file then rename to prevent corruption
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            _tmp = self._state_file.with_suffix(".tmp")
            _tmp.write_text(json.dumps(payload))
            _tmp.replace(self._state_file)
        except Exception as exc:
            self.logger.warning(f"[State] Could not persist state: {exc}")

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    def _setup_signal_handlers(self):
        try:
            loop = asyncio.get_running_loop()
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

_ALL_AGENTS = [
    "RegimeAgent", "TrendAgent", "MomentumAgent", "MeanReversionAgent",
    "VolatilityAgent", "BreadthAgent", "PatternAgent", "IntermarketAgent",
    "SessionBreakoutAgent", "DivergenceAgent", "ScalpingAgent",
    "VwapScalpAgent", "SqueezeBreakoutAgent", "OrderFlowAgent",
    "LLMSentimentAgent",
]


def _build_runner_cli() -> "argparse.Namespace":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m tradingagents_v2.runner",
        description="Run the live trading bot (or dry-run simulation).",
    )
    parser.add_argument(
        "--profile", default="balanced",
        choices=["safe", "balanced", "risky", "risky_equity", "scalp", "hft"],
        help="Risk/strategy profile (default: balanced)",
    )
    parser.add_argument(
        "--config", default="config.demo.yaml",
        help="Path to YAML config file (default: config.demo.yaml)",
    )
    parser.add_argument(
        "--simulation", action="store_true",
        help="Run in simulation / paper-trading mode (no real orders sent to MT5)",
    )
    parser.add_argument(
        "--agents", nargs="+", default=None, metavar="AGENT",
        help=(
            "Run ONLY these agents (space-separated). "
            "Available: " + ", ".join(_ALL_AGENTS) + ". "
            "Mutually exclusive with --disable-agents."
        ),
    )
    parser.add_argument(
        "--disable-agents", nargs="+", default=None,
        dest="disable_agents", metavar="AGENT",
        help="Disable these agents; all others remain active. "
             "Mutually exclusive with --agents.",
    )
    parser.add_argument(
        "--tune-file", nargs="+", default=None, dest="tune_file",
        metavar="FILE",
        help=(
            "Load tuned params. Accepts:\n"
            "  - A single .tuned.json (applies all symbols inside it)\n"
            "  - Multiple SYMBOL:file pairs, e.g. USDJPY:usdjpy.tuned.json EURUSD:eurusd.tuned.json"
        ),
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Override which symbols to trade (space-separated, e.g. USDJPY EURUSD).",
    )
    parser.add_argument(
        "--mid-tf", default=None, dest="mid_tf",
        help="Override the mid timeframe (e.g. 15m, 1H). Applies to all symbols.",
    )
    parser.add_argument(
        "--list-agents", action="store_true", dest="list_agents",
        help="Print all available agent names and exit.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug decision logging: writes a rich JSONL file "
             "(debug_decisions_YYYY-MM-DD.jsonl) with full pipeline detail "
             "for every trade decision (agent votes, fusion, alignment, "
             "recipe, risk check, sizing, execution).",
    )

    # Handle --list-agents before argparse runs full validation
    if "--list-agents" in sys.argv:
        print("Available agents:")
        for a in _ALL_AGENTS:
            print(f"  {a}")
        sys.exit(0)

    return parser.parse_args()


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    args = _build_runner_cli()

    if args.agents and args.disable_agents:
        print("[ERROR] --agents and --disable-agents are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    # Load config and apply profile
    try:
        from tradingagents_v2.config.yaml_config import load_config_from_yaml
        cfg = load_config_from_yaml(args.config, profile=args.profile)
    except Exception as e:
        print(f"[ERROR] Could not load config '{args.config}': {e}", file=sys.stderr)
        sys.exit(1)

    # Apply symbol selection
    if args.symbols:
        cfg.symbols = list(args.symbols)
        logging.getLogger("runner").info(f"Symbol selection: {cfg.symbols}")

    # Apply tuned per-symbol overrides
    if args.tune_file:
        from tradingagents_v2.backtesting.symbol_tuner import SymbolTuner
        merged_overrides: dict = {}
        for entry in args.tune_file:
            if ":" in entry and not entry.startswith("/"):
                # SYMBOL:path format
                sym, path = entry.split(":", 1)
                sym = sym.strip().upper()
                file_overrides = SymbolTuner.load_overrides(path.strip())
                if sym in file_overrides:
                    merged_overrides[sym] = file_overrides[sym]
                elif file_overrides:
                    # File has overrides but not for this symbol — take first
                    merged_overrides[sym] = next(iter(file_overrides.values()))
            else:
                # Plain path — load all symbols from file
                file_overrides = SymbolTuner.load_overrides(entry)
                merged_overrides.update(file_overrides)
        cfg_dict = cfg.model_dump()
        cfg_dict["symbol_overrides"] = merged_overrides
        from tradingagents_v2.config.settings import TradingConfig as _TC
        cfg = _TC(**cfg_dict)
        logging.getLogger("runner").info(
            f"Loaded tuned overrides for {list(merged_overrides.keys())} "
            f"from {args.tune_file}"
        )

    # Apply agent selection
    if args.agents:
        unknown = [a for a in args.agents if a not in _ALL_AGENTS]
        if unknown:
            print(f"[ERROR] Unknown agent(s): {unknown}. Run --list-agents for valid names.",
                  file=sys.stderr)
            sys.exit(1)
        cfg.agents.enabled_agents = list(args.agents)
        logging.getLogger("runner").info(f"Agent selection (only): {cfg.agents.enabled_agents}")
    elif args.disable_agents:
        unknown = [a for a in args.disable_agents if a not in _ALL_AGENTS]
        if unknown:
            print(f"[ERROR] Unknown agent(s): {unknown}. Run --list-agents for valid names.",
                  file=sys.stderr)
            sys.exit(1)
        cfg.agents.enabled_agents = [a for a in _ALL_AGENTS if a not in args.disable_agents]
        logging.getLogger("runner").info(
            f"Agent selection (excluding {args.disable_agents}): {cfg.agents.enabled_agents}"
        )

    # Apply mid-timeframe override
    if args.mid_tf:
        cfg.timeframes.mid = [args.mid_tf]
        logging.getLogger("runner").info(f"Mid timeframe override: {args.mid_tf}")

    # Apply debug decision logging
    if args.debug:
        cfg.journal.debug_decisions = True
        logging.getLogger("runner").info("Debug decision logging enabled")

    runner = TradingRunner(config=cfg, simulation=args.simulation)
    asyncio.run(runner.run_forever())
