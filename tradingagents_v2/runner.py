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
import logging
import signal
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
        # Format: "bot/<profile>" e.g. "bot/scalp", "bot/balanced"
        executor_cfg["order_comment"] = f"bot/{self.config.profile}"
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
                # 1. Staged SL lock-in (uses live price from positions, no bar load)
                self.trailing_stop_mgr.update_trails(self.data_loader)
                # 2. Structural SL/TP ratchet (pivot-based, every surveillance tick)
                #    Skipped when `exit_rules.skip_structural_sl_tp_ratchet` is True
                #    (e.g. scalp profile — ratchet would overwrite tight 1m-ATR targets)
                if not _skip_ratchet:
                    self.trailing_stop_mgr.update_structural_sl_tp(self.data_loader)
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
                self.logger.info(
                    f"New trading day {today_utc} — "
                    f"start-of-day equity: {self._start_of_day_balance:.2f}"
                )
            if self._start_of_day_balance > 0:
                real_dd = (portfolio.equity - self._start_of_day_balance) / self._start_of_day_balance
                portfolio = portfolio.model_copy(
                    update={"daily_drawdown": real_dd, "max_daily_drawdown": real_dd}
                )

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
        for symbol in symbols:
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

                state = await self.graph.run(
                    symbol=symbol,
                    features=features,
                    portfolio_state=portfolio,
                    risk_limits=self.risk_limits,
                    features_by_tf=features_by_tf,
                )

                results[symbol] = {
                    "decision": state.get("decision", "stop"),
                    "executed": state.get("metadata", {}).get("executed", False),
                    "order_ticket": state.get("metadata", {}).get("order_ticket"),
                    "errors": state.get("errors", []),
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
