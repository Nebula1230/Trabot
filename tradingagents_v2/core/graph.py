"""
Main trading graph orchestrator using LangGraph.
"""

from typing import Dict, List, Any, Optional, TypedDict
import asyncio
import logging
import numpy as np
from datetime import datetime

from langgraph.graph import StateGraph, END

from .types import (
    AgentOutput, TimeframeFusion, TradeRecipe, TradePlan, 
    TechnicalFeatures, PortfolioState, RiskLimits
)
from .agent_base import BaseAgent, AgentRegistry
from ..execution.mt5_executor import MT5Executor


def _get_tracer():
    """Lazy import to avoid circular dependency (graph <-> backtesting)."""
    from ..backtesting.debug_tracer import get_tracer
    return get_tracer()


class TradingState(TypedDict, total=False):
    """State object for the trading graph (LangGraph-compatible TypedDict)."""
    symbol: str
    features: Optional[TechnicalFeatures]               # primary (mid) — backward compat
    features_by_tf: Dict[str, TechnicalFeatures]        # per-tier features
    agent_outputs: Dict[str, AgentOutput]
    timeframe_fusion: Optional[TimeframeFusion]
    trade_recipe: Optional[TradeRecipe]
    trade_plan: Optional[TradePlan]
    portfolio_state: Optional[PortfolioState]
    risk_limits: Optional[RiskLimits]
    decision: str
    errors: List[str]
    metadata: Dict[str, Any]


class TradingGraph:
    """Main trading decision graph."""
    
    def __init__(self, agent_registry: AgentRegistry, config: Dict[str, Any] = None,
                 executor: MT5Executor = None, data_loader=None):
        self.agent_registry = agent_registry
        self.config = config or {}
        self.logger = logging.getLogger("TradingGraph")
        self.executor = executor or MT5Executor(config)
        self.data_loader = data_loader   # optional: used for structural SL/TP

        # Kelly criterion tracker — adjusts position size based on per-symbol track record.
        # Disabled by config (kelly.enabled: false) or when executor is in simulation mode.
        from ..risk.kelly_tracker import KellyTracker
        kelly_cfg = self.config.get("kelly", {})
        if kelly_cfg.get("enabled", True) and not getattr(self.executor, "simulation_mode", True):
            self._kelly_tracker = KellyTracker(self.executor, self.config)
        else:
            self._kelly_tracker = None

        # Build the graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the trading decision graph."""
        
        # Create the graph
        workflow = StateGraph(TradingState)
        
        # Add nodes
        workflow.add_node("run_agents", self._run_agents)
        workflow.add_node("fuse_timeframes", self._fuse_timeframes)
        workflow.add_node("check_alignment", self._check_alignment)
        workflow.add_node("generate_recipe", self._generate_recipe)
        workflow.add_node("risk_check", self._risk_check)
        workflow.add_node("create_plan", self._create_plan)
        workflow.add_node("execute_trade", self._execute_trade)
        
        # Define the flow
        workflow.set_entry_point("run_agents")
        workflow.add_edge("run_agents", "fuse_timeframes")
        workflow.add_edge("fuse_timeframes", "check_alignment")
        workflow.add_conditional_edges(
            "check_alignment",
            self._should_continue,
            {
                "continue": "generate_recipe",
                "stop": END
            }
        )
        workflow.add_edge("generate_recipe", "risk_check")
        workflow.add_conditional_edges(
            "risk_check",
            self._risk_approved,
            {
                "approved": "create_plan",
                "rejected": END
            }
        )
        workflow.add_edge("create_plan", "execute_trade")
        workflow.add_edge("execute_trade", END)
        
        return workflow.compile()
    
    async def _run_agents(self, state: TradingState) -> TradingState:
        """Run all enabled agents in parallel via asyncio.gather."""
        try:
            self.logger.info(f"Running agents for {state['symbol']}")
            # Stamp the moment bar data was consumed so _execute_trade can
            # reject the decision if the cycle took too long and the signal is stale.
            import time as _time
            state.setdefault("metadata", {})["signal_born_at"] = _time.time()
            agents = [a for a in self.agent_registry.get_all_agents() if a.enabled]
            features_by_tf: Dict[str, TechnicalFeatures] = state.get("features_by_tf") or {}
            primary_features = state.get("features")

            async def _run_one(agent: BaseAgent):
                # Route agent to its timeframe-specific features when available
                tf_features = features_by_tf.get(agent.timeframe) or primary_features
                # Pass current_price in context so agents can use it for swing breakout detection
                ctx: Dict[str, Any] = {}
                if self.executor is not None:
                    prices = self.executor.get_current_price(state["symbol"])
                    if prices:
                        bid, ask = prices
                        ctx["current_price"] = (bid + ask) / 2.0
                try:
                    result = await agent.run(tf_features, context=ctx)
                    return agent.name, result
                except Exception as e:
                    return agent.name, e

            raw = await asyncio.gather(*[_run_one(a) for a in agents])

            results = {}
            for name, result in raw:
                if isinstance(result, Exception):
                    self.logger.error(f"Agent {name} failed: {result}")
                    state["errors"].append(f"Agent {name}: {result}")
                elif result is not None:
                    results[name] = result

            state["agent_outputs"] = results
            self.logger.info(f"Completed {len(results)} agents")

        except Exception as e:
            self.logger.error(f"Error running agents: {e}")
            state["errors"].append(f"Agent execution: {e}")

        return state

    # BreadthAgent and other market-context agents excluded from tier fusion
    # (they are gates/context, not directional votes)
    _FILTER_AGENTS: frozenset = frozenset({"BreadthAgent"})

    # ── Regime preference for each agent ────────────────────────────────────
    # How much each agent's vote should be amplified (trending) or reduced
    # (ranging) relative to the base weight.
    # Format: {agent_name: (trending_mult, ranging_mult)}
    # trending: trendiness > 0.55 (ADX > 28-ish)
    # ranging:  trendiness < 0.40 (ADX < 20-ish)
    _REGIME_PREFS: Dict[str, tuple] = {
        # trending_mult > 1 = amplified in trends, < 1 = dampened
        # ranging_mult  > 1 = amplified in ranges
        "TrendAgent":           (2.00, 0.40),   # dominant in trending conditions
        "MomentumAgent":        (1.60, 0.60),   # strong trend-following confirmation
        "IntermarketAgent":     (1.50, 0.75),   # macro trend confirmation
        "PatternAgent":         (1.30, 0.75),   # breakout patterns matter in trends
        "RegimeAgent":          (1.10, 1.10),   # always relevant
        "VolatilityAgent":      (1.00, 1.00),   # neutral: captures both
        "SessionBreakoutAgent": (1.30, 0.60),   # breakouts valid in trending sessions
        "BreadthAgent":         (1.00, 1.00),   # kept as filter, not scored
        # Mean reversion FIGHTS the trend — nearly muted when trending.
        # In strong trends (trendiness=0.80), contribution drops to ~10%.
        # In ranging markets it is the dominant signal (1.80×).
        "MeanReversionAgent":   (0.10, 1.80),
        # Regular RSI divergences look bearish in every strong uptrend.
        # They are continuation patterns in trends, not reversals.
        # Mute significantly when trending; boost in ranging for reversals.
        "DivergenceAgent":      (0.20, 1.60),
    }

    async def _fuse_timeframes(self, state: TradingState) -> TradingState:
        """Fuse agent outputs across timeframes."""
        try:
            self.logger.info("Fusing timeframe signals")
            outputs = state["agent_outputs"]

            # BreadthAgent is a market-context filter, not a directional signal.
            FILTER_AGENTS = self._FILTER_AGENTS

            # ── Detect regime for dynamic agent weighting ─────────────────────
            # Use trendiness from RegimeAgent (already computed on D1 features).
            # trending: trendiness > 0.55 → amplify trend-following agents
            # ranging:  trendiness < 0.40 → amplify mean-reversion agents
            trendiness_raw = 0.50
            if "RegimeAgent" in outputs:
                # RegimeAgent runs on MID (1H) features — its trendiness reading
                # reflects the intraday regime, which updates faster than D1 ADX.
                trendiness_raw = outputs["RegimeAgent"].evidence.get("trendiness", 0.50)

            TREND_THRESH  = float(self.config.get("regime_weighting", {}).get("trend_threshold", 0.55))
            RANGE_THRESH  = float(self.config.get("regime_weighting", {}).get("range_threshold", 0.40))
            is_trending   = trendiness_raw > TREND_THRESH
            is_ranging    = trendiness_raw < RANGE_THRESH

            # ── Direct score post-processing for counter-trend agents ──────────
            # MeanReversionAgent and DivergenceAgent produce bearish signals in
            # every sustained uptrend (RSI overbought, bearish divergences) and
            # bullish signals in every downtrend. In a trending market these are
            # INCORRECT direction calls. We mute them proportionally to trendiness
            # BEFORE they enter the weighted-average, so they cannot drag the
            # fusion score away from the true trend direction.
            #
            # At trendiness=0.55 (threshold): no dampening (factor=1.0)
            # At trendiness=0.70:             factor≈0.50 (halved)
            # At trendiness=0.85:             factor≈0.0  (fully muted)
            if is_trending:
                _t_dampen_start = TREND_THRESH
                _t_dampen_range = 0.30       # full mute at trendiness+0.30
                if "MeanReversionAgent" in outputs:
                    _mr_factor = max(0.0, 1.0 - (trendiness_raw - _t_dampen_start) / _t_dampen_range)
                    _mr = outputs["MeanReversionAgent"]
                    outputs["MeanReversionAgent"] = AgentOutput(
                        timeframe=_mr.timeframe,
                        dir_score=float(_mr.dir_score * _mr_factor),
                        conf=_mr.conf,
                        rationale=_mr.rationale + f" [trend_mute×{_mr_factor:.2f}]",
                        evidence=_mr.evidence,
                    )
                if "DivergenceAgent" in outputs:
                    _div_factor = max(0.0, 1.0 - (trendiness_raw - _t_dampen_start) / (_t_dampen_range * 1.5))
                    _div = outputs["DivergenceAgent"]
                    outputs["DivergenceAgent"] = AgentOutput(
                        timeframe=_div.timeframe,
                        dir_score=float(_div.dir_score * _div_factor),
                        conf=_div.conf,
                        rationale=_div.rationale + f" [trend_mute×{_div_factor:.2f}]",
                        evidence=_div.evidence,
                    )
            # ─────────────────────────────────────────────────────────────────

            def _regime_mult(agent_name: str) -> float:
                """Return the regime-adjusted weight multiplier for this agent."""
                prefs = self._REGIME_PREFS.get(agent_name, (1.0, 1.0))
                trend_m, range_m = prefs
                if is_trending:
                    # Blend toward trending multiplier proportionally to how
                    # far above the threshold we are (soft transition)
                    t = min((trendiness_raw - TREND_THRESH) / (1.0 - TREND_THRESH), 1.0)
                    return 1.0 + (trend_m - 1.0) * t
                if is_ranging:
                    t = min((RANGE_THRESH - trendiness_raw) / RANGE_THRESH, 1.0)
                    return 1.0 + (range_m - 1.0) * t
                return 1.0   # neutral regime — no adjustment

            long_outputs, mid_outputs, short_outputs = [], [], []
            for name, output in outputs.items():
                if name in FILTER_AGENTS:
                    continue
                if output.timeframe == "long":
                    long_outputs.append((name, output))
                elif output.timeframe == "mid":
                    mid_outputs.append((name, output))
                elif output.timeframe == "short":
                    short_outputs.append((name, output))

            # Build a lookup of profile-configured agent weights (set by _build_registry)
            _agent_weights = {
                a.name: a.weight
                for a in self.agent_registry.get_all_agents()
            }

            # ── Correlation auto-weighting ────────────────────────────────────
            # Pairs of agents that tend to measure similar things get their
            # combined weight reduced so they don't double-count the same signal.
            # Rule: for each correlated pair (A, B), if both exist in this cycle
            # and their dir_scores agree in sign, reduce each by sqrt(penalty).
            # sqrt keeps the total effect moderate (e.g. 0.80 penalty → each ×0.894).
            # Uses a fixed correlation map calibrated from typical signal overlap.
            _CORR_PAIRS = [
                # (agent_A, agent_B, penalty)  — penalty applied when both same-sign
                ("TrendAgent",      "MomentumAgent",        0.80),
                ("TrendAgent",      "RegimeAgent",           0.85),
                ("MomentumAgent",   "SessionBreakoutAgent",  0.85),
                ("ScalpingAgent",   "VwapScalpAgent",        0.80),
                ("ScalpingAgent",   "SqueezeBreakoutAgent",  0.85),
                ("VwapScalpAgent",  "OrderFlowAgent",        0.85),
                # SHORT-tier overlaps: VolatilityAgent shares ATR-expansion+ROC
                # direction logic with ScalpingAgent and OrderFlowAgent;
                # PatternAgent's swing-break check duplicates ScalpingAgent's.
                ("VolatilityAgent", "ScalpingAgent",         0.85),
                ("VolatilityAgent", "OrderFlowAgent",        0.85),
                ("PatternAgent",    "ScalpingAgent",         0.90),
            ]
            _corr_adj = {}   # agent_name → cumulative multiplicative adjustment
            for agent_a, agent_b, penalty in _CORR_PAIRS:
                if agent_a in outputs and agent_b in outputs:
                    score_a = outputs[agent_a].dir_score
                    score_b = outputs[agent_b].dir_score
                    # Only penalise when both agents agree (same sign + both non-trivial)
                    if score_a * score_b > 0 and abs(score_a) > 0.05 and abs(score_b) > 0.05:
                        adj = penalty ** 0.5   # sqrt: moderate, symmetric reduction
                        _corr_adj[agent_a] = _corr_adj.get(agent_a, 1.0) * adj
                        _corr_adj[agent_b] = _corr_adj.get(agent_b, 1.0) * adj

            if _corr_adj:
                self.logger.debug(
                    f"Corr adj: { {k: f'{v:.3f}' for k, v in _corr_adj.items()} }"
                )

            def weighted_avg(named_outs):
                """
                Compute weighted-average direction score for one timeframe tier.

                Returns (dir_score, mean_conf, consensus) where:
                  dir_score  — confidence-weighted average of agent dir_scores,
                               further dampened when agents disagree within the tier.
                  mean_conf  — arithmetic mean of agent confidences (true 0–1 scale).
                  consensus  — fraction of actively-voting agents (|score|>0.05)
                               that agree with the majority direction (1.0 = full
                               consensus, 0.5 = half disagree).
                """
                if not named_outs:
                    return 0.0, 0.0, 1.0

                total_score = sum(
                    o.dir_score * o.conf * _agent_weights.get(n, 1.0) * _regime_mult(n)
                    * _corr_adj.get(n, 1.0)
                    for n, o in named_outs
                )
                total_weight = sum(
                    o.conf * _agent_weights.get(n, 1.0) * _regime_mult(n)
                    * _corr_adj.get(n, 1.0)
                    for n, o in named_outs
                )
                mean_conf = sum(o.conf for _, o in named_outs) / len(named_outs)

                if total_weight <= 0:
                    return 0.0, mean_conf, 1.0

                raw_score = total_score / total_weight

                # ── Within-tier disagreement dampening ────────────────────────
                # Count agents with a meaningful vote (|score| > 0.05).
                # If many agents contradict the majority direction, dampen the
                # tier score so weak-consensus signals don't trigger trades.
                #   consensus=1.0 → disagree_scale=1.00 (all agree)
                #   consensus=0.5 → disagree_scale=0.65 (half disagree)
                #   consensus=0.0 → disagree_scale=0.30 (maximum disagreement)
                _VT = 0.05
                active = [(n, o) for n, o in named_outs if abs(o.dir_score) > _VT]
                if len(active) > 1:
                    tier_sign = 1.0 if raw_score >= 0 else -1.0
                    votes_maj = sum(1 for _, o in active if o.dir_score * tier_sign > 0)
                    consensus = votes_maj / len(active)
                    disagree_scale = float(np.clip(0.30 + 0.70 * consensus, 0.30, 1.0))
                    raw_score *= disagree_scale
                else:
                    consensus = 1.0   # 0 or 1 active agent — no disagreement

                return raw_score, mean_conf, consensus

            dir_long,  conf_long,  csns_long  = weighted_avg(long_outputs)
            dir_mid,   conf_mid,   csns_mid   = weighted_avg(mid_outputs)
            dir_short, conf_short, csns_short = weighted_avg(short_outputs)

            regime_trendiness = trendiness_raw

            regime_label = ("trending" if is_trending else "ranging" if is_ranging else "neutral")
            self.logger.debug(
                f"Regime: {regime_label} (trendiness={trendiness_raw:.2f}) — "
                f"agent weights adjusted"
            )

            breadth_score = 0.0
            if "BreadthAgent" in outputs:
                breadth_score = outputs["BreadthAgent"].dir_score

            # ── Tier-specific trendiness dampening ────────────────────────────
            # Each tier has its own sensitivity to intraday choppiness:
            #
            #  LONG  (D1): multi-day trend is valid regardless of intraday noise.
            #              Floor 0.85 — only very deep ranging dampens D1 slightly.
            #  MID  (1H): affected when the session is choppy.
            #              Floor 0.65 — moderate dampening.
            #  SHORT (5m/15m): most sensitive to intraday structure.
            #              Floor 0.50 — same as before (full effect).
            dir_long  *= float(np.clip(regime_trendiness / 0.4, 0.85, 1.0))
            dir_mid   *= float(np.clip(regime_trendiness / 0.5, 0.65, 1.0))
            dir_short *= float(np.clip(regime_trendiness / 0.6, 0.50, 1.0))

            # ── Cross-tier alignment strength ─────────────────────────────────
            # Geometric mean of tier magnitudes when all three tiers agree in
            # direction.  0 if any tier contradicts the others.
            # Used downstream to reward high-conviction setups with a win_prob boost.
            if dir_long * dir_mid > 0 and dir_long * dir_short > 0:
                alignment_strength = float(
                    (abs(dir_long) * abs(dir_mid) * abs(dir_short)) ** (1.0 / 3.0)
                )
            else:
                alignment_strength = 0.0

            state["timeframe_fusion"] = TimeframeFusion(
                dir_long=float(np.clip(dir_long, -1.0, 1.0)),
                dir_mid=float(np.clip(dir_mid, -1.0, 1.0)),
                dir_short=float(np.clip(dir_short, -1.0, 1.0)),
                conf_long=float(np.clip(conf_long, 0.0, 1.0)),
                conf_mid=float(np.clip(conf_mid, 0.0, 1.0)),
                conf_short=float(np.clip(conf_short, 0.0, 1.0)),
                regime_trendiness=regime_trendiness,
                breadth_score=breadth_score,
                alignment_strength=float(np.clip(alignment_strength, 0.0, 1.0)),
                consensus_long=float(np.clip(csns_long,  0.0, 1.0)),
                consensus_mid=float(np.clip(csns_mid,   0.0, 1.0)),
                consensus_short=float(np.clip(csns_short, 0.0, 1.0)),
            )
            self.logger.info(
                f"Fusion: L={dir_long:.3f}(csns={csns_long:.2f}), "
                f"M={dir_mid:.3f}(csns={csns_mid:.2f}), "
                f"S={dir_short:.3f}(csns={csns_short:.2f}), "
                f"align_str={alignment_strength:.3f}"
            )
            # ── Debug tracer: record fusion ──
            _tr = _get_tracer()
            _tr.record_fusion(state["timeframe_fusion"])
            _tr.record_agent_votes(state.get("agent_outputs", {}))

        except Exception as e:
            self.logger.error(f"Error fusing timeframes: {e}")
            state["errors"].append(f"Timeframe fusion: {e}")

        return state

    async def _check_alignment(self, state: TradingState) -> TradingState:
        """Check if timeframes are aligned for trading."""
        try:
            fusion = state.get("timeframe_fusion")
            if not fusion:
                state["decision"] = "stop"
                return state

            # Read thresholds from config (alignment: section), with safe defaults
            al_cfg = self.config.get("alignment", {})
            long_min    = float(al_cfg.get("long_min_score",  0.35))
            mid_min     = float(al_cfg.get("mid_min_score",   0.30))
            short_min   = float(al_cfg.get("short_min_score", 0.25))
            breadth_min = float(al_cfg.get("breadth_min",    -0.50))
            pullback_tol = float(al_cfg.get("pullback_tolerance", 0.30))

            # ── Require full alignment option (no pullback entries) ─────────
            # When require_full_alignment=true, the pullback path is disabled.
            # Only setups where ALL THREE tiers agree will be accepted.
            # This is the highest-win-rate configuration but fewest trades.
            require_full_align = bool(al_cfg.get("require_full_alignment", False))

            # ── D1 ADX gate: only enter when a real trend is established ───────
            # ADX < adx_trend_min means the D1 trend is ambiguous / ranging.
            # Trend-following entries in choppy D1 deliver ~35% win rate.
            # Set adx_trend_min: 0 to disable (backward compat default).
            adx_trend_min = float(al_cfg.get("adx_trend_min", 0.0))

            # ── Session-aware threshold scaling ───────────────────────────────
            # Forex liquidity follows the clock.  During the Asian dead-zone
            # (22:00–06:00 UTC) spreads widen and EUR/GBP signals are noisy.
            # Raise the minimum score thresholds so only high-conviction
            # setups pass; during active sessions keep full sensitivity.
            #
            # IMPORTANT: use the bar's own timestamp (backtest) or wall clock (live).
            # Using datetime.now() in backtest makes the dead zone depend on when
            # the backtest is *run*, not on when the bar *occurred* — a test run
            # at 10:00 UTC would never apply the dead zone to any historical bar.
            from datetime import timezone
            _loader_ts = getattr(self.data_loader, "_current_bar_ts", None)
            if _loader_ts is not None:
                # Backtest path: derive UTC hour from the bar's unix timestamp
                utc_hour = datetime.fromtimestamp(int(_loader_ts), tz=timezone.utc).hour
            else:
                # Live path: use the real wall clock
                utc_hour = datetime.now(timezone.utc).hour
            dead_zone_start = int(self.config.get("alignment", {}).get("dead_zone_start_utc", 22))
            dead_zone_end   = int(self.config.get("alignment", {}).get("dead_zone_end_utc",    6))
            dead_zone_factor = float(self.config.get("alignment", {}).get("dead_zone_factor",  1.4))

            in_dead_zone = (
                (dead_zone_start > dead_zone_end and (utc_hour >= dead_zone_start or utc_hour < dead_zone_end))
                or (dead_zone_start < dead_zone_end and dead_zone_start <= utc_hour < dead_zone_end)
            )
            if in_dead_zone:
                long_min  *= dead_zone_factor
                mid_min   *= dead_zone_factor
                short_min *= dead_zone_factor
                self.logger.debug(
                    f"Dead zone active (UTC {utc_hour:02d}:xx) — "
                    f"thresholds ×{dead_zone_factor:.1f}"
                )

            # ── Open zone (London/NY open false-breakout filter) ──────────────
            # During London open (08-09 UTC) and NY open (14-14 UTC) spreads widen
            # and initial moves frequently reverse — require stronger alignment.
            open_zone_start  = int(self.config.get("alignment", {}).get("open_zone_start_utc",  99))
            open_zone_end    = int(self.config.get("alignment", {}).get("open_zone_end_utc",     99))
            open_zone_factor = float(self.config.get("alignment", {}).get("open_zone_factor",   1.0))
            in_open_zone = (
                open_zone_start < open_zone_end
                and open_zone_start <= utc_hour < open_zone_end
            )
            if in_open_zone and open_zone_factor > 1.0:
                long_min  *= open_zone_factor
                mid_min   *= open_zone_factor
                short_min *= open_zone_factor
                self.logger.debug(
                    f"Open zone active (UTC {utc_hour:02d}:xx) — "
                    f"thresholds ×{open_zone_factor:.2f}"
                )

            # ── Full alignment (all 3 TFs agree) ──────────────────────────────
            bull_full = (
                fusion.dir_long  >  long_min and
                fusion.dir_mid   >  mid_min  and
                fusion.dir_short >  short_min
            )
            bear_full = (
                fusion.dir_long  < -long_min and
                fusion.dir_mid   < -mid_min  and
                fusion.dir_short < -short_min
            )

            # ── Pullback entry (strong D1 trend + mild counter-TF pullback) ───
            # Classic "buy the dip in an uptrend" / "sell the rally in a downtrend".
            # Requires a strong D1 signal and neither M nor S excessively against.
            if require_full_align:
                bull_pullback = False
                bear_pullback = False
            else:
                bull_pullback = (
                    fusion.dir_long  >  long_min and
                    fusion.dir_mid   > -pullback_tol and
                    fusion.dir_short > -pullback_tol
                )
                bear_pullback = (
                    fusion.dir_long  < -long_min and
                    fusion.dir_mid   <  pullback_tol and
                    fusion.dir_short <  pullback_tol
                )

            # Market breadth gate (blocks trading in very weak markets)
            breadth_ok = fusion.breadth_score >= breadth_min

            # Asymmetric mean-reversion gate:
            # A bearish MR signal (score < -0.5) blocks longs only;
            # a bullish MR signal (score > +0.5) blocks shorts only.
            mean_rev_score = 0.0
            outputs = state["agent_outputs"]
            if "MeanReversionAgent" in outputs:
                mean_rev_score = outputs["MeanReversionAgent"].dir_score
            _mr_thr = float(al_cfg.get("mean_rev_block_threshold", 0.50))
            mean_rev_blocks_long  = mean_rev_score < -_mr_thr   # bearish reversion → don't buy
            mean_rev_blocks_short = mean_rev_score >  _mr_thr   # bullish reversion → don't sell

            bull_aligned = bull_full or bull_pullback
            bear_aligned = bear_full or bear_pullback

            # Apply directional gates before the combined check
            if mean_rev_blocks_long:  bull_aligned = False
            if mean_rev_blocks_short: bear_aligned = False

            # ── D1 ADX gate ────────────────────────────────────────────────────
            # Block all entries when the long-TF (D1) ADX is below the minimum,
            # indicating a choppy / ranging daily chart where trend signals fail.
            if adx_trend_min > 0 and (bull_aligned or bear_aligned):
                _ftf = state.get("features_by_tf", {})
                _d1_feat = _ftf.get("long") or _ftf.get("1D")
                if _d1_feat is not None:
                    _d1_adx = getattr(_d1_feat, "adx_14", 0.0)
                    if _d1_adx < adx_trend_min:
                        self.logger.info(
                            f"ADX gate: D1 ADX={_d1_adx:.1f} < {adx_trend_min} "
                            f"— trend not established, blocking entry"
                        )
                        bull_aligned = False
                        bear_aligned = False
                        state.setdefault("metadata", {})["block_reason"] = (
                            f"adx_gate:{_d1_adx:.1f}<{adx_trend_min}"
                        )

            # ── Agent consensus gate ─────────────────────────────────────────
            # Require a minimum number of individual agents to vote in the trade
            # direction, regardless of whether tier scores pass their thresholds.
            # This prevents a single high-weight agent from dragging the tier
            # score over the bar while all others are neutral or opposed.
            min_consensus = int(al_cfg.get("min_agent_consensus", 0))
            vote_threshold = float(al_cfg.get("consensus_vote_threshold", 0.05))
            if min_consensus > 0 and (bull_aligned or bear_aligned):
                # Cap the required consensus to the number of directional agents
                # that actually ran this cycle.  When fewer agents are enabled
                # (e.g. --agents TrendAgent) the absolute threshold becomes
                # unachievable and would block every signal.
                _n_directional = sum(
                    1 for name in outputs if name not in self._FILTER_AGENTS
                )
                _effective_min = min(min_consensus, max(1, _n_directional))
                sign = 1.0 if bull_aligned else -1.0
                votes = sum(
                    1 for name, out in outputs.items()
                    if name not in self._FILTER_AGENTS
                    and out.dir_score * sign > vote_threshold
                )
                if votes < _effective_min:
                    self.logger.info(
                        f"Consensus gate: {votes}/{_effective_min} "
                        f"(cfg={min_consensus}, active={_n_directional}) agents voting "
                        f"{'bull' if bull_aligned else 'bear'} — blocked"
                    )
                    bull_aligned = False
                    bear_aligned = False

            if (bull_aligned or bear_aligned) and breadth_ok:
                state["decision"] = "continue"
                if bull_full:
                    entry_type, direction = "full-alignment", "bullish"
                elif bull_pullback:
                    entry_type, direction = "pullback-entry", "bullish"
                elif bear_full:
                    entry_type, direction = "full-alignment", "bearish"
                else:
                    entry_type, direction = "pullback-entry", "bearish"
                meta = state.setdefault("metadata", {})
                meta["entry_type"] = entry_type
                # Store the alignment direction so _generate_recipe uses the
                # correct side instead of re-deriving from dir_long (which
                # can disagree with short-tier scalp signals).
                meta["alignment_direction"] = direction
                self.logger.info(
                    f"Aligned [{entry_type}] ({direction}) — proceeding "
                    f"[L={fusion.dir_long:.3f} M={fusion.dir_mid:.3f} S={fusion.dir_short:.3f} "
                    f"breadth={fusion.breadth_score:.3f}]"
                )
            else:
                reasons = []
                if not bull_aligned and not bear_aligned:
                    reasons.append(f"L={fusion.dir_long:.3f} M={fusion.dir_mid:.3f} S={fusion.dir_short:.3f}")
                if not breadth_ok:
                    reasons.append(f"breadth={fusion.breadth_score:.3f}<{breadth_min}")
                if mean_rev_blocks_long:
                    reasons.append(f"mean-rev blocks LONG (score={mean_rev_score:.2f})")
                if mean_rev_blocks_short:
                    reasons.append(f"mean-rev blocks SHORT (score={mean_rev_score:.2f})")
                state["decision"] = "stop"
                state.setdefault("metadata", {})["block_reason"] = "alignment:" + ";".join(reasons)
                self.logger.info(f"Timeframes not aligned — stopping ({'; '.join(reasons)})")

            # ── Debug tracer: record alignment result ──
            _tr = _get_tracer()
            _meta = state.get("metadata", {})
            _tr.record_alignment(
                aligned=state.get("decision") == "continue",
                atype=_meta.get("entry_type", ""),
                direction=_meta.get("alignment_direction", ""),
                block_reason=_meta.get("block_reason", ""),
            )

        except Exception as e:
            self.logger.error(f"Error checking alignment: {e}")
            state["errors"].append(f"Alignment check: {e}")
            state["decision"] = "stop"

        return state

    def _should_continue(self, state: TradingState) -> str:
        decision = state.get("decision", "stop")
        if decision == "stop":
            _get_tracer().commit_bar()  # bar trace done (alignment failed)
        return decision

    async def _generate_recipe(self, state: TradingState) -> TradingState:
        """Generate trading recipe based on agent outputs."""
        try:
            self.logger.info("Generating trading recipe")
            fusion = state["timeframe_fusion"]
            outputs = state["agent_outputs"]
            symbol = state["symbol"]

            # Use the direction determined by alignment logic (which tier
            # actually fired), not fusion.dir_long (which ignores scalp
            # short-tier dominance and can produce the wrong direction).
            _align_dir = state.get("metadata", {}).get("alignment_direction", "")
            if _align_dir == "bullish":
                direction = "long"
            elif _align_dir == "bearish":
                direction = "short"
            else:
                # Fallback when alignment_direction is missing (legacy path)
                direction = "long" if fusion.dir_long > 0 else "short"
            entry_type = state.get("metadata", {}).get("entry_type", "full-alignment")

            pattern_info = ""
            if "PatternAgent" in outputs:
                pattern_info = outputs["PatternAgent"].rationale

            # ── Win probability + R:R (mode-aware) ────────────────────────────
            # In scalp mode the SHORT tier drives conviction; in swing mode D1 does.
            tight_cfg = self.config.get("tight_sl_tp", {})
            scalp_mode = tight_cfg.get("enabled", False)
            long_strength = abs(fusion.dir_long)     # kept for non-scalp path

            if scalp_mode:
                # Short-tier driven win probability.
                # Recalibrated: previous 0.50+s*0.20 overclaimed (real WR ≈47%).
                # Conservative base 0.47 + moderate boost for strong signals.
                short_strength = abs(fusion.dir_short)
                win_prob = 0.47 + short_strength * 0.15
                # Mid tier alignment bonus/penalty (15m direction matters for 1m trades)
                same_dir = (fusion.dir_mid * fusion.dir_short) > 0
                if same_dir:
                    win_prob += abs(fusion.dir_mid) * 0.06
                else:
                    win_prob -= abs(fusion.dir_mid) * 0.05
            else:
                long_strength = abs(fusion.dir_long)
                mid_strength  = abs(fusion.dir_mid)
                short_strength_v = abs(fusion.dir_short)
                # ── Recalibrated win probability ──────────────────────────────
                # Old formula: 0.50 + long*0.25 → claims ~55% but actual is ~42%.
                # New formula is conservative: each tier contributes proportionally.
                # Full alignment (all same sign): multiplied by alignment_strength.
                # Short-tier same direction: small boost. Short opposed: penalty.
                win_prob = 0.48 + long_strength * 0.18 + mid_strength * 0.06
                short_same = (fusion.dir_short * fusion.dir_long) > 0
                if short_same:
                    win_prob += short_strength_v * 0.04
                else:
                    # Short tier opposed = high risk of immediate reversal
                    win_prob -= short_strength_v * 0.08
                same_dir = (fusion.dir_mid * fusion.dir_long) > 0
                if not same_dir:
                    win_prob -= mid_strength * 0.05

            # Market breadth bonus
            if fusion.breadth_score > 0.3:
                win_prob += 0.03
            elif fusion.breadth_score < -0.3:
                win_prob -= 0.03
            # Trendiness bonus: confirmed trend = better win rate
            if fusion.regime_trendiness > 0.65:
                win_prob += 0.05
            elif fusion.regime_trendiness > 0.55:
                win_prob += 0.02
            elif fusion.regime_trendiness < 0.30:
                win_prob -= 0.06
            # Pullback entries are modestly lower-conviction
            if entry_type == "pullback-entry":
                win_prob -= 0.03
            # ── Cross-tier alignment strength bonus ───────────────────────────
            align_str = fusion.alignment_strength
            if align_str > 0.50:
                win_prob += 0.05
            elif align_str > 0.30:
                win_prob += 0.03
            elif align_str > 0.10:
                win_prob += 0.01
            win_prob = float(np.clip(win_prob, 0.35, 0.80))

            # ── Risk-reward ratio ──────────────────────────────────────────────
            # When tight_sl_tp is active, use its configured R:R (sl/tp mult ratio)
            # so the recipe EV matches what is actually placed on the broker.
            if scalp_mode and tight_cfg.get("sl_atr_mult", 0) > 0:
                rr = float(tight_cfg["tp_atr_mult"]) / float(tight_cfg["sl_atr_mult"])
                rr = float(np.clip(rr, 1.0, 4.0))
            else:
                long_strength = abs(fusion.dir_long)   # may already be set above
                base_rr = 2.0
                rr = base_rr + long_strength * 0.5
                rr = float(np.clip(rr, 1.5, 3.0))

            # ── Spread cost correction ────────────────────────────────────────
            # For a LONG: you enter at ASK and your TP fills at BID (TP_dist - spread),
            # while SL triggers at BID which is already (SL_dist) below entry.
            # Net R:R = (gross_TP - spread) / (gross_SL + spread).
            # We express spread as a fraction of ATR to stay unit-consistent with rr.
            spread_atr = 0.0
            _feat = state.get("features")
            if _feat and _feat.atr_14 > 0 and not self.executor.simulation_mode:
                try:
                    _si = self.executor.get_symbol_info(symbol)
                    if _si is not None:
                        _sp_pts  = getattr(_si, "spread", 0)
                        _pt_size = getattr(_si, "point",  0.00001)
                        spread_atr = (_sp_pts * _pt_size) / _feat.atr_14
                except Exception:
                    pass
            if spread_atr > 0:
                net_rr = float(np.clip(
                    (rr - spread_atr) / max(1.0 + spread_atr, 1e-3),
                    0.1, rr
                ))
            else:
                net_rr = rr

            # ── Expected value: E = P×R:R - (1-P) ─────────────────────────────
            # Uses the spread-adjusted (true) R:R so EV reflects broker costs.
            expected_value = float(win_prob * net_rr - (1.0 - win_prob))

            profile       = self.config.get("profile", "")
            profile_tag   = f"{profile.upper()}_" if profile else ""
            state["trade_recipe"] = TradeRecipe(
                name=f"{profile_tag}{direction.upper()}_{entry_type.upper().replace('-','_')}_{symbol}",
                direction=direction,
                entry_trigger=f"Break of {pattern_info or 'recent high/low'}",
                win_probability=win_prob,
                expected_value=expected_value,
                risk_reward_ratio=net_rr,   # spread-adjusted (honest) R:R
            )
            self.logger.info(f"Generated recipe: {state['trade_recipe'].name}")
            # ── Debug tracer: record recipe ──
            _get_tracer().record_recipe(state["trade_recipe"])

        except Exception as e:
            self.logger.error(f"Error generating recipe: {e}")
            state["errors"].append(f"Recipe generation: {e}")

        return state

    async def _risk_check(self, state: TradingState) -> TradingState:
        """Check risk management rules."""
        state = await self._risk_check_inner(state)
        # ── Debug tracer: record risk result (covers all early-return paths) ──
        if state.get("decision") != "approved":
            _reason = state.get("metadata", {}).get("block_reason", "rejected")
            _get_tracer().record_risk(passed=False, block_reason=_reason)
        return state

    async def _risk_check_inner(self, state: TradingState) -> TradingState:
        """Actual risk check implementation."""
        try:
            recipe = state.get("trade_recipe")
            if not recipe:
                state["decision"] = "rejected"
                return state

            min_win_prob = float(self.config.get("alignment", {}).get("min_win_prob", 0.46))
            min_ev       = float(self.config.get("alignment", {}).get("min_ev",       0.10))

            if recipe.win_probability < min_win_prob:
                self.logger.info(f"Win probability too low: {recipe.win_probability:.3f} < {min_win_prob}")
                state["decision"] = "rejected"
                state.setdefault("metadata", {})["block_reason"] = f"win_prob:{recipe.win_probability:.3f}<{min_win_prob}"
                return state

            if recipe.expected_value < min_ev:
                self.logger.info(f"Expected value too low: {recipe.expected_value:.3f} < {min_ev}")
                state["decision"] = "rejected"
                state.setdefault("metadata", {})["block_reason"] = f"min_ev:{recipe.expected_value:.3f}<{min_ev}"
                return state

            # ── Signal confidence gate ─────────────────────────────────────────
            # Use the SHORT-tier confidence for scalp (where the dominant agents
            # run) so we reject low-conviction 1m signals before sizing.
            # Use average of LONG+MID confidence for swing modes.
            min_conf = float(self.config.get("alignment", {}).get("min_confidence", 0.0))
            if min_conf > 0:
                fusion = state.get("timeframe_fusion")
                if fusion is not None:
                    scalp_mode = bool(self.config.get("tight_sl_tp", {}).get("enabled", False))
                    if scalp_mode:
                        signal_conf = fusion.conf_short
                    else:
                        # Average only the tiers that actually have agents running.
                        # When fewer agents are enabled (e.g. --agents TrendAgent),
                        # empty tiers have conf=0 and would unfairly halve the score.
                        _tier_confs = [c for c in (fusion.conf_long, fusion.conf_mid)
                                       if c > 0]
                        signal_conf = (sum(_tier_confs) / len(_tier_confs)
                                       if _tier_confs else 0.0)
                    if signal_conf < min_conf:
                        self.logger.info(
                            f"Signal confidence too low: {signal_conf:.3f} < {min_conf} "
                            f"(scalp={scalp_mode})"
                        )
                        state["decision"] = "rejected"
                        state.setdefault("metadata", {})["block_reason"] = f"confidence:{signal_conf:.3f}<{min_conf}"
                        return state

            portfolio = state.get("portfolio_state")
            limits = state.get("risk_limits") or RiskLimits()
            if portfolio:
                if portfolio.daily_drawdown < -limits.max_daily_drawdown_pct / 100:
                    self.logger.info("Daily drawdown limit exceeded")
                    state["decision"] = "rejected"
                    state.setdefault("metadata", {})["block_reason"] = f"daily_drawdown:{portfolio.daily_drawdown*100:.2f}%"
                    return state
                if len(portfolio.open_positions) >= limits.max_concurrent_trades:
                    self.logger.info("Max concurrent trades reached")
                    state["decision"] = "rejected"
                    state.setdefault("metadata", {})["block_reason"] = f"max_concurrent:{len(portfolio.open_positions)}/{limits.max_concurrent_trades}"
                    return state

                # ── Currency correlation guard ─────────────────────────────────
                # Prevent stacking too many positions in the same macro direction.
                # USD pairs: track long/short USD exposure.
                # Cross pairs: track JPY-short (risk-on) and EUR-long exposure.
                max_corr = int(self.config.get("risk", {}).get("max_correlated_positions", 2))
                _USD_QUOTE = frozenset([
                    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD",
                    "XAUUSD", "XAGUSD", "CHFUSD", "CADUSD",
                ])
                _USD_BASE = frozenset([
                    "USDJPY", "USDCHF", "USDCAD", "USDSGD",
                    "USDMXN", "USDNOK", "USDSEK",
                ])
                # JPY crosses: BUY = short JPY (risk-on)
                _JPY_CROSS = frozenset(["EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"])
                # EUR crosses: BUY = long EUR
                _EUR_CROSS = frozenset(["EURGBP", "EURCHF", "EURJPY", "EURAUD", "EURCAD", "EURNZD"])

                def _usd_dir(sym: str, pos_type: str) -> str:
                    """Return correlation bucket string, or '' for untracked pairs."""
                    s = sym.upper()
                    if s in _USD_QUOTE:
                        return "long_usd" if pos_type == "SELL" else "short_usd"
                    if s in _USD_BASE:
                        return "long_usd" if pos_type == "BUY"  else "short_usd"
                    if s in _JPY_CROSS:
                        return "short_jpy" if pos_type == "BUY" else "long_jpy"
                    if s in _EUR_CROSS:
                        return "long_eur" if pos_type == "BUY" else "short_eur"
                    return ""

                signal = state["symbol"].upper()
                signal_type = "SELL" if recipe.direction == "short" else "BUY"
                new_usd_dir = _usd_dir(signal, signal_type)

                if new_usd_dir and max_corr > 0:
                    # Count existing open positions in the same USD direction
                    corr_count = 0
                    for open_sym, positions in portfolio.open_positions_map.items():
                        for p in positions:
                            if _usd_dir(open_sym, p["type"]) == new_usd_dir:
                                corr_count += 1
                    if corr_count >= max_corr:
                        self.logger.info(
                            f"Correlation guard: {corr_count}/{max_corr} {new_usd_dir} "
                            f"positions already open — refusing {signal} {signal_type}"
                        )
                        state["decision"] = "rejected"
                        state.setdefault("metadata", {})["block_reason"] = f"correlation:{new_usd_dir}:{corr_count}/{max_corr}"
                        return state

                if signal in portfolio.open_positions:
                    # ── Scale-in evaluation ───────────────────────────────────
                    sc_cfg = self.config.get("scale_in", {})
                    scale_in_enabled      = sc_cfg.get("enabled", False)
                    max_per_symbol        = int(sc_cfg.get("max_positions_per_symbol", 2))
                    require_profit        = sc_cfg.get("require_profit", True)

                    existing = portfolio.open_positions_map.get(signal, [])
                    entry_type = state.get("metadata", {}).get("entry_type", "")

                    # Conditions for scale-in
                    _require_full = sc_cfg.get("require_full_alignment", True)
                    _ok_entry = (
                        entry_type == "full-alignment"
                        or (not _require_full and entry_type == "pullback-entry")
                    )
                    can_scale = (
                        scale_in_enabled
                        and _ok_entry
                        and len(existing) < max_per_symbol           # cap per symbol
                    )

                    if can_scale:
                        # Must match existing direction (no averaging against the trade)
                        signal_dir = "BUY" if recipe.direction == "long" else "SELL"
                        all_same_dir = all(p["type"] == signal_dir for p in existing)
                        if not all_same_dir:
                            self.logger.info(
                                f"Scale-in blocked for {signal}: signal {signal_dir} "
                                f"conflicts with existing position direction"
                            )
                            state["decision"] = "rejected"
                            state.setdefault("metadata", {})["block_reason"] = f"scale_in:dir_conflict:{signal_dir}"
                            return state

                        # Existing position must be in profit (never scale into a loser)
                        if require_profit:
                            total_profit = sum(p["profit"] for p in existing)
                            if total_profit <= 0:
                                self.logger.info(
                                    f"Scale-in blocked for {signal}: existing "
                                    f"position not in profit (P&L={total_profit:.2f})"
                                )
                                state["decision"] = "rejected"
                                state.setdefault("metadata", {})["block_reason"] = f"scale_in:not_in_profit:{total_profit:.2f}"
                                return state

                        self.logger.info(
                            f"Scale-in approved for {signal} "
                            f"({len(existing)}/{max_per_symbol} positions, "
                            f"signal={signal_dir}, entry={entry_type})"
                        )
                        state.setdefault("metadata", {})["scale_in"] = True
                    else:
                        if not scale_in_enabled:
                            reason = "scale-in disabled"
                        elif not _ok_entry:
                            reason = f"entry_type={entry_type} (needs full-alignment)"
                        else:
                            reason = f"max {max_per_symbol} positions/symbol reached"
                        self.logger.info(f"Position already open for {signal} — {reason}")
                        state["decision"] = "rejected"
                        state.setdefault("metadata", {})["block_reason"] = f"scale_in:{reason}"
                        return state

            # ── Agent consensus gate ─────────────────────────────────────────
            # Require a minimum number of agents to vote in the trade direction.
            # BreadthAgent is a market-context filter (not directional) — excluded.
            min_consensus = int(
                self.config.get("alignment", {}).get("min_agent_consensus", 5)
            )
            if min_consensus > 0:
                _agent_outs  = state.get("agent_outputs", {})
                _dir_sign    = 1.0 if recipe.direction == "long" else -1.0
                _vote_thr    = float(
                    self.config.get("alignment", {}).get("consensus_vote_threshold", 0.05)
                )
                _NO_VOTE     = {"BreadthAgent"}
                _voters      = {n: o for n, o in _agent_outs.items()
                                if n not in _NO_VOTE}
                # Cap to the number of agents that actually ran this cycle
                _effective_min = min(min_consensus, max(1, len(_voters)))
                _votes_for   = sum(
                    1 for o in _voters.values()
                    if o.dir_score * _dir_sign > _vote_thr
                )
                if _votes_for < _effective_min:
                    self.logger.info(
                        f"Consensus gate: {_votes_for}/{len(_voters)} agents agree "
                        f"({recipe.direction}) — need {_effective_min} (cfg={min_consensus})"
                    )
                    state["decision"] = "rejected"
                    state.setdefault("metadata", {})["block_reason"] = f"consensus:{_votes_for}/{len(_voters)}<{_effective_min}"
                    return state
                self.logger.debug(
                    f"Consensus: {_votes_for}/{len(_voters)} agents agree ({recipe.direction})"
                )

            # ── Key level proximity guard ────────────────────────────────────
            # Reject trades aimed directly into a nearby weekly pivot level.
            # Entering a LONG into resistance or a SHORT into support is a
            # low-probability setup — the level typically pushes price back.
            _pivot_buf = float(
                self.config.get("risk", {}).get("pivot_buffer_atr", 0.50)
            )
            _feat = state.get("features")
            if _feat and _pivot_buf > 0:
                if recipe.direction == "long":
                    _dist = getattr(_feat, "nearest_resist_atr", 10.0)
                    if _dist < _pivot_buf:
                        self.logger.info(
                            f"Pivot guard: resistance {_dist:.2f}×ATR away "
                            f"— rejecting LONG (buffer={_pivot_buf})"
                        )
                        state["decision"] = "rejected"
                        state.setdefault("metadata", {})["block_reason"] = f"pivot_resist:{_dist:.2f}atr<{_pivot_buf}"
                        return state
                elif recipe.direction == "short":
                    _dist = getattr(_feat, "nearest_support_atr", 10.0)
                    if _dist < _pivot_buf:
                        self.logger.info(
                            f"Pivot guard: support {_dist:.2f}×ATR away "
                            f"— rejecting SHORT (buffer={_pivot_buf})"
                        )
                        state["decision"] = "rejected"
                        state.setdefault("metadata", {})["block_reason"] = f"pivot_support:{_dist:.2f}atr<{_pivot_buf}"
                        return state

            state["decision"] = "approved"
            self.logger.info("Risk check passed")

            # ── Debug tracer: record risk-passed ──
            _get_tracer().record_risk(passed=True)

        except Exception as e:
            self.logger.error(f"Error in risk check: {e}")
            state["errors"].append(f"Risk check: {e}")
            state["decision"] = "rejected"

        return state

    def _risk_approved(self, state: TradingState) -> str:
        decision = state.get("decision", "rejected")
        if decision != "approved":
            _get_tracer().commit_bar()  # bar trace done (risk rejected)
        return decision

    async def _create_plan(self, state: TradingState) -> TradingState:
        """Create detailed trade execution plan."""
        try:
            self.logger.info("Creating trade plan")
            recipe = state.get("trade_recipe")
            features = state.get("features")
            if not recipe or not features:
                return state

            limits = state.get("risk_limits") or RiskLimits()
            portfolio = state.get("portfolio_state")
            equity = portfolio.equity if portfolio else 10000.0
            is_scale_in = state.get("metadata", {}).get("scale_in", False)

            # Scale-in uses half the normal risk and a tighter 1×ATR stop
            # to add to a winner without over-exposing.
            if is_scale_in:
                sc_cfg = self.config.get("scale_in", {})
                risk_fraction = float(sc_cfg.get("risk_fraction", 0.5))
                sl_atr_mult   = float(sc_cfg.get("sl_atr_multiplier", 1.0))
                self.logger.info(
                    f"Scale-in plan: risk×{risk_fraction}, SL={sl_atr_mult}×ATR"
                )
            else:
                risk_fraction = 1.0
                sl_atr_mult   = 2.0

            risk_amount   = equity * limits.base_risk_pct / 100 * risk_fraction

            # ── Per-symbol risk multiplier ────────────────────────────────────
            # Applied BEFORE VIX/Kelly so that volatile/hard-to-predict assets
            # (e.g. XAUUSD) are capped at a fraction of the profile's base risk
            # regardless of any other scaling.  Values < 1.0 reduce risk; the
            # key "XAUUSD" catches any case where gold is re-added to a profile.
            _sym_risk_overrides = self.config.get("symbol_risk_multipliers", {})
            _sym_mult = float(_sym_risk_overrides.get(state["symbol"].upper(), 1.0))
            if _sym_mult != 1.0:
                risk_amount *= _sym_mult
                self.logger.info(
                    f"Symbol risk override [{state['symbol']}]: "
                    f"×{_sym_mult:.2f} → risk=${risk_amount:.2f}"
                )

            # ── VIX-based risk scaling (only downward) ────────────────────────
            # When macro fear is elevated (VIX high or rising fast), slippage
            # and gap risk spike — reduce position size to compensate.
            # We NEVER increase risk above the configured base; only reduce it.
            # vix_signal is pre-computed in features.vix_dir FOR THIS SYMBOL
            # (already sign-mapped). We need the raw VIX level, not the mapped
            # directional score, so we read it from the long-tf features directly.
            long_features  = (state.get("features_by_tf") or {}).get("long") or features
            raw_vix_signal = getattr(long_features, "vix_dir", 0.0)   # [-1, 1]
            vix_cfg        = self.config.get("vix_risk_scaling", {})
            vix_enabled    = vix_cfg.get("enabled", True)
            vix_threshold  = float(vix_cfg.get("threshold", 0.40))   # start reducing above here
            vix_floor      = float(vix_cfg.get("floor", 0.50))       # min fraction of base risk

            # Use the absolute magnitude: high VIX is bad for both longs and shorts
            # (we take the magnitude of the  IntermarketAgent's VIX leg, which is
            # the raw vix_signal from the macro cache weighted by the symbol's VIX weight).
            # For a simpler proxy we look at the breadth_score (also reflects fear): 
            # but vix_dir on the features is sign-mapped per pair so we need a raw VIX proxy.
            # Best available: IntermarketAgent evidence carries vix_dir (raw, not sign-mapped).
            raw_vix_fear = 0.0
            agent_outputs = state.get("agent_outputs", {})
            if "IntermarketAgent" in agent_outputs:
                # vix_dir in evidence is already the symbol-mapped score.
                # Flip the absolute value back to fear intensity (high |vix| = high fear)
                raw_vix_fear = abs(agent_outputs["IntermarketAgent"].evidence.get("vix_dir", 0.0))

            if vix_enabled and raw_vix_fear > vix_threshold:
                # Linear reduction: at threshold → no reduction; at 1.0 → floor
                excess    = (raw_vix_fear - vix_threshold) / max(1.0 - vix_threshold, 1e-9)
                vix_scale = 1.0 - excess * (1.0 - vix_floor)
                vix_scale = float(np.clip(vix_scale, vix_floor, 1.0))
                risk_amount *= vix_scale
                self.logger.info(
                    f"VIX risk scaling: vix_fear={raw_vix_fear:.2f} → "
                    f"risk ×{vix_scale:.2f} ({risk_amount:.2f})"
                )

            # ── Kelly criterion position sizing ──────────────────────────────
            # Scale risk up/down based on per-symbol closed-trade track record.
            # Only active after min_trades history; defaults to 1.0× until then.
            if self._kelly_tracker is not None:
                kelly_mult = self._kelly_tracker.get_multiplier(state["symbol"])
                if kelly_mult != 1.0:
                    risk_amount *= kelly_mult
                    self.logger.info(
                        f"Kelly sizing [{state['symbol']}]: ×{kelly_mult:.2f} "
                        f"→ risk={risk_amount:.2f}"
                    )

            # For SL/TP sizing: use 1H ATR as the PRIMARY anchor.
            # Old logic used max(D1, H1, mid) ATR which always picked D1 ATR
            # (~106 pips for EURUSD). With time-stops of 8-12h, price can only
            # move ~30-80 pips, so the trade never reaches SL or TP and always
            # exits at a tiny R-multiple via TIME-STOP.
            # New logic: prefer H1 ATR (proportional to the intraday holding
            # period), use D1 ATR only as a cap to prevent oversized stops,
            # and use mid ATR as a floor to avoid noise stops.
            _plan_features_by_tf = state.get("features_by_tf", {})
            _mid_feats = _plan_features_by_tf.get("mid")
            _h1_feats  = _plan_features_by_tf.get("1H")   # always loaded by preload
            _long_feats = _plan_features_by_tf.get("long")

            # Primary: 1H ATR — matches the typical intraday holding period
            _h1_atr = (_h1_feats.atr_14 if _h1_feats and _h1_feats.atr_14 > 0 else 0.0)
            # Floor: mid-TF ATR — never go below this (avoids noise stops on 1m)
            _mid_atr = (_mid_feats.atr_14 if _mid_feats and _mid_feats.atr_14 > 0 else 0.0)
            # Cap: D1 ATR — never wider than D1 scale
            _d1_atr = (_long_feats.atr_14 if _long_feats and _long_feats.atr_14 > 0 else 0.0)

            if _h1_atr > 0:
                atr_for_sizing = _h1_atr
            elif _mid_atr > 0:
                atr_for_sizing = _mid_atr
            else:
                atr_for_sizing = features.atr_14
            # Floor: at least mid-TF ATR (so 1m noise doesn't create 2-pip stops)
            if _mid_atr > 0:
                atr_for_sizing = max(atr_for_sizing, _mid_atr)
            # Cap: never wider than D1 ATR (prevents extreme stops on volatile days)
            if _d1_atr > 0:
                atr_for_sizing = min(atr_for_sizing, _d1_atr)

            stop_distance_raw = atr_for_sizing * sl_atr_mult

            live_price = self.executor.get_current_price(state["symbol"])
            if live_price:
                entry_price = live_price[1] if recipe.direction == "long" else live_price[0]
            else:
                entry_price = features.ema20   # fallback

            # ── Structure-aware SL / TP ────────────────────────────────────────
            # 1. Load pivot levels (PP, R1/R2, S1/S2) in absolute price terms.
            # 2. SL: place beyond the nearest structural level on the adverse side
            #        + 0.25×ATR buffer to absorb normal wick noise.
            # 3. TP: target the next structural level on the profit side.
            #        Fall back to R:R multiple when no suitable level exists.
            # 4. R:R is derived from the actual SL/TP distances rather than
            #        being a fixed ratio — this makes it market-driven.
            # 5. Minimum SL distance: 1.5×ATR (never expose less than this).
            atr = atr_for_sizing
            sl_buffer = 0.25 * atr              # wick noise buffer beyond level
            min_stop  = 1.5 * atr               # absolute floor for SL distance

            # get_structural_levels() merges weekly formula pivots with D1 swing
            # highs/lows so that SL/TP targets reflect actual price memory.
            pivots = self.data_loader.get_structural_levels(
                state["symbol"], entry_price, atr
            ) if self.data_loader is not None else None

            # Use per-symbol swing highs/lows as an additional SL anchor.
            # swing_lows/swing_highs come from the features (computed in loader).
            swing_lows  = features.swing_lows  if features.swing_lows  else []
            swing_highs = features.swing_highs if features.swing_highs else []

            if recipe.direction == "long":
                # SL: below the nearest support pivot (or swing low below entry).
                # For LONG trades the SL should be as far below entry as possible
                # (wider = more room) while still being logically sound.
                # pivot structural_sl is the primary anchor.
                # swing_lows are used as an ALTERNATIVE only if they give MORE room
                # (i.e. swing_sl < structural_sl — the swing node is further away).
                # NEVER use swing_sl to make the stop TIGHTER; that places SL at the
                # nearest 15m micro-swing which gets hit by normal noise.
                structural_sl = None
                if pivots:
                    structural_sl = pivots["nearest_support"] - sl_buffer
                if swing_lows:
                    lows_below = [l for l in swing_lows if l < entry_price]
                    if lows_below:
                        swing_sl = max(lows_below) - sl_buffer
                        # Only widen the SL (use swing if it gives more room)
                        if structural_sl is None or swing_sl < structural_sl:
                            structural_sl = swing_sl

                # Floor: never less than min_stop below entry
                if structural_sl is None or (entry_price - structural_sl) < min_stop:
                    structural_sl = entry_price - min_stop

                stop_loss = structural_sl
                stop_distance = entry_price - stop_loss

                # TP: next resistance above entry.
                # In a strong trend (ADX > 40) skip R1 and aim one level further — D1
                # structural levels from get_structural_levels() make R2 meaningful.
                if pivots:
                    resist_candidates = sorted(
                        l for l in (pivots["r1"], pivots["r2"]) if l > entry_price + min_stop
                    )
                    adx_strong = features.adx_14 > 40.0
                    if adx_strong and len(resist_candidates) >= 2:
                        take_profit = resist_candidates[1]  # aim for 2nd level in strong trend
                    elif resist_candidates:
                        take_profit = resist_candidates[0]
                    else:
                        take_profit = entry_price + stop_distance * recipe.risk_reward_ratio
                else:
                    take_profit = entry_price + stop_distance * recipe.risk_reward_ratio

            else:  # short
                # SL: above nearest resistance pivot (or swing high above entry).
                # For SHORT trades the SL should be as far above entry as possible
                # (wider = more room) while still being logically sound.
                # pivot structural_sl is the primary anchor.
                # swing_highs are used as an ALTERNATIVE only if they give MORE room
                # (i.e. swing_sl > structural_sl — the swing node is further away).
                # NEVER use swing_sl to make the stop TIGHTER; that places SL at the
                # nearest 15m micro-swing which gets hit by normal noise.
                structural_sl = None
                if pivots:
                    structural_sl = pivots["nearest_resist"] + sl_buffer
                if swing_highs:
                    highs_above = [h for h in swing_highs if h > entry_price]
                    if highs_above:
                        swing_sl = min(highs_above) + sl_buffer
                        # Only widen the SL (use swing if it gives more room)
                        if structural_sl is None or swing_sl > structural_sl:
                            structural_sl = swing_sl

                if structural_sl is None or (structural_sl - entry_price) < min_stop:
                    structural_sl = entry_price + min_stop

                stop_loss = structural_sl
                stop_distance = stop_loss - entry_price

                # TP: next support below entry (S1, then S2).
                # In a strong downtrend (ADX > 40) use the further level.
                if pivots:
                    support_candidates = sorted(
                        (l for l in (pivots["s1"], pivots["s2"]) if l < entry_price - min_stop),
                        reverse=True,  # highest first = nearest below entry
                    )
                    adx_strong = features.adx_14 > 40.0
                    if adx_strong and len(support_candidates) >= 2:
                        take_profit = support_candidates[1]  # aim for 2nd (lower) level
                    elif support_candidates:
                        take_profit = support_candidates[0]
                    else:
                        take_profit = entry_price - stop_distance * recipe.risk_reward_ratio
                else:
                    take_profit = entry_price - stop_distance * recipe.risk_reward_ratio

            # Log structural context for transparency
            if pivots:
                self.logger.info(
                    f"Structure SL/TP [{state['symbol']}] "
                    f"entry={entry_price:.5f} "
                    f"sl={stop_loss:.5f} ({stop_distance/atr:.2f}×ATR) "
                    f"tp={take_profit:.5f} (R:R={abs(take_profit-entry_price)/max(stop_distance,1e-9):.2f})"
                )

            # ── Minimum R:R enforcement ────────────────────────────────────────
            # Structural pivot TP can land closer than 1R when pivots are tight.
            # Always ensure TP is at least recipe.risk_reward_ratio × SL distance
            # so the recipe EV claim is honoured in practice.
            _min_tp_dist = stop_distance * recipe.risk_reward_ratio
            if recipe.direction == "long":
                take_profit = max(take_profit, entry_price + _min_tp_dist)
            else:
                take_profit = min(take_profit, entry_price - _min_tp_dist)

            # ── Smart SL/TP override (scalp mode) ─────────────────────────────
            # For scalp/fast-exit profiles flat ATR multiples are replaced by a
            # multi-factor calculation that adapts to:
            #
            #   SL side
            #   1. 1m swing structure — SL placed just beyond the nearest micro
            #      swing node on the adverse side (true "thesis invalidation" level)
            #   2. Vol expansion guard — if ATR_5 > ATR_14 (vol spiking), widen SL
            #      proportionally so the burst doesn't shake us out
            #   3. Spread floor — SL ≥ 2.5× current spread to avoid instant stop-out
            #   4. Hard floor — never less than sl_atr_mult × ATR_1m
            #
            #   TP side
            #   5. ADX-adaptive R:R — strong trend (ADX > 35) → extend TP 1.4×;
            #      ranging (ADX < 20) → cut to 0.8× to grab the mean-reversion fast
            #   6. Squeeze breakout bonus — when BB inside Keltner, breakout travels
            #      further → TP × 1.25
            #   7. Micro-swing target — if the next 1m swing node on the profit side
            #      is within 5×ATR, use it (real price memory beats a raw multiple)
            #   8. Cap — TP never beyond 5×ATR (keeps the scalp thesis valid)
            tight_cfg = self.config.get("tight_sl_tp", {})
            if tight_cfg.get("enabled", False):
                short_feats  = (state.get("features_by_tf") or {}).get("short") or features
                tight_atr    = short_feats.atr_14 if short_feats else atr
                atr_5_short  = (short_feats.atr_5 if short_feats else tight_atr) or tight_atr
                sl_floor_m   = float(tight_cfg.get("sl_atr_mult", 2.0))
                base_rr      = float(tight_cfg.get("tp_atr_mult", 3.0)) / max(sl_floor_m, 1e-9)

                # ── 1. Swing structure anchor ────────────────────────────────
                swing_sl_dist = None
                if short_feats:
                    _buf = 0.15 * tight_atr
                    _max = 4.0  * tight_atr
                    _sh  = sorted(short_feats.swing_highs or [])
                    _sl  = sorted(short_feats.swing_lows  or [])
                    if recipe.direction == "long" and _sl:
                        cands = [l for l in _sl if l < entry_price and entry_price - l <= _max]
                        if cands:
                            swing_sl_dist = entry_price - max(cands) + _buf
                    elif recipe.direction == "short" and _sh:
                        cands = [h for h in _sh if h > entry_price and h - entry_price <= _max]
                        if cands:
                            swing_sl_dist = min(cands) - entry_price + _buf

                # ── 2. Volatility expansion adjustment ──────────────────────
                vol_ratio     = float(np.clip(atr_5_short / max(tight_atr, 1e-10), 0.5, 2.0))
                vol_expansion = float(np.clip(vol_ratio, 1.0, 1.5))   # only widen, never compress

                # ── 3 & 4. Resolve raw SL distance ──────────────────────────
                floor_sl = sl_floor_m * tight_atr * vol_expansion
                raw_sl_dist = max(swing_sl_dist, floor_sl) if swing_sl_dist is not None else floor_sl

                # ── 3. Spread floor (live mode only) ────────────────────────
                if not self.executor.simulation_mode:
                    _si = self.executor.get_symbol_info(state["symbol"])
                    if _si:
                        _spread_price = getattr(_si, "spread", 0) * getattr(_si, "trade_tick_size", 0.0001)
                        raw_sl_dist   = max(raw_sl_dist, _spread_price * 2.5)

                # ── 5. ADX-adaptive R:R ──────────────────────────────────────
                adx_val = short_feats.adx_14 if short_feats else 20.0
                if adx_val >= 35:
                    adaptive_rr = base_rr * 1.4    # trending — let it run
                elif adx_val <= 20:
                    adaptive_rr = base_rr * 0.8    # ranging  — take it fast
                else:
                    t = (adx_val - 20.0) / 15.0
                    adaptive_rr = base_rr * (0.8 + t * 0.6)

                # ── 6. Squeeze breakout bonus ────────────────────────────────
                in_squeeze = (short_feats and short_feats.bb_width > 0
                              and short_feats.keltner_width > 0
                              and short_feats.bb_width < short_feats.keltner_width)
                if in_squeeze:
                    adaptive_rr = min(adaptive_rr * 1.25, base_rr * 2.0)

                # ── 7. Micro-swing target on profit side ─────────────────────
                swing_tp_dist = None
                if short_feats:
                    _max_tp = 5.0 * tight_atr
                    _sh2 = sorted(short_feats.swing_highs or [])
                    _sl2 = sorted(short_feats.swing_lows  or [])
                    if recipe.direction == "long" and _sh2:
                        cands = [h for h in _sh2 if h > entry_price and h - entry_price <= _max_tp]
                        if cands:
                            swing_tp_dist = min(cands) - entry_price
                    elif recipe.direction == "short" and _sl2:
                        cands = [l for l in _sl2 if l < entry_price and entry_price - l <= _max_tp]
                        if cands:
                            swing_tp_dist = entry_price - max(cands)

                rr_tp_dist = raw_sl_dist * adaptive_rr
                if swing_tp_dist is not None and swing_tp_dist > raw_sl_dist:
                    tp_dist = min(max(swing_tp_dist, rr_tp_dist), 5.0 * tight_atr)
                else:
                    tp_dist = min(rr_tp_dist, 5.0 * tight_atr)

                # ── Apply ────────────────────────────────────────────────────
                if recipe.direction == "long":
                    stop_loss   = entry_price - raw_sl_dist
                    take_profit = entry_price + tp_dist
                else:
                    stop_loss   = entry_price + raw_sl_dist
                    take_profit = entry_price - tp_dist
                stop_distance = raw_sl_dist

                actual_rr = tp_dist / max(raw_sl_dist, 1e-9)
                self.logger.info(
                    f"Smart SL/TP [{state['symbol']}] "
                    f"entry={entry_price:.5f} sl={stop_loss:.5f} tp={take_profit:.5f} "
                    f"atr_1m={tight_atr:.5f} sl_dist={raw_sl_dist:.5f} "
                    f"adx={adx_val:.1f} vol_ratio={vol_ratio:.2f} R:R={actual_rr:.2f}"
                    + (" [swing-sl]" if swing_sl_dist else "")
                    + (" [swing-tp]" if swing_tp_dist else "")
                    + (" [squeeze]"  if in_squeeze    else "")
                )

            # ── Lot calculation (forex-correct) ───────────────────────────────
            # The P&L per lot for a price move of Δ is:
            #   P&L = (Δ / tick_size) × tick_value   (tick_value is already in
            #                                          account currency, MT5 handles
            #                                          all quote-currency conversions)
            # So: lots = risk_amount / (stop_distance / tick_size × tick_value)
            #
            # The old formula (stop_distance × contract_size) was only correct
            # when quote_currency == account_currency (e.g. EURUSD on a USD account).
            # It is wrong by up to 200× for pairs like USDJPY on a EUR account.
            symbol_info = self.executor.get_symbol_info(state["symbol"])
            tick_size  = getattr(symbol_info, "trade_tick_size",  0.0001) if symbol_info else 0.0001
            tick_value = getattr(symbol_info, "trade_tick_value", 10.0)   if symbol_info else 10.0
            vol_min    = getattr(symbol_info, "volume_min",  0.01) if symbol_info else 0.01
            vol_step   = getattr(symbol_info, "volume_step", 0.01) if symbol_info else 0.01
            digits     = getattr(symbol_info, "digits", 5)         if symbol_info else 5

            if tick_size > 0 and tick_value > 0:
                risk_per_lot = (stop_distance / tick_size) * tick_value
            else:
                # Fallback to old formula (simulation / unknown instrument)
                contract_size = getattr(symbol_info, "trade_contract_size", 100_000) if symbol_info else 100_000
                risk_per_lot = stop_distance * contract_size

            raw_lots = risk_amount / max(risk_per_lot, 1e-9)
            if vol_step > 0:
                raw_lots = round(round(raw_lots / vol_step) * vol_step, 8)
            quantity = max(vol_min, raw_lots)

            # Recalculate risk_amount to reflect actual position size after
            # lot-step rounding.  Without this, R-multiples are inconsistent
            # because the planned risk_amount can differ significantly from
            # the true dollar risk at the stop-loss level.
            risk_amount = risk_per_lot * quantity

            # ── Round SL/TP to broker digits & enforce minimum stop distance ──
            # MT5 rejects orders if SL/TP don't match the symbol's decimal places
            # or are closer than stops_level points from the current price.
            stop_loss   = round(stop_loss,   digits)
            take_profit = round(take_profit, digits)
            min_stop_pts = getattr(symbol_info, "trade_stops_level", 0) if symbol_info else 0
            if tick_size > 0 and min_stop_pts > 0:
                min_dist = min_stop_pts * tick_size
                if recipe.direction == "long":
                    stop_loss   = min(stop_loss,   round(entry_price - min_dist, digits))
                    take_profit = max(take_profit, round(entry_price + min_dist, digits))
                else:
                    stop_loss   = max(stop_loss,   round(entry_price + min_dist, digits))
                    take_profit = min(take_profit, round(entry_price - min_dist, digits))

            fusion = state["timeframe_fusion"]
            state["trade_plan"] = TradePlan(
                symbol=state["symbol"],
                recipe=recipe,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_amount=risk_amount,
                # Use the tier that drives conviction in this mode:
                # scalp → conf_short (1m ScalpingAgent at 3× weight)
                # swing → conf_long (D1 thesis is the primary driver)
                confidence=(
                    fusion.conf_short
                    if bool(self.config.get("tight_sl_tp", {}).get("enabled", False))
                    else fusion.conf_long
                ),
                timeframes_aligned=["long", "mid", "short"],
            )
            self.logger.info(f"Created trade plan: entry={entry_price:.4f}, sl={stop_loss:.4f}, tp={take_profit:.4f}")

            # ── Debug tracer: record plan details ──
            _tr = _get_tracer()
            _long_atr = (_long_feats.atr_14 if _long_feats else 0.0)
            _mid_atr = (_mid_feats.atr_14 if _mid_feats else 0.0)
            _short_atr = ((state.get("features_by_tf") or {}).get("short") or features).atr_14
            _pip_sz = 0.01 if "JPY" in state["symbol"] else 0.0001
            _tr.record_plan(
                entry=entry_price, sl=stop_loss, tp=take_profit,
                atr_sizing=atr_for_sizing, atr_long=_long_atr,
                atr_mid=_mid_atr, atr_short=_short_atr,
                sl_mult=sl_atr_mult, pivots=pivots,
                swing_highs=swing_highs, swing_lows=swing_lows,
                quantity=quantity, risk_amount=risk_amount,
                pip_size=_pip_sz,
            )

        except Exception as e:
            self.logger.error(f"Error creating trade plan: {e}")
            state["errors"].append(f"Trade plan creation: {e}")

        return state

    async def _execute_trade(self, state: TradingState) -> TradingState:
        """Execute the trade via MT5Executor."""
        try:
            trade_plan = state.get("trade_plan")
            if not trade_plan:
                self.logger.warning("No trade plan to execute")
                return state

            # ── Signal freshness check ─────────────────────────────────────────────────────
            # The bar data used to build this plan was fetched at signal_born_at.
            # With 21 symbols processed sequentially, the last symbol in the list
            # could be executing a plan based on bars that are 3-5 minutes old.
            # If the signal age exceeds the configured limit we skip the order
            # entirely: the entry price, SL, and TP may all be wrong.
            import time as _time
            max_signal_age = float(
                self.config.get("execution", {}).get("signal_max_age_seconds", 0)
            )
            _born = state.get("metadata", {}).get("signal_born_at", 0)
            if max_signal_age > 0 and _born > 0:
                signal_age = _time.time() - _born
                if signal_age > max_signal_age:
                    self.logger.warning(
                        f"{state['symbol']}: signal is {signal_age:.0f}s old "
                        f"(limit={max_signal_age:.0f}s) — discarding stale plan"
                    )
                    state["errors"].append(f"stale signal ({signal_age:.0f}s > {max_signal_age:.0f}s)")
                    return state

            self.logger.info(f"Executing trade for {state['symbol']}")
            result = self.executor.place_bracket_order(trade_plan)
            if result:
                state["metadata"]["executed"] = True
                state["metadata"]["execution_time"] = datetime.now().isoformat()
                state["metadata"]["order_ticket"] = result.get("order_ticket")
                self.logger.info(f"Order placed: ticket={result.get('order_ticket')}")
            else:
                # None is the normal return when the executor skips placement for a
                # legitimate reason (market closed, stale tick, spread too wide).
                # Those cases already emit a WARNING in the executor — just note
                # silently here rather than logging as ERROR, which is misleading.
                self.logger.info(
                    f"Order not placed for {state['symbol']} "
                    "(executor returned None — see executor log for reason)"
                )

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            state["errors"].append(f"Trade execution: {e}")

        # ── Debug tracer: record execution ──
        _get_tracer().record_executed(bool(state.get("metadata", {}).get("executed")))
        _get_tracer().commit_bar()

        return state
    
    async def run(self, symbol: str, features: TechnicalFeatures,
                  portfolio_state: PortfolioState = None,
                  risk_limits: RiskLimits = None,
                  features_by_tf: Dict[str, TechnicalFeatures] = None,
                  exit_check_only: bool = False) -> TradingState:
        """Run the complete trading decision process."""

        # Initialize state as a dict (LangGraph requirement)
        state: TradingState = {
            "symbol": symbol,
            "features": features,
            "features_by_tf": features_by_tf or {},
            "agent_outputs": {},
            "timeframe_fusion": None,
            "trade_recipe": None,
            "trade_plan": None,
            "decision": "continue",
            "errors": [],
            "metadata": {},
            # Critical: portfolio_state and risk_limits MUST be in the state dict
            # so that _risk_check / _create_plan can read them via state.get().
            "portfolio_state": portfolio_state,
            "risk_limits": risk_limits,
        }

        if exit_check_only:
            # Only run agents + fuse — do not check alignment / risk / execute
            try:
                state = await self._run_agents(state)
                state = await self._fuse_timeframes(state)
            except Exception as e:
                self.logger.error(f"Error in exit_check_only run for {symbol}: {e}")
                state["errors"].append(str(e))
            return state

        # Run the graph
        try:
            final_state = await self.graph.ainvoke(state)
            return final_state
        except Exception as e:
            self.logger.error(f"Error running trading graph: {e}")
            state["errors"].append(f"Graph execution: {e}")
            return state