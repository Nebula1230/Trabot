"""
Main configuration settings for the TradingAgents system.
"""

import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class TimeframeConfig(BaseModel):
    """Timeframe configuration."""
    long: List[str] = Field(default=["1D", "1W"], description="Long-term timeframes")
    mid: List[str] = Field(default=["4H", "1H"], description="Medium-term timeframes")
    short: List[str] = Field(default=["15m", "5m"], description="Short-term timeframes")


class AlignmentThresholds(BaseModel):
    """Alignment thresholds for different timeframes.

    Values may be set to -1.0 to effectively disable a tier gate (useful when
    a profile has no agents for that timeframe tier, e.g. scalp has no LONG/MID
    agents so long/mid are set to -1.0 so the strict > comparison always passes).
    """
    long: float = Field(default=0.4, ge=-1.0, le=1.0, description="Long timeframe alignment threshold")
    mid: float = Field(default=0.3, ge=-1.0, le=1.0, description="Mid timeframe alignment threshold")
    short: float = Field(default=0.2, ge=-1.0, le=1.0, description="Short timeframe alignment threshold")


class RiskConfig(BaseModel):
    """Risk management configuration."""
    base_risk_pct: float = Field(default=0.25, ge=0.01, le=5.0, description="Base risk per trade as % of equity")
    max_daily_drawdown_pct: float = Field(default=2.0, ge=0.1, le=20.0, description="Maximum daily drawdown %")
    max_weekly_drawdown_pct: float = Field(default=0.0, ge=0.0, le=30.0, description="Maximum weekly drawdown % (0 = disabled)")
    max_concurrent_trades: int = Field(default=3, ge=1, le=20, description="Maximum concurrent trades")
    per_symbol_leverage_cap: float = Field(default=3.0, ge=1.0, le=10.0, description="Per-symbol leverage cap")
    portfolio_leverage_cap: float = Field(default=5.0, ge=1.0, le=20.0, description="Portfolio leverage cap")
    max_correlated_positions: int = Field(default=2, ge=1, le=10, description="Max open positions in the same currency direction (e.g. long USD)")
    # Weekly pivot level proximity guard (ATR units).  0 = disabled.
    pivot_buffer_atr: float = Field(default=0.50, ge=0.0, description="Reject entries closer than N×ATR to a weekly pivot level")
    # Overtrading guards — MUST be in the Pydantic model so model_dump() exposes
    # them to runner.py.  If missing, Pydantic drops them silently and guards are
    # disabled (root cause of 1556 scalp trades / −$3858 in one session).
    max_daily_trades: int = Field(default=0, ge=0, description="Hard daily cap on total orders placed (0 = disabled)")
    max_daily_sl_per_symbol: int = Field(default=0, ge=0, description="Stop trading a symbol after N stop-loss exits in one day (0 = disabled)")
    entry_cooldown_minutes: float = Field(default=0.0, ge=0.0, description="Minimum minutes between successive entries on the same symbol (0 = disabled)")
    streak_cooldown_base_hours: float = Field(default=4.0, ge=0.0, description="Base streak cooldown hours after 2 consecutive losses (doubles each re-trigger)")
    streak_cooldown_max_hours: float = Field(default=24.0, ge=0.0, description="Maximum streak cooldown cap in hours")
    max_lot: float = Field(default=0.0, ge=0.0, description="Hard ceiling on position size in lots (0 = disabled)")
    margin_free_pct_threshold: float = Field(default=0.20, ge=0.05, le=0.50, description="Minimum free margin as fraction of equity to allow new entry (0.20 = 20%)")
    spread_guard_fraction: float = Field(default=0.20, ge=0.05, le=0.50, description="Max spread as fraction of stop distance to allow fill (0.20 = 20%)")
    slippage_pips: float = Field(default=0.3, ge=0.0, le=5.0, description="Random ±slippage in pips on every backtest fill")


class ExecutionConfig(BaseModel):
    """Execution configuration."""
    slippage_bp: int = Field(default=2, ge=0, le=50, description="Slippage in basis points")
    spread_guard_bp: int = Field(default=3, ge=0, le=100, description="Spread guard in basis points")
    news_blackout_minutes: int = Field(default=10, ge=0, le=60, description="News blackout period in minutes")
    max_tick_age_ms: int = Field(default=1000, ge=100, le=10000, description="Maximum tick age in milliseconds")
    use_bracket_orders: bool = Field(default=True, description="Use bracket orders for entry/exit")
    partial_take_profits: bool = Field(default=True, description="Enable partial take profits")
    trailing_stops: bool = Field(default=True, description="Enable trailing stops")


class ProbabilityConfig(BaseModel):
    """Probability and calibration configuration."""
    recipe_min_trades: int = Field(default=50, ge=10, le=1000, description="Minimum trades for recipe calibration")
    min_win_prob: float = Field(default=0.48, ge=0.4, le=0.8, description="Minimum win probability")
    min_expectancy_r: float = Field(default=0.10, ge=0.01, le=1.0, description="Minimum expectancy in R units")


class DebateConfig(BaseModel):
    """Debate configuration."""
    rounds: int = Field(default=1, ge=0, le=3, description="Number of debate rounds")
    enabled: bool = Field(default=True, description="Enable debate between Pro/Contra")


class MT5Config(BaseModel):
    """MetaTrader 5 configuration."""
    host:      str = Field(default="localhost", description="mt5linux host (mt5docker container or localhost)")
    port:      int = Field(default=18812,       description="mt5linux API port (mt5linux default; docker-compose maps 18812→18812)")
    login:     int = Field(default=0,        description="MT5 account login number")
    password:  str = Field(default="",       description="MT5 account password")
    server:    str = Field(default="",       description="MT5 broker server name")
    magic_number: int = Field(default=424242, description="Magic number for orders")
    slippage: int = Field(default=10, ge=1, le=100, description="Default slippage in points")
    retry_attempts: int = Field(default=3, ge=1, le=10, description="Retry attempts for failed orders")
    retry_delay_ms: int = Field(default=1000, ge=100, le=10000, description="Delay between retries in milliseconds")
    max_spread_fraction: float = Field(default=0.20, ge=0.0, le=1.0, description="Max spread as fraction of stop distance (0=disabled). Instrument-agnostic: 0.20 = skip if spread > 20% of planned stop")


class AgentConfig(BaseModel):
    """Agent configuration."""
    enabled_agents: List[str] = Field(default_factory=list, description="List of enabled agent names")
    agent_weights: Dict[str, float] = Field(default_factory=dict, description="Agent weight multipliers")
    custom_prompts: Dict[str, str] = Field(default_factory=dict, description="Custom prompts for agents")


_VALID_LLM_PROVIDERS = frozenset({
    "openai", "anthropic", "google", "ollama", "lmstudio", "openrouter", "xai"
})
_VALID_ANALYSTS = frozenset({"market", "social", "news", "fundamentals"})


class LLMAgentsConfig(BaseModel):
    """Configuration for the optional LLM agent bridge (tradingagents package)."""

    enabled: bool = Field(default=False, description="Enable LLM sentiment agent")
    upstream_path: str = Field(
        default="",
        description=(
            "Absolute path to the cloned TauricResearch/TradingAgents repo root. "
            "Used to resolve the tradingagents package when a local shadow exists."
        ),
    )
    analysts: List[str] = Field(
        default_factory=lambda: ["market", "news"],
        description="Analyst modules: market | social | news | fundamentals",
    )
    throttle_hours: float = Field(
        default=4.0, ge=0.25,
        description="Re-run LLM pipeline at most every N hours per symbol",
    )
    timeout_seconds: int = Field(
        default=120, ge=10, le=600,
        description="Hard wall-clock cap (seconds) per call including retries",
    )
    max_retries: int = Field(
        default=2, ge=0, le=5,
        description="Number of additional retry attempts on transient errors",
    )
    cb_fail_threshold: int = Field(
        default=3, ge=1, le=20,
        description="Consecutive failures before the circuit breaker opens",
    )
    cb_cooldown_minutes: float = Field(
        default=15.0, ge=1.0, le=240.0,
        description="Minutes to wait in OPEN state before allowing a probe (HALF-OPEN)",
    )
    llm_provider: str = Field(
        default="openai",
        description="LLM provider: openai | anthropic | google | ollama | lmstudio | openrouter | xai",
    )
    deep_think_llm: str = Field(default="gpt-4o-mini", description="Model for deep (slow) reasoning")
    quick_think_llm: str = Field(default="gpt-4o-mini", description="Model for fast extraction")
    backend_url: str = Field(
        default="",
        description="LLM API base URL (leave blank to use the provider default)",
    )
    weight: float = Field(default=1.0, ge=0.0, le=5.0, description="Agent weight in timeframe fusion")

    # ── Pydantic v2 validators ────────────────────────────────────────────────

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        v_lower = str(v).lower().strip()
        if v_lower not in _VALID_LLM_PROVIDERS:
            raise ValueError(
                f"llm_provider '{v}' is not supported. "
                f"Choose from: {sorted(_VALID_LLM_PROVIDERS)}"
            )
        return v_lower

    @field_validator("analysts", mode="before")
    @classmethod
    def _validate_analysts(cls, v: list) -> list:
        invalid = [a for a in v if a not in _VALID_ANALYSTS]
        if invalid:
            raise ValueError(
                f"Invalid analyst(s): {invalid}. "
                f"Choose from: {sorted(_VALID_ANALYSTS)}"
            )
        if not v:
            raise ValueError("analysts list must contain at least one entry")
        return v

    @field_validator("upstream_path", mode="before")
    @classmethod
    def _validate_upstream_path(cls, v: str) -> str:
        v = str(v).strip()
        if v and not os.path.isdir(v):
            raise ValueError(
                f"llm_agents.upstream_path '{v}' does not exist or is not a directory. "
                "Clone https://github.com/TauricResearch/TradingAgents first."
            )
        return v


class JournalConfig(BaseModel):
    """Trade journal / logging configuration."""
    log_dir:         str  = Field(default="logs",  description="Directory for log files")
    log_decisions:   bool = Field(default=True,    description="Log every signal cycle")
    log_trades:      bool = Field(default=True,    description="Log every executed trade")
    debug_decisions: bool = Field(default=False,   description="Log rich per-decision debug detail (agent votes, fusion, alignment, sizing, etc.)")


class TradingConfig(BaseSettings):
    """Main configuration for the TradingAgents system."""
    
    # Basic settings
    symbols: List[str] = Field(default=["NASDAQ:NVDA", "NASDAQ:MSFT"], description="Trading symbols")
    timeframes: TimeframeConfig = Field(default_factory=TimeframeConfig, description="Timeframe configuration")
    
    # Alignment and decision making
    alignment_thresholds: AlignmentThresholds = Field(default_factory=AlignmentThresholds, description="Alignment thresholds")
    
    # Risk management
    risk: RiskConfig = Field(default_factory=RiskConfig, description="Risk management configuration")
    
    # Execution
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig, description="Execution configuration")
    
    # Probability and calibration
    probability: ProbabilityConfig = Field(default_factory=ProbabilityConfig, description="Probability configuration")
    
    # Debate system
    debate: DebateConfig = Field(default_factory=DebateConfig, description="Debate configuration")
    
    # MT5 integration
    mt5: MT5Config = Field(default_factory=MT5Config, description="MT5 configuration")
    
    # Agent configuration
    agents: AgentConfig = Field(default_factory=AgentConfig, description="Agent configuration")

    # LLM agent bridge (optional)
    llm_agents: LLMAgentsConfig = Field(default_factory=LLMAgentsConfig, description="LLM sentiment agent config")

    # Cycle interval
    interval_seconds: int = Field(default=3600, ge=10, description="Seconds between trading cycles")

    # Real-time dual-loop config
    realtime: Dict[str, Any] = Field(default_factory=dict, description="Real-time surveillance loop config")

    # Raw alignment block — passed straight through to TradingGraph
    # Keys: long_min_score, mid_min_score, short_min_score, pullback_tolerance,
    #       breadth_min, mean_rev_guard, min_win_prob, min_ev
    alignment: Dict[str, Any] = Field(default_factory=dict, description="Alignment thresholds (raw)")

    # Scale-in (pyramiding) config — passed through to TradingGraph
    scale_in: Dict[str, Any] = Field(default_factory=dict, description="Scale-in / pyramiding config")

    # Exit rules — proactive trade cancellation conditions
    exit_rules: Dict[str, Any] = Field(default_factory=dict, description="Exit / trade cancellation rules")

    # Trailing stop / TP management config
    trailing: Dict[str, Any] = Field(default_factory=dict, description="Trailing stop and TP management config")

    # Tight ATR-based SL/TP override (scalp mode) — bypasses structural pivot targets
    tight_sl_tp: Dict[str, Any] = Field(default_factory=dict, description="Tight ATR-based SL/TP override for scalp")

    # Regime-conditional agent weighting
    regime_weighting: Dict[str, Any] = Field(default_factory=dict, description="Regime-based agent weight scaling")

    # VIX-based risk scaling
    vix_risk_scaling: Dict[str, Any] = Field(default_factory=dict, description="VIX-driven risk reduction config")

    # Per-symbol risk multipliers (symbol → fraction of base risk, e.g. {"XAUUSD": 0.50})
    symbol_risk_multipliers: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-symbol risk fraction overrides. Values < 1.0 reduce position size for that symbol.",
    )

    # Kelly criterion adaptive position sizing
    kelly: Dict[str, Any] = Field(default_factory=dict, description="Kelly criterion sizing config")

    # Confidence-scaled position sizing
    confidence_sizing: Dict[str, Any] = Field(default_factory=dict, description="Confidence-scaled position sizing config")

    # Streak-momentum position sizing
    streak_sizing: Dict[str, Any] = Field(default_factory=dict, description="Streak-momentum position sizing config")

    # Holiday / thin-liquidity sizing damper
    holiday_sizing_damper: Dict[str, Any] = Field(default_factory=dict, description="Reduce sizing during thin-liquidity windows (Dec 22-Jan 2)")

    # Adaptive agent weight learning — uses AgentCalibrationTracker hit-rates
    # to periodically adjust `agent.weight` based on closed-trade accuracy.
    # Keys: enabled, sensitivity, min_trades, max_mult, min_mult, shrink_n,
    #       update_interval_hours, log_updates
    adaptive_weights: Dict[str, Any] = Field(
        default_factory=dict,
        description="Adaptive agent weight learning from closed-trade hit-rates",
    )

    # Per-symbol parameter overrides (produced by SymbolTuner).
    # Keys are symbol names; values are dicts with nested overrides, e.g.:
    #   {"USDJPY": {"confidence_sizing": {"floor": 0.90, "ceil": 1.10},
    #               "streak_sizing": {"loss_cut": 0.60},
    #               "exit_rules": {"ct_mid_flip_threshold": 0.55}}}
    symbol_overrides: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-symbol parameter overrides (from tuning layer)",
    )

    # Active risk profile (safe / balanced / risky)
    profile: str = Field(default="balanced", description="Risk profile preset")

    # Journal / logging
    journal: JournalConfig = Field(default_factory=JournalConfig, description="Journal configuration")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Performance monitoring
    enable_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    monitoring_interval_seconds: int = Field(default=60, ge=10, le=3600, description="Monitoring interval")
    
    # Signal ranking: when enabled in portfolio mode, all symbols' signals are
    # collected first, then ranked by expected value (EV) before filling slots.
    # This ensures the best setups get priority regardless of symbol list order.
    rank_signals: bool = Field(default=False, description="Rank competing signals by EV in portfolio mode")

    # Backtesting
    enable_backtesting: bool = Field(default=True, description="Enable backtesting capabilities")
    backtest_start_date: Optional[str] = Field(default=None, description="Backtest start date (YYYY-MM-DD)")
    backtest_end_date: Optional[str] = Field(default=None, description="Backtest end date (YYYY-MM-DD)")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"
    
    def get_agent_weight(self, agent_name: str) -> float:
        """Get weight for a specific agent."""
        return self.agents.agent_weights.get(agent_name, 1.0)
    
    def is_agent_enabled(self, agent_name: str) -> bool:
        """Check if an agent is enabled."""
        if not self.agents.enabled_agents:
            return True  # All agents enabled by default
        return agent_name in self.agents.enabled_agents
    
    def get_custom_prompt(self, agent_name: str) -> Optional[str]:
        """Get custom prompt for a specific agent."""
        return self.agents.custom_prompts.get(agent_name)
    
    def validate_config(self) -> bool:
        """Validate configuration settings."""
        try:
            # Validate risk percentages
            if self.risk.base_risk_pct * self.risk.max_concurrent_trades > 10.0:
                raise ValueError("Total risk exceeds 10% of equity")
            
            # Validate leverage caps
            if self.risk.per_symbol_leverage_cap > self.risk.portfolio_leverage_cap:
                raise ValueError("Per-symbol leverage cap cannot exceed portfolio leverage cap")
            
            # Validate probability thresholds (risky/scalp profiles go as low as 0.42)
            if self.probability.min_win_prob < 0.40:
                raise ValueError("Minimum win probability should be >= 0.40")
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TradingConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict) 