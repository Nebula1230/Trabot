"""
Core types and data structures for the TradingAgents system.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np


class Timeframe(str, Enum):
    """Trading timeframes."""
    LONG = "long"      # 1D, 1W
    MID = "mid"        # 4H, 1H  
    SHORT = "short"    # 15m, 5m


class Direction(str, Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class AgentOutput(BaseModel):
    """Standard output format for all agents."""
    timeframe: Timeframe
    dir_score: float = Field(..., ge=-1.0, le=1.0)  # -1 strong short, +1 strong long
    conf: float = Field(..., ge=0.0, le=1.0)        # confidence 0-1
    rationale: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class TimeframeFusion(BaseModel):
    """Combined signals across timeframes."""
    dir_long: float = Field(..., ge=-1.0, le=1.0)
    dir_mid: float = Field(..., ge=-1.0, le=1.0)
    dir_short: float = Field(..., ge=-1.0, le=1.0)
    conf_long: float = Field(..., ge=0.0, le=1.0)
    conf_mid: float = Field(..., ge=0.0, le=1.0)
    conf_short: float = Field(..., ge=0.0, le=1.0)
    regime_trendiness: float = Field(..., ge=0.0, le=1.0)
    breadth_score: float = Field(..., ge=-1.0, le=1.0)
    # ── Fusion quality metrics (added: better signal quality representation) ──
    # Geometric mean of tier magnitudes when all 3 tiers agree in direction,
    # else 0.0.  Used to boost win_prob for high-conviction setups.
    alignment_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    # Fraction of actively-voting agents within each tier that agree with the
    # tier direction (1.0 = full consensus, 0.5 = half disagree).
    consensus_long:  float = Field(default=1.0, ge=0.0, le=1.0)
    consensus_mid:   float = Field(default=1.0, ge=0.0, le=1.0)
    consensus_short: float = Field(default=1.0, ge=0.0, le=1.0)


class TradeRecipe(BaseModel):
    """Trading recipe with entry conditions."""
    name: str
    direction: Direction
    entry_trigger: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    invalidation: Optional[str] = None
    win_probability: float = Field(..., ge=0.0, le=1.0)
    expected_value: float
    risk_reward_ratio: float


class TradePlan(BaseModel):
    """Complete trade execution plan."""
    symbol: str
    recipe: TradeRecipe
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    confidence: float
    timeframes_aligned: List[Timeframe]
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MarketData(BaseModel):
    """Market data structure."""
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: Optional[float] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TechnicalFeatures(BaseModel):
    """Technical analysis features."""
    # Structure
    swing_highs: List[float] = Field(default_factory=list)
    swing_lows: List[float] = Field(default_factory=list)
    hh_hl_count: int = 0
    lh_ll_count: int = 0
    last_break: Optional[str] = None
    
    # Moving Averages
    ema20: float
    ema50: float
    ema200: float
    ema20_slope: float
    ema50_slope: float
    ema200_slope: float
    
    # Momentum
    rsi_14: float
    rsi_4: float
    macd_hist: float
    macd_hist_delta: float
    roc_10: float
    bb_percent_b: float
    
    # Volatility
    atr_14: float
    atr_5: float
    realized_vol: float
    bb_width: float
    keltner_width: float
    
    # Regime
    adx_14: float
    rvi: float
    hurst_exponent: Optional[float] = None
    atr_price_ratio: float
    vwap_distance: float
    
    # Breadth
    index_trend: float
    advance_decline: float
    above_50ema_pct: float
    above_200ema_pct: float
    sector_direction: float

    # Macro / intermarket signals (pre-mapped to this symbol's directional bias by DataLoader)
    # Positive value = macro context bullish for this pair; negative = bearish.
    dxy_dir: float = 0.0      # DXY (USD index) trend, mapped per pair
    vix_dir: float = 0.0      # VIX risk sentiment, mapped per pair
    crude_dir: float = 0.0    # WTI crude trend, mapped per pair
    yield_dir: float = 0.0    # US 10Y yield trend, mapped per pair

    # Session breakout: Asian range (00:00–08:00 UTC) above/below signal
    # Positive = above Asian high (bullish breakout); negative = below Asian low.
    # Only populated for H1 (mid) features; zero for D1/15m tiers.
    session_break_score: float = 0.0

    # RSI divergence (pre-computed on OHLC bars; most useful on 1H/4H mid tier)
    # Regular divergence:  price LL + RSI HL → bullish reversal signal.
    #                      price HH + RSI LH → bearish reversal signal.
    # Hidden divergence:   price HL + RSI LL → bullish continuation (trend pullback).
    #                      price LH + RSI HH → bearish continuation.
    bull_div_score: float = 0.0   # [0, 1] — strength of bullish divergence signal
    bear_div_score: float = 0.0   # [0, 1] — strength of bearish divergence signal

    # Weekly pivot-point proximity (ATR-normalized distance to nearest key level).
    # 10.0 = price is far from any pivot level (no obstacle).
    # < 0.5 = price is very close; entering into the level is high-risk.
    nearest_support_atr: float = 10.0   # distance to nearest support below in ATR units
    nearest_resist_atr:  float = 10.0   # distance to nearest resistance above in ATR units

    # Cross-symbol correlation divergence signals (pre-computed by DataLoader).
    # Positive = symbol is stronger than its correlations imply (bullish bias).
    # Negative = symbol is weaker than expected (bearish bias).
    corr_dxy_divergence:  float = 0.0   # divergence from normal DXY relationship
    corr_pair_divergence: float = 0.0   # divergence from most-correlated FX peer
    corr_risk_divergence: float = 0.0   # divergence from risk-appetite proxy (VIX-inverse)

    # Annualization factor used when computing realized_vol (bars per year for the
    # timeframe of this feature set).  Allows agents to de-annualize back to
    # per-bar units without a hardcoded timeframe assumption.
    # 1m FX:  252*1440 = 362 880   15m: 252*96 = 24 192   1H: 252*24 = 6 048   1D: 252
    bars_per_year: float = 252.0


class PortfolioState(BaseModel):
    """Current portfolio state."""
    equity: float
    margin_used: float
    free_margin: float
    daily_pnl: float
    daily_drawdown: float
    open_positions: List[str] = Field(default_factory=list)
    # Maps symbol → list of {ticket, type ("BUY"/"SELL"), profit, price_open}
    # Used for scale-in decisions (check direction + profit status per position)
    open_positions_map: Dict[str, List[Dict]] = Field(default_factory=dict)
    max_daily_drawdown: float
    leverage_used: float
    # Number of symbols in the portfolio universe (used for concentration-aware
    # risk scaling).  1 = single-symbol backtest, N = multi-symbol.
    # When set, graph.py caps per-symbol risk so that the theoretical max
    # portfolio exposure matches what a live multi-symbol run would produce.
    n_portfolio_symbols: int = 1


class RiskLimits(BaseModel):
    """Risk management limits."""
    base_risk_pct: float = 0.25
    max_daily_drawdown_pct: float = 2.0
    max_concurrent_trades: int = 3
    per_symbol_leverage_cap: float = 3.0
    portfolio_leverage_cap: float = 5.0
    max_correlated_positions: int = 2
    min_win_prob: float = 0.48
    min_expectancy_r: float = 0.10
    margin_free_pct_threshold: float = 0.20
    spread_guard_fraction: float = 0.20
    slippage_pips: float = 0.3


class ExecutionConfig(BaseModel):
    """Execution configuration."""
    slippage_bp: int = 2
    spread_guard_bp: int = 3
    news_blackout_minutes: int = 10
    max_tick_age_ms: int = 1000
    use_bracket_orders: bool = True
    partial_take_profits: bool = True
    trailing_stops: bool = True 