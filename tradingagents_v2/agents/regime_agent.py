"""
Regime Agent - Identifies trending vs choppy conditions and volatility state.
"""

import numpy as np
from typing import Dict, Any

from ..core.agent_base import BaseAgent
from ..core.types import AgentOutput, TechnicalFeatures, Timeframe


class RegimeAgent(BaseAgent):
    """
    Regime Agent: Identifies trending vs choppy + volatility state.
    
    Goal: Identify trending vs. choppy + volatility state.
    Features: ADX(14), RVI, Hurst exponent (optional), ATR/price, realized vol, 
             % time above/below anchored VWAP.
    Output: trendiness (0..1), vol_state (low/normal/high). 
            Penalize breakout trades in low-trendiness.
    """
    
    name: str = "RegimeAgent"
    timeframe: Timeframe = Timeframe.MID
    
    def get_required_features(self) -> list:
        """Return list of required feature names."""
        return [
            'adx_14', 'rvi', 'atr_price_ratio', 'realized_vol', 
            'vwap_distance', 'bb_width', 'keltner_width'
        ]
    
    async def analyze(self, features: TechnicalFeatures, context: Dict[str, Any] = None) -> AgentOutput:
        """
        Analyze market regime using technical features.
        
        Args:
            features: Technical analysis features
            context: Additional context
            
        Returns:
            AgentOutput with regime analysis
        """
        
        # Calculate trendiness score
        trendiness = self._calculate_trendiness(features)
        
        # Calculate volatility state
        vol_state = self._calculate_volatility_state(features)
        
        # Calculate directional score based on regime
        dir_score = self._calculate_directional_score(features, trendiness, vol_state)
        
        # Calculate confidence
        confidence = self.calculate_confidence(features, context)
        
        # Generate rationale
        rationale = self._generate_rationale(trendiness, vol_state, dir_score)
        
        # Prepare evidence
        evidence = {
            'trendiness': trendiness,
            'volatility_state': vol_state,
            'adx_14': features.adx_14,
            'rvi': features.rvi,
            'atr_price_ratio': features.atr_price_ratio,
            'realized_vol': features.realized_vol,
            'bb_width': features.bb_width,
            'keltner_width': features.keltner_width
        }
        
        return AgentOutput(
            timeframe=self.timeframe,
            dir_score=dir_score,
            conf=confidence,
            rationale=rationale,
            evidence=evidence
        )
    
    def _calculate_trendiness(self, features: TechnicalFeatures) -> float:
        """Calculate trendiness score (0-1)."""
        
        # ADX-based trendiness (0-1)
        adx_trendiness = min(features.adx_14 / 50.0, 1.0)
        
        # RVI-based trendiness
        rvi_trendiness = min(abs(features.rvi) / 0.5, 1.0)
        
        # ATR/Price ratio trendiness (higher volatility often means trending)
        atr_trendiness = min(features.atr_price_ratio / 0.05, 1.0)
        
        # Bollinger Band width (narrow bands suggest ranging, wide suggest trending)
        bb_trendiness = min(features.bb_width / 0.1, 1.0)
        
        # Combine trendiness factors
        trendiness = (
            adx_trendiness * 0.4 +
            rvi_trendiness * 0.2 +
            atr_trendiness * 0.2 +
            bb_trendiness * 0.2
        )
        
        return min(max(trendiness, 0.0), 1.0)
    
    def _calculate_volatility_state(self, features: TechnicalFeatures) -> str:
        """Calculate volatility state (low/normal/high).

        realized_vol is stored as annualised percentage (e.g. 0.08 = 8% pa).
        Thresholds are calibrated to annual vol, not per-bar vol.
          low    < 5%  pa  — very quiet FX / pre-news
          normal   5–20%pa  — typical EUR/USD, GBP/JPY
          high   > 20% pa  — indices (DAX, US30) and XAUUSD in volatile conditions
        """
        if features.realized_vol < 0.05:
            return "low"
        elif features.realized_vol > 0.20:
            return "high"
        else:
            return "normal"
    
    def _calculate_directional_score(self, features: TechnicalFeatures, 
                                   trendiness: float, vol_state: str) -> float:
        """
        Calculate directional score based on regime analysis.
        
        Returns:
            float: -1 (bearish regime) to +1 (bullish regime)
        """
        
        # VWAP distance is in ATR units. ±1 ATR → tanh(±0.5) ≈ ±0.46; ±2 ATR → ±0.76.
        vwap_score = float(np.tanh(features.vwap_distance / 2.0))

        # In ranging markets, ATTENUATE (not invert) the VWAP signal.
        # Inverting means "above VWAP = bearish" which is MeanReversionAgent's job.
        # Scale proportionally: trendiness=0.5 → full weight; trendiness=0 → 30% weight.
        trend_scale = float(np.clip(trendiness / 0.5, 0.30, 1.0))
        vwap_score *= trend_scale

        # RVI carries directional momentum (its sign = closes > opens on average).
        # tanh(rvi * 4) maps the typical ±0.5 RVI range to approximately ±0.96.
        rvi_score = float(np.tanh(features.rvi * 4.0))

        # Blend: VWAP distance (65%) + RVI direction (35%)
        dir_score = 0.65 * vwap_score + 0.35 * rvi_score

        if vol_state == "high":
            dir_score *= 0.7
        elif vol_state == "low":
            dir_score *= 1.15

        return float(np.clip(dir_score, -1.0, 1.0))
    
    def _generate_rationale(self, trendiness: float, vol_state: str, dir_score: float) -> str:
        """Generate human-readable rationale for the regime analysis."""
        
        # Trendiness description
        if trendiness > 0.7:
            trend_desc = "strong trending"
        elif trendiness > 0.4:
            trend_desc = "moderately trending"
        else:
            trend_desc = "ranging/choppy"
        
        # Volatility description
        vol_desc = f"{vol_state} volatility"
        
        # Direction description
        if dir_score > 0.5:
            direction = "bullish"
        elif dir_score < -0.5:
            direction = "bearish"
        else:
            direction = "neutral"
        
        # Combine into rationale
        rationale = (
            f"Market showing {trend_desc} conditions with {vol_desc}. "
            f"Current regime bias is {direction} "
            f"(trendiness: {trendiness:.2f}, vol_state: {vol_state})"
        )
        
        return rationale
    
    def calculate_confidence(self, features: TechnicalFeatures, context: Dict[str, Any] = None) -> float:
        """Calculate confidence score for regime analysis."""
        
        # Start with base confidence
        confidence = 0.5
        
        # Higher confidence with strong trend (clear ADX signal)
        if features.adx_14 > 25:
            confidence += 0.2
        elif features.adx_14 < 15:
            # Low ADX = regime is directionless — we can still read it but less confidently
            confidence -= 0.10

        # Higher confidence with clear volatility state
        if features.realized_vol > 0.05:  # outside "quiet" region
            confidence += 0.1

        # Higher confidence with clear VWAP distance
        if abs(features.vwap_distance) > 0.01:
            confidence += 0.1

        return float(np.clip(confidence, 0.10, 1.0))