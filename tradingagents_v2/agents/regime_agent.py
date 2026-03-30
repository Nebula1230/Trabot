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
        # NaN guard: if any critical feature is NaN/Inf, return neutral
        _critical = [
            features.adx_14, features.rvi, features.atr_price_ratio,
            features.realized_vol, features.vwap_distance,
            features.bb_width, features.keltner_width,
        ]
        if any(not np.isfinite(v) for v in _critical):
            self.logger.debug("[CALC] RegimeAgent NaN/Inf in features — returning neutral")
            return AgentOutput(
                timeframe=self.timeframe, dir_score=0.0, conf=0.1,
                rationale="Insufficient data (NaN detected)", evidence={},
            )
        
        # Calculate trendiness score
        trendiness = self._calculate_trendiness(features)
        
        # Calculate volatility state
        vol_state = self._calculate_volatility_state(features)
        
        # Calculate directional score based on regime
        dir_score = self._calculate_directional_score(features, trendiness, vol_state)
        
        # Calculate confidence
        confidence = self.calculate_confidence(features, context)

        self.logger.debug(
            f"[CALC] RegimeAgent trendiness={trendiness:.3f} vol_state={vol_state} "
            f"→ dir={dir_score:+.4f} conf={confidence:.3f} | "
            f"adx={features.adx_14:.1f} rvi={features.rvi:.3f} "
            f"atr_ratio={features.atr_price_ratio:.5f} "
            f"bb_w={features.bb_width:.5f} kc_w={features.keltner_width:.5f}"
        )
        
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
        """Calculate trendiness score (0-1).

        ADX is the primary trending indicator.  Supplemented by RVI
        (close vs open bias) and BB-width expansion (trending = wide bands).
        Calibrated so ADX=25 → trendiness≈0.50 (standard trend threshold).
        """
        # ADX: 25 = standard trend threshold, 50+ = very strong trend
        adx_trendiness = float(np.clip(features.adx_14 / 50.0, 0.0, 1.0))

        # RVI: abs(rvi) > 0.3 = strong close-vs-open directional bias
        rvi_trendiness = float(np.clip(abs(features.rvi) / 0.4, 0.0, 1.0))

        # BB/KC ratio: when BB > KC, bands have expanded = trending
        kc = max(features.keltner_width, 1e-9)
        band_ratio = features.bb_width / kc
        bb_trendiness = float(np.clip((band_ratio - 0.8) / 0.6, 0.0, 1.0))

        # ATR/price ratio: elevated vol → trending environment
        # Adaptive divisor: FX pairs (atr_price_ratio ~0.0003–0.0006)
        # need a much smaller divisor than indices (~0.01–0.02).
        _apr = features.atr_price_ratio
        _apr_div = 0.002 if _apr < 0.005 else 0.02
        atr_trendiness = float(np.clip(_apr / _apr_div, 0.0, 1.0))

        trendiness = (
            adx_trendiness * 0.45 +
            rvi_trendiness * 0.20 +
            bb_trendiness  * 0.20 +
            atr_trendiness * 0.15
        )

        return float(np.clip(trendiness, 0.0, 1.0))
    
    def _calculate_volatility_state(self, features: TechnicalFeatures) -> str:
        """Calculate volatility state with more granularity.

        realized_vol is annualised (e.g. 0.08 = 8% pa).
        Returns 4 states for finer risk control.
        """
        rv = features.realized_vol
        if rv < 0.03:
            return "very_low"    # dead quiet
        elif rv < 0.06:
            return "low"         # quiet FX
        elif rv <= 0.18:
            return "normal"      # typical EUR/USD, GBP/JPY
        elif rv <= 0.30:
            return "high"        # elevated (indices/gold/volatile events)
        else:
            return "extreme"     # crisis/flash crash
    
    def _calculate_directional_score(self, features: TechnicalFeatures, 
                                   trendiness: float, vol_state: str) -> float:
        """
        Calculate directional score based on regime analysis.
        
        Returns:
            float: -1 (bearish regime) to +1 (bullish regime)
        """
        # VWAP distance: ±1 ATR → tanh(±0.5) ≈ ±0.46; ±2 ATR → ±0.76.
        vwap_score = float(np.tanh(features.vwap_distance / 2.0))

        # In ranging markets, attenuate (not invert) the VWAP signal.
        trend_scale = float(np.clip(trendiness / 0.5, 0.30, 1.0))
        vwap_score *= trend_scale

        # RVI carries directional momentum (sign = closes > opens on average).
        rvi_score = float(np.tanh(features.rvi * 4.0))

        # Blend: VWAP distance (60%) + RVI direction (40%)
        dir_score = 0.60 * vwap_score + 0.40 * rvi_score

        # Vol state scaling: extreme vol dampens (noise), very low dampens (no energy)
        _vol_scale = {
            "very_low": 0.6,
            "low":      1.1,
            "normal":   1.0,
            "high":     0.75,
            "extreme":  0.4,
        }
        dir_score *= _vol_scale.get(vol_state, 1.0)

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

        # Higher confidence with clear VWAP distance (in ATR units)
        if abs(features.vwap_distance) > 0.5:
            confidence += 0.1

        return float(np.clip(confidence, 0.10, 1.0))