"""
Trend Agent - Analyzes market structure and moving average alignment.
"""

import numpy as np
from typing import Dict, Any

from ..core.agent_base import BaseAgent
from ..core.types import AgentOutput, TechnicalFeatures, Timeframe


class TrendAgent(BaseAgent):
    """
    Trend Agent: Direction via market structure + filtered MAs.

    Goal: Direction via market structure + MAs.
    Features: HH/HL or LH/LL detection, EMA ribbon (20/50/200), slope & stack.
    Signal: Directional score from structure + MA alignment.
    """

    name: str = "TrendAgent"
    timeframe: Timeframe = Timeframe.LONG

    def get_required_features(self) -> list:
        """Return list of required feature names."""
        return [
            'swing_highs', 'swing_lows', 'hh_hl_count', 'lh_ll_count',
            'ema20', 'ema50', 'ema200', 'ema20_slope', 'ema50_slope', 'ema200_slope',
            'adx_14', 'atr_14',
        ]

    async def analyze(self, features: TechnicalFeatures, context: Dict[str, Any] = None) -> AgentOutput:
        """
        Analyze trend using market structure and moving averages.

        Args:
            features: Technical analysis features
            context: Additional context

        Returns:
            AgentOutput with trend analysis
        """

        # Guard against NaN/Inf in critical floats
        _critical = [features.ema20, features.ema50, features.ema200,
                     features.ema20_slope, features.ema50_slope, features.ema200_slope,
                     features.adx_14, features.atr_14]
        if any(not np.isfinite(v) for v in _critical):
            return AgentOutput(
                timeframe=self.timeframe, dir_score=0.0, conf=0.1,
                rationale="Insufficient data (NaN detected)", evidence={},
            )

        # Calculate structure score
        structure_score = self._calculate_structure_score(features)

        # Calculate MA alignment score
        ma_score = self._calculate_ma_alignment(features)

        # Calculate slope score
        slope_score = self._calculate_slope_score(features)

        # Combine scores for final directional score
        dir_score = self._combine_scores(structure_score, ma_score, slope_score, features)

        # Calculate confidence
        confidence = self.calculate_confidence(features, context)

        self.logger.debug(
            f"[CALC] TrendAgent struct={structure_score:+.3f} ma={ma_score:+.3f} "
            f"slope={slope_score:+.3f} → dir={dir_score:+.4f} conf={confidence:.3f} | "
            f"hh={features.hh_hl_count} ll={features.lh_ll_count} "
            f"break={features.last_break} ema20={features.ema20:.5f} "
            f"ema50={features.ema50:.5f} ema200={features.ema200:.5f}"
        )

        # Generate rationale
        rationale = self._generate_rationale(structure_score, ma_score, slope_score, dir_score)

        # Prepare evidence
        evidence = {
            'structure_score': structure_score,
            'ma_score': ma_score,
            'slope_score': slope_score,
            'hh_hl_count': features.hh_hl_count,
            'lh_ll_count': features.lh_ll_count,
            'ema_alignment': self._get_ema_alignment(features),
            'last_break': features.last_break
        }

        return AgentOutput(
            timeframe=self.timeframe,
            dir_score=dir_score,
            conf=confidence,
            rationale=rationale,
            evidence=evidence
        )

    def _calculate_structure_score(self, features: TechnicalFeatures) -> float:
        """Calculate score based on market structure (HH/HL vs LH/LL).

        Uses the NET ratio of bullish-to-bearish swings instead of a fixed
        denominator, so the score self-calibrates to the lookback window.
        """
        hh = features.hh_hl_count
        ll = features.lh_ll_count
        total = hh + ll

        if total == 0:
            structure_score = 0.0
        else:
            # Net ratio in [-1, +1]: +1 = all HH/HL, -1 = all LH/LL
            structure_score = (hh - ll) / total

        # Adjust based on last break
        if features.last_break == "bullish":
            structure_score += 0.2
        elif features.last_break == "bearish":
            structure_score -= 0.2

        return float(np.clip(structure_score, -1.0, 1.0))

    def _calculate_ma_alignment(self, features: TechnicalFeatures) -> float:
        """Calculate score based on EMA alignment and stack."""

        # Full EMA stack: strongest trend signal
        if features.ema20 > features.ema50 > features.ema200:
            ma_score = 0.8
        elif features.ema20 < features.ema50 < features.ema200:
            ma_score = -0.8
        else:
            # Partial / mixed alignment: give a weak directional nudge
            # Only check ema20 vs ema50 when NOT in a full stack to avoid
            # double-counting (full-stack already captures this relationship).
            if features.ema20 > features.ema50:
                ma_score = 0.2
            elif features.ema20 < features.ema50:
                ma_score = -0.2
            else:
                ma_score = 0.0

        return float(np.clip(ma_score, -1.0, 1.0))

    def _calculate_slope_score(self, features: TechnicalFeatures) -> float:
        """Calculate score based on EMA slopes, ATR-normalized.

        Normalizes slope by ATR so the score is instrument-independent:
        slope/ATR of 0.05 on EURUSD has the same meaning as on USDJPY.
        Weighted average: EMA20 is fastest-reacting, EMA200 is slowest.
        """
        atr = max(features.atr_14, 1e-9)
        s20  = features.ema20_slope  / atr
        s50  = features.ema50_slope  / atr
        s200 = features.ema200_slope / atr

        # Weighted: fast EMA matters most for recent direction
        avg = s20 * 0.50 + s50 * 0.30 + s200 * 0.20
        # tanh(avg * 20): a 0.05 ATR/bar slope → tanh(1) ≈ 0.76
        return float(np.tanh(avg * 20))

    def _combine_scores(self, structure_score: float, ma_score: float, slope_score: float,
                         features: TechnicalFeatures = None) -> float:
        """Combine individual scores into final directional score.

        ADX-aware weighting: when ADX > 25 (strong trend), boost the score;
        when ADX < 15 (range-bound), attenuate to avoid whipsaws.
        """
        combined_score = (
            structure_score * 0.45 +
            ma_score * 0.30 +
            slope_score * 0.25
        )

        # ADX scaling: strong trend amplifies, weak trend dampens
        if features is not None:
            adx = features.adx_14
            if adx > 25:
                combined_score *= min(1.0 + (adx - 25) / 50, 1.3)
            elif adx < 15:
                combined_score *= max(0.5, adx / 15)

        return float(np.clip(combined_score, -1.0, 1.0))

    def _get_ema_alignment(self, features: TechnicalFeatures) -> str:
        """Get human-readable EMA alignment description."""

        if features.ema20 > features.ema50 > features.ema200:
            return "bullish_stack"
        elif features.ema20 < features.ema50 < features.ema200:
            return "bearish_stack"
        else:
            return "mixed"

    def _generate_rationale(self, structure_score: float, ma_score: float, 
                           slope_score: float, dir_score: float) -> str:
        """Generate human-readable rationale for the trend analysis."""

        # Structure description
        if structure_score > 0.5:
            structure_desc = "strong bullish structure"
        elif structure_score > 0.2:
            structure_desc = "moderate bullish structure"
        elif structure_score < -0.5:
            structure_desc = "strong bearish structure"
        elif structure_score < -0.2:
            structure_desc = "moderate bearish structure"
        else:
            structure_desc = "neutral structure"

        # MA description
        if ma_score > 0.5:
            ma_desc = "bullish MA alignment"
        elif ma_score < -0.5:
            ma_desc = "bearish MA alignment"
        else:
            ma_desc = "mixed MA alignment"

        # Slope description
        if slope_score > 0.3:
            slope_desc = "positive momentum"
        elif slope_score < -0.3:
            slope_desc = "negative momentum"
        else:
            slope_desc = "neutral momentum"

        # Overall direction
        if dir_score > 0.5:
            direction = "bullish"
        elif dir_score < -0.5:
            direction = "bearish"
        else:
            direction = "neutral"

        rationale = (
            f"Market shows {structure_desc} with {ma_desc} and {slope_desc}. "
            f"Overall trend bias is {direction} "
            f"(structure: {structure_score:.2f}, MA: {ma_score:.2f}, slope: {slope_score:.2f})"
        )

        return rationale

    def calculate_confidence(self, features: TechnicalFeatures, context: Dict[str, Any] = None) -> float:
        """Calculate confidence score for trend analysis."""

        # Start with base confidence
        confidence = 0.5

        # Higher confidence with clear structure
        if abs(features.hh_hl_count - features.lh_ll_count) >= 2:
            confidence += 0.2

        # Higher confidence with clear EMA stack
        if (features.ema20 > features.ema50 > features.ema200) or (features.ema20 < features.ema50 < features.ema200):
            confidence += 0.2

        # Higher confidence with consistent slopes
        slopes = [features.ema20_slope, features.ema50_slope, features.ema200_slope]
        if all(s > 0 for s in slopes) or all(s < 0 for s in slopes):
            confidence += 0.1

        # Ensure confidence is within bounds
        return min(max(confidence, 0.0), 1.0) 