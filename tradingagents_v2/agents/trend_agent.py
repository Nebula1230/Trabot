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
            'ema20', 'ema50', 'ema200', 'ema20_slope', 'ema50_slope', 'ema200_slope'
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

        # Calculate structure score
        structure_score = self._calculate_structure_score(features)

        # Calculate MA alignment score
        ma_score = self._calculate_ma_alignment(features)

        # Calculate slope score
        slope_score = self._calculate_slope_score(features)

        # Combine scores for final directional score
        dir_score = self._combine_scores(structure_score, ma_score, slope_score)

        # Calculate confidence
        confidence = self.calculate_confidence(features, context)

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
        """Calculate score based on market structure (HH/HL vs LH/LL)."""

        # Higher highs and higher lows indicate bullish structure
        if features.hh_hl_count > features.lh_ll_count:
            # Bullish structure
            structure_score = min(features.hh_hl_count / 3.0, 1.0)
        elif features.lh_ll_count > features.hh_hl_count:
            # Bearish structure
            structure_score = -min(features.lh_ll_count / 3.0, 1.0)
        else:
            # Neutral structure
            structure_score = 0.0

        # Adjust based on last break
        # last_break is set to "bullish" or "bearish" by _detect_swings
        if features.last_break == "bullish":
            structure_score += 0.2
        elif features.last_break == "bearish":
            structure_score -= 0.2

        return np.clip(structure_score, -1.0, 1.0)

    def _calculate_ma_alignment(self, features: TechnicalFeatures) -> float:
        """Calculate score based on EMA alignment and stack."""

        # Check if EMAs are properly stacked (bullish: 20 > 50 > 200)
        if (features.ema20 > features.ema50 > features.ema200):
            # Bullish stack
            ma_score = 0.8
        elif (features.ema20 < features.ema50 < features.ema200):
            # Bearish stack
            ma_score = -0.8
        else:
            # Mixed alignment
            ma_score = 0.0

        # Adjust based on price position relative to EMAs
        if features.ema20 > features.ema50:
            ma_score += 0.2
        else:
            ma_score -= 0.2

        return np.clip(ma_score, -1.0, 1.0)

    def _calculate_slope_score(self, features: TechnicalFeatures) -> float:
        """Calculate score based on EMA slopes."""

        # Average slope across EMAs
        avg_slope = (features.ema20_slope + features.ema50_slope + features.ema200_slope) / 3.0

        # Normalize slope to [-1, 1] range
        # Assuming slope is in price units per period
        normalized_slope = np.tanh(avg_slope * 100)  # Scale factor for normalization

        return normalized_slope

    def _combine_scores(self, structure_score: float, ma_score: float, slope_score: float) -> float:
        """Combine individual scores into final directional score."""

        # Weighted combination
        # Structure gets highest weight as it's most reliable
        combined_score = (
            structure_score * 0.5 +
            ma_score * 0.3 +
            slope_score * 0.2
        )

        return np.clip(combined_score, -1.0, 1.0)

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