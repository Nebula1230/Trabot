"""
Pattern Agent - Price pattern recognition and analysis.
"""

import numpy as np
from typing import Dict, Any

from ..core.agent_base import BaseAgent
from ..core.types import AgentOutput, TechnicalFeatures, Timeframe


class PatternAgent(BaseAgent):
    """
    Pattern Agent: Price pattern recognition and analysis.

    Goal: Price pattern recognition and analysis.
    Features: Flags, pullbacks, inside bars, breakouts, support/resistance.
    Signal: Pattern-based directional score.
    """

    name: str = "PatternAgent"
    timeframe: Timeframe = Timeframe.SHORT

    def get_required_features(self) -> list:
        """Return list of required feature names."""
        # last_break is Optional — None means neutral (no clear breakout), not missing
        return [
            'swing_highs', 'swing_lows', 'bb_percent_b',
            'atr_14', 'ema20', 'ema50'
        ]

    async def analyze(self, features: TechnicalFeatures, context: Dict[str, Any] = None) -> AgentOutput:
        """
        Analyze price patterns for trading opportunities.

        Args:
            features: Technical analysis features
            context: Additional context

        Returns:
            AgentOutput with pattern analysis
        """

        # Calculate swing pattern score
        swing_score = self._calculate_swing_patterns(features)

        # Calculate breakout score
        breakout_score = self._calculate_breakout_patterns(features)

        # Calculate consolidation score
        consolidation_score = self._calculate_consolidation_patterns(features)

        # Calculate support/resistance score
        sr_score = self._calculate_support_resistance(features)

        # Combine scores for final pattern score
        dir_score = self._combine_pattern_scores(swing_score, breakout_score, consolidation_score, sr_score)

        # Calculate confidence
        confidence = self.calculate_confidence(features, context)

        # Generate rationale
        rationale = self._generate_rationale(swing_score, breakout_score, consolidation_score, sr_score, dir_score)

        # Prepare evidence
        evidence = {
            'swing_score': swing_score,
            'breakout_score': breakout_score,
            'consolidation_score': consolidation_score,
            'sr_score': sr_score,
            'swing_highs': features.swing_highs,
            'swing_lows': features.swing_lows,
            'last_break': features.last_break,
            'bb_percent_b': features.bb_percent_b,
            'atr_14': features.atr_14,
            'ema20': features.ema20,
            'ema50': features.ema50
        }

        return AgentOutput(
            timeframe=self.timeframe,
            dir_score=dir_score,
            conf=confidence,
            rationale=rationale,
            evidence=evidence
        )

    def _calculate_swing_patterns(self, features: TechnicalFeatures) -> float:
        """Calculate score based on swing high/low patterns."""

        swing_highs = features.swing_highs
        swing_lows = features.swing_lows

        # Swing pattern analysis
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Check for higher highs and higher lows (bullish)
            if (swing_highs[-1] > swing_highs[-2] and 
                swing_lows[-1] > swing_lows[-2]):
                swing_score = 0.7
            # Check for lower highs and lower lows (bearish)
            elif (swing_highs[-1] < swing_highs[-2] and 
                  swing_lows[-1] < swing_lows[-2]):
                swing_score = -0.7
            # Check for higher highs but lower lows (potential reversal)
            elif swing_highs[-1] > swing_highs[-2] and swing_lows[-1] < swing_lows[-2]:
                swing_score = 0.3
            # Check for lower highs but higher lows (potential reversal)
            elif swing_highs[-1] < swing_highs[-2] and swing_lows[-1] > swing_lows[-2]:
                swing_score = -0.3
            else:
                swing_score = 0.0
        else:
            swing_score = 0.0

        return np.clip(swing_score, -1.0, 1.0)

    def _calculate_breakout_patterns(self, features: TechnicalFeatures) -> float:
        """Calculate score based on breakout patterns."""

        last_break = features.last_break
        bb_percent = features.bb_percent_b

        # Breakout pattern analysis.
        # last_break is set to "bullish" or "bearish" by _detect_swings.
        # Confirm breakout strength with BB position: near upper band on a bullish
        # break → strong; lower band on bearish break → strong; otherwise moderate.
        if last_break == "bullish":
            if bb_percent > 0.8:  # Strong breakout (price near upper BB)
                breakout_score = 0.8
            else:  # Moderate breakout
                breakout_score = 0.4
        elif last_break == "bearish":
            if bb_percent < 0.2:  # Strong breakout (price near lower BB)
                breakout_score = -0.8
            else:  # Moderate breakout
                breakout_score = -0.4
        else:
            # No clear breakout
            breakout_score = 0.0

        return np.clip(breakout_score, -1.0, 1.0)

    def _calculate_consolidation_patterns(self, features: TechnicalFeatures) -> float:
        """Calculate directional score based on consolidation breakout context.

        When price is NOT in the BB midzone (i.e. breaking out of consolidation),
        return a weak directional bias aligned with the breakout direction.
        Inside the midzone the score is neutral — the last_break field (already
        scored in _calculate_breakout_patterns) handles that case.
        """
        bb_percent = features.bb_percent_b
        last_break = features.last_break

        if 0.4 < bb_percent < 0.6:
            # Inside consolidation range — neutral
            return 0.0

        # Outside midzone: confirm the side with a weak directional nudge
        if bb_percent >= 0.6:
            # Upper half — mild bullish lean
            consolidation_score = 0.2 if last_break == "bullish" else 0.1
        else:
            # Lower half — mild bearish lean
            consolidation_score = -0.2 if last_break == "bearish" else -0.1

        return float(np.clip(consolidation_score, -1.0, 1.0))

    def _calculate_support_resistance(self, features: TechnicalFeatures) -> float:
        """Calculate score based on support/resistance levels."""

        ema20 = features.ema20
        ema50 = features.ema50
        bb_percent = features.bb_percent_b

        # Support/resistance analysis using EMAs
        # Price near EMA20 (short-term support/resistance)
        if 0.45 < bb_percent < 0.55:  # Price near middle of BB
            if ema20 > ema50:  # Bullish EMA alignment
                sr_score = 0.3
            elif ema20 < ema50:  # Bearish EMA alignment
                sr_score = -0.3
            else:
                sr_score = 0.0
        else:
            sr_score = 0.0

        return np.clip(sr_score, -1.0, 1.0)

    def _combine_pattern_scores(self, swing_score: float, breakout_score: float,
                               consolidation_score: float, sr_score: float) -> float:
        """Combine individual pattern scores into final score."""

        # Weighted combination
        # Swing patterns and breakouts get highest weights
        combined_score = (
            swing_score * 0.4 +
            breakout_score * 0.4 +
            sr_score * 0.15 +
            consolidation_score * 0.05
        )

        return np.clip(combined_score, -1.0, 1.0)

    def _generate_rationale(self, swing_score: float, breakout_score: float,
                           consolidation_score: float, sr_score: float, dir_score: float) -> str:
        """Generate human-readable rationale for the pattern analysis."""

        # Swing pattern description
        if swing_score > 0.5:
            swing_desc = "bullish swing pattern"
        elif swing_score > 0.2:
            swing_desc = "moderate bullish swing pattern"
        elif swing_score < -0.5:
            swing_desc = "bearish swing pattern"
        elif swing_score < -0.2:
            swing_desc = "moderate bearish swing pattern"
        else:
            swing_desc = "neutral swing pattern"

        # Breakout description
        if breakout_score > 0.5:
            breakout_desc = "bullish breakout pattern"
        elif breakout_score > 0.2:
            breakout_desc = "moderate bullish breakout pattern"
        elif breakout_score < -0.5:
            breakout_desc = "bearish breakout pattern"
        elif breakout_score < -0.2:
            breakout_desc = "moderate bearish breakout pattern"
        else:
            breakout_desc = "no clear breakout pattern"

        # Overall pattern
        if dir_score > 0.5:
            pattern = "strong bullish pattern formation"
        elif dir_score > 0.2:
            pattern = "moderate bullish pattern formation"
        elif dir_score < -0.5:
            pattern = "strong bearish pattern formation"
        elif dir_score < -0.2:
            pattern = "moderate bearish pattern formation"
        else:
            pattern = "neutral pattern formation"

        rationale = (
            f"Swing analysis shows {swing_desc}, breakout analysis indicates {breakout_desc}. "
            f"Overall pattern formation is {pattern} "
            f"(Swing: {swing_score:.2f}, Breakout: {breakout_score:.2f}, "
            f"Consolidation: {consolidation_score:.2f}, S/R: {sr_score:.2f})"
        )

        return rationale

    def calculate_confidence(self, features: TechnicalFeatures, context: Dict[str, Any] = None) -> float:
        """Calculate confidence score for pattern analysis."""

        # Start with base confidence
        confidence = 0.5

        # Higher confidence with clear swing patterns
        if len(features.swing_highs) >= 2 and len(features.swing_lows) >= 2:
            confidence += 0.2

        # Higher confidence with clear breakouts
        if features.last_break in ["high", "low"]:
            confidence += 0.2

        # Higher confidence with clear BB position
        if features.bb_percent_b < 0.3 or features.bb_percent_b > 0.7:
            confidence += 0.1

        # Ensure confidence is within bounds
        return min(max(confidence, 0.0), 1.0) 