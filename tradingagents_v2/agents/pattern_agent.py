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
        return [
            'swing_highs', 'swing_lows', 'bb_percent_b',
            'atr_14', 'ema20', 'ema50', 'adx_14', 'vwap_distance',
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

        # Guard against NaN/Inf in critical floats
        _critical = [features.bb_percent_b, features.ema20, features.ema50,
                     features.adx_14, features.vwap_distance, features.atr_14]
        if any(not np.isfinite(v) for v in _critical):
            return AgentOutput(
                timeframe=self.timeframe, dir_score=0.0, conf=0.1,
                rationale="Insufficient data (NaN detected)", evidence={},
            )

        # Calculate swing pattern score
        swing_score = self._calculate_swing_patterns(features)

        # Calculate breakout score
        breakout_score = self._calculate_breakout_patterns(features)

        # Calculate consolidation score
        consolidation_score = self._calculate_consolidation_patterns(features)

        # Calculate support/resistance score
        sr_score = self._calculate_support_resistance(features)

        # Combine scores for final pattern score
        dir_score = self._combine_pattern_scores(swing_score, breakout_score, consolidation_score, sr_score, features)

        # Calculate confidence
        confidence = self.calculate_confidence(features, context)

        self.logger.debug(
            f"[CALC] PatternAgent swing={swing_score:+.3f} breakout={breakout_score:+.3f} "
            f"consol={consolidation_score:+.3f} sr={sr_score:+.3f} "
            f"\u2192 dir={dir_score:+.4f} conf={confidence:.3f} | "
            f"break={features.last_break} bb%b={features.bb_percent_b:.3f}"
        )

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
        """Calculate score based on swing high/low patterns.

        Detects structural patterns: trending swings (HH/HL, LH/LL),
        range expansion (widening formation), and range contraction
        (narrowing wedge/triangle → breakout setup).
        """
        swing_highs = features.swing_highs
        swing_lows = features.swing_lows

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[-1] > swing_highs[-2]
            hl = swing_lows[-1] > swing_lows[-2]
            lh = swing_highs[-1] < swing_highs[-2]
            ll = swing_lows[-1] < swing_lows[-2]

            if hh and hl:        # Higher highs + higher lows = bullish trend
                swing_score = 0.7
            elif lh and ll:      # Lower highs + lower lows = bearish trend
                swing_score = -0.7
            elif hh and ll:      # Expanding range = volatile, directionless
                swing_score = 0.0
            elif lh and hl:      # Narrowing wedge = breakout imminent, neutral bias
                # Direction depends on prior trend; use EMA alignment as tiebreak
                swing_score = 0.1 if features.ema20 > features.ema50 else -0.1
            elif hh:
                swing_score = 0.35
            elif ll:
                swing_score = -0.35
            else:
                swing_score = 0.0
        else:
            swing_score = 0.0

        return float(np.clip(swing_score, -1.0, 1.0))

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
        """Calculate directional score based on BB position only.

        Pure price-position signal — last_break is NOT used here to
        avoid double-counting with _calculate_breakout_patterns.
        Inside the BB midzone the score is neutral.
        """
        bb_percent = features.bb_percent_b

        if 0.4 < bb_percent < 0.6:
            return 0.0

        # Outside midzone: weak directional nudge from BB position alone
        if bb_percent >= 0.6:
            consolidation_score = 0.15
        else:
            consolidation_score = -0.15

        return float(np.clip(consolidation_score, -1.0, 1.0))

    def _calculate_support_resistance(self, features: TechnicalFeatures) -> float:
        """Calculate score based on support/resistance levels.

        Uses VWAP distance (in ATR units) and EMA alignment to assess
        whether price is bouncing off or breaking through key levels.
        Price near VWAP + holding above EMA20 = support holding (bullish).
        Price far from VWAP in direction of EMA trend = breakout confirmation.
        """
        vd = features.vwap_distance   # ATR-normalized distance from VWAP
        bb_percent = features.bb_percent_b

        # EMA alignment base direction
        if features.ema20 > features.ema50:
            ema_dir = 1.0   # bullish alignment
        elif features.ema20 < features.ema50:
            ema_dir = -1.0  # bearish alignment
        else:
            ema_dir = 0.0

        # Near VWAP (within 0.5 ATR): price is testing a key level
        if abs(vd) < 0.5:
            # Bouncing off VWAP in direction of EMA = support/resistance holding
            sr_score = ema_dir * 0.3
        elif abs(vd) < 1.5:
            # Moderate distance: confirm with EMA and BB position
            if vd > 0 and ema_dir > 0 and bb_percent > 0.6:
                sr_score = 0.4  # above VWAP, bullish EMAs, upper BB = strength
            elif vd < 0 and ema_dir < 0 and bb_percent < 0.4:
                sr_score = -0.4  # below VWAP, bearish EMAs, lower BB = weakness
            else:
                sr_score = ema_dir * 0.15
        else:
            # Far from VWAP: extended, S/R signal is weak
            sr_score = 0.0

        return float(np.clip(sr_score, -1.0, 1.0))

    def _combine_pattern_scores(self, swing_score: float, breakout_score: float,
                               consolidation_score: float, sr_score: float,
                               features: TechnicalFeatures = None) -> float:
        """Combine individual pattern scores into final score.

        ADX-aware: strong trends boost directional patterns;
        weak ADX dampens to avoid whipsaw signals.
        """
        combined_score = (
            swing_score * 0.35 +
            breakout_score * 0.30 +
            sr_score * 0.20 +
            consolidation_score * 0.15
        )

        if features is not None:
            adx = features.adx_14
            if adx > 30:
                combined_score *= min(1.0 + (adx - 30) / 60, 1.25)
            elif adx < 15:
                combined_score *= max(0.5, adx / 15)

        return float(np.clip(combined_score, -1.0, 1.0))

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
        """Calculate confidence score for pattern analysis.

        Adds ADX awareness and penalizes conflicting sub-scores.
        """
        confidence = 0.5

        # Higher confidence with clear swing patterns
        if len(features.swing_highs) >= 2 and len(features.swing_lows) >= 2:
            confidence += 0.15

        # Higher confidence with a confirmed structural breakout
        if features.last_break in ["bullish", "bearish"]:
            confidence += 0.15

        # Higher confidence with clear BB position
        if features.bb_percent_b < 0.3 or features.bb_percent_b > 0.7:
            confidence += 0.1

        # ADX awareness: strong trend = more trustworthy patterns
        adx = features.adx_14
        if adx > 30:
            confidence += min((adx - 30) / 60, 0.10)
        elif adx < 15:
            confidence -= 0.10

        # Penalize conflicting sub-scores (swing vs breakout disagree)
        swing_score = self._calculate_swing_patterns(features)
        breakout_score = self._calculate_breakout_patterns(features)
        if swing_score != 0.0 and breakout_score != 0.0:
            if np.sign(swing_score) != np.sign(breakout_score):
                confidence -= 0.15

        return float(np.clip(confidence, 0.0, 1.0)) 