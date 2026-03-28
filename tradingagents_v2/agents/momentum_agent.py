"""
Momentum Agent - Confirms impulse moves and avoids exhausted moves.
"""

import numpy as np
from typing import Dict, Any

from ..core.agent_base import BaseAgent
from ..core.types import AgentOutput, TechnicalFeatures, Timeframe


class MomentumAgent(BaseAgent):
    """
    Momentum Agent: Confirms impulse moves in trend direction while avoiding exhausted moves.

    Goal: Confirm impulse moves in trend direction while avoiding exhausted moves.
    Features: RSI regime logic, MACD histogram changes, ROC, BB %B.
    Signal: Momentum confirmation score.
    """

    name: str = "MomentumAgent"
    timeframe: Timeframe = Timeframe.MID

    def get_required_features(self) -> list:
        """Return list of required feature names."""
        return [
            'rsi_14', 'rsi_4', 'macd_hist', 'macd_hist_delta', 'roc_10', 'bb_percent_b'
        ]

    async def analyze(self, features: TechnicalFeatures, context: Dict[str, Any] = None) -> AgentOutput:
        """
        Analyze momentum using RSI, MACD, and other momentum indicators.

        Args:
            features: Technical analysis features
            context: Additional context

        Returns:
            AgentOutput with momentum analysis
        """

        # Calculate RSI momentum score
        rsi_score = self._calculate_rsi_momentum(features)

        # Calculate MACD momentum score
        macd_score = self._calculate_macd_momentum(features)

        # Calculate ROC momentum score
        roc_score = self._calculate_roc_momentum(features)

        # Calculate BB momentum score
        bb_score = self._calculate_bb_momentum(features)

        # Combine scores for final momentum score
        dir_score = self._combine_momentum_scores(rsi_score, macd_score, roc_score, bb_score)

        # Calculate confidence
        confidence = self.calculate_confidence(features, context)

        # Generate rationale
        rationale = self._generate_rationale(rsi_score, macd_score, roc_score, bb_score, dir_score)

        # Prepare evidence
        evidence = {
            'rsi_score': rsi_score,
            'macd_score': macd_score,
            'roc_score': roc_score,
            'bb_score': bb_score,
            'rsi_14': features.rsi_14,
            'rsi_4': features.rsi_4,
            'macd_hist': features.macd_hist,
            'macd_hist_delta': features.macd_hist_delta,
            'roc_10': features.roc_10,
            'bb_percent_b': features.bb_percent_b
        }

        return AgentOutput(
            timeframe=self.timeframe,
            dir_score=dir_score,
            conf=confidence,
            rationale=rationale,
            evidence=evidence
        )

    def _calculate_rsi_momentum(self, features: TechnicalFeatures) -> float:
        """Calculate momentum score based on RSI regime logic.

        Interpretation: RSI is an impulse gauge, not a mean-reversion signal.
        - 40-70 rising: healthy trend momentum
        - >70 still rising: strong momentum (not overbought penalty — MeanReversionAgent handles that)
        - <40 falling: weak / bearish momentum
        """
        rsi_14 = features.rsi_14
        rsi_4 = features.rsi_4
        rising = rsi_4 > rsi_14  # short RSI above long RSI = rising momentum

        if rsi_14 > 70 and rising:       # Strong bullish momentum, trend running
            rsi_score = 0.7
        elif 50 < rsi_14 <= 70 and rising:  # Healthy bullish impulse
            rsi_score = 0.6
        elif 40 < rsi_14 <= 50 and rising:  # Recovering, mild bullish
            rsi_score = 0.2
        elif 40 < rsi_14 < 70 and not rising:  # Moderating — weaken signal
            rsi_score = -0.1
        elif rsi_14 <= 40 and rising:    # Bouncing from weakness — neutral
            rsi_score = 0.0
        elif rsi_14 < 30 and not rising:   # Deep oversold AND still falling — strong bear
            rsi_score = -0.7
        elif rsi_14 <= 40 and not rising:  # Falling into weakness — bearish
            rsi_score = -0.5
        elif rsi_14 > 70 and not rising:  # Fading from highs — mild pullback
            rsi_score = -0.2
        else:
            rsi_score = 0.0

        return np.clip(rsi_score, -1.0, 1.0)

    def _calculate_macd_momentum(self, features: TechnicalFeatures) -> float:
        """Calculate momentum score based on MACD histogram changes."""

        macd_hist = features.macd_hist
        macd_delta = features.macd_hist_delta

        # MACD momentum analysis
        if macd_hist > 0 and macd_delta > 0:  # Increasing bullish momentum
            macd_score = 0.8
        elif macd_hist > 0 and macd_delta < 0:  # Decreasing bullish momentum
            macd_score = 0.2
        elif macd_hist < 0 and macd_delta < 0:  # Increasing bearish momentum
            macd_score = -0.8
        elif macd_hist < 0 and macd_delta > 0:  # Decreasing bearish momentum
            macd_score = -0.2
        else:  # Neutral
            macd_score = 0.0

        return np.clip(macd_score, -1.0, 1.0)

    def _calculate_roc_momentum(self, features: TechnicalFeatures) -> float:
        """Calculate momentum score based on Rate of Change."""

        roc = features.roc_10

        # roc_10 is a fractional return: (close - close[10]) / close[10].
        # Normalise by atr_price_ratio (same units) so the score is
        # instrument-independent: a 1-ATR 10-bar move maps to tanh(1) ≈ 0.76.
        # Without this, EURUSD roc ≈ 0.001 gives tanh(0.001*10) ≈ 0.01 — near zero.
        atr_ratio = max(features.atr_price_ratio, 0.001)
        roc_score = float(np.tanh(roc / atr_ratio))

        return float(np.clip(roc_score, -1.0, 1.0))

    def _calculate_bb_momentum(self, features: TechnicalFeatures) -> float:
        """Calculate momentum score based on Bollinger Band position.

        Momentum interpretation (NOT mean-reversion — MeanReversionAgent handles that):
        - Price above upper band or near top: momentum is running bullish
        - Price below lower band or near bottom: momentum is running bearish
        - Price in mid-range: neutral momentum
        """
        bb_percent = features.bb_percent_b

        if bb_percent > 1.0:    # Outside upper band — strong bullish momentum
            bb_score = 0.6
        elif bb_percent > 0.8:  # Near upper band — bullish momentum
            bb_score = 0.4
        elif bb_percent > 0.6:  # Upper half — mild bullish bias
            bb_score = 0.2
        elif bb_percent < 0.0:  # Outside lower band — strong bearish momentum
            bb_score = -0.6
        elif bb_percent < 0.2:  # Near lower band — bearish momentum
            bb_score = -0.4
        elif bb_percent < 0.4:  # Lower half — mild bearish bias
            bb_score = -0.2
        else:                   # Middle range — neutral
            bb_score = 0.0

        return np.clip(bb_score, -1.0, 1.0)

    def _combine_momentum_scores(self, rsi_score: float, macd_score: float, 
                                roc_score: float, bb_score: float) -> float:
        """Combine individual momentum scores into final score."""

        # Weighted combination
        # MACD gets highest weight as it's most reliable for momentum
        combined_score = (
            macd_score * 0.4 +
            rsi_score * 0.3 +
            roc_score * 0.2 +
            bb_score * 0.1
        )

        return np.clip(combined_score, -1.0, 1.0)

    def _generate_rationale(self, rsi_score: float, macd_score: float, 
                           roc_score: float, bb_score: float, dir_score: float) -> str:
        """Generate human-readable rationale for the momentum analysis."""

        # RSI description
        if rsi_score > 0.5:
            rsi_desc = "strong bullish RSI momentum"
        elif rsi_score > 0.2:
            rsi_desc = "moderate bullish RSI momentum"
        elif rsi_score < -0.3:
            rsi_desc = "bearish RSI momentum"
        else:
            rsi_desc = "neutral RSI momentum"

        # MACD description
        if macd_score > 0.5:
            macd_desc = "strong bullish MACD momentum"
        elif macd_score > 0.2:
            macd_desc = "moderate bullish MACD momentum"
        elif macd_score < -0.5:
            macd_desc = "strong bearish MACD momentum"
        elif macd_score < -0.2:
            macd_desc = "moderate bearish MACD momentum"
        else:
            macd_desc = "neutral MACD momentum"

        # Overall momentum
        if dir_score > 0.5:
            momentum = "strong bullish momentum"
        elif dir_score > 0.2:
            momentum = "moderate bullish momentum"
        elif dir_score < -0.5:
            momentum = "strong bearish momentum"
        elif dir_score < -0.2:
            momentum = "moderate bearish momentum"
        else:
            momentum = "neutral momentum"

        rationale = (
            f"RSI shows {rsi_desc}, MACD indicates {macd_desc}. "
            f"Overall momentum is {momentum} "
            f"(RSI: {rsi_score:.2f}, MACD: {macd_score:.2f}, ROC: {roc_score:.2f}, BB: {bb_score:.2f})"
        )

        return rationale

    def calculate_confidence(self, features: TechnicalFeatures, context: Dict[str, Any] = None) -> float:
        """Calculate confidence score for momentum analysis."""

        # Start with base confidence
        confidence = 0.5

        # Higher confidence with clear RSI signals
        if 30 < features.rsi_14 < 70:
            confidence += 0.1

        # Higher confidence with clear MACD momentum
        if abs(features.macd_hist_delta) > 0.05:
            confidence += 0.2

        # Higher confidence with clear ROC
        if abs(features.roc_10) > 0.02:
            confidence += 0.1

        # Higher confidence with clear BB position
        if features.bb_percent_b < 0.3 or features.bb_percent_b > 0.7:
            confidence += 0.1

        # Ensure confidence is within bounds
        return min(max(confidence, 0.0), 1.0) 