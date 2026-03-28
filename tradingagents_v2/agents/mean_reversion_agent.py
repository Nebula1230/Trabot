"""
Mean Reversion Agent - Detects extension and potential reversal points.
"""

import numpy as np
from typing import Dict, Any

from ..core.agent_base import BaseAgent
from ..core.types import AgentOutput, TechnicalFeatures, Timeframe


class MeanReversionAgent(BaseAgent):
    """
    Mean Reversion Agent: Detects extension and potential reversal points.

    Goal: Detect extension and potential reversal points.
    Features: BB %B, Keltner position, RSI extremes, VWAP distance.
    Signal: Mean reversion probability score.
    """

    name: str = "MeanReversionAgent"
    timeframe: Timeframe = Timeframe.SHORT

    def get_required_features(self) -> list:
        """Return list of required feature names."""
        return [
            'bb_percent_b', 'keltner_width', 'rsi_14', 'vwap_distance',
            'atr_14', 'realized_vol'
        ]

    async def analyze(self, features: TechnicalFeatures, context: Dict[str, Any] = None) -> AgentOutput:
        """
        Analyze mean reversion potential using various indicators.

        Args:
            features: Technical analysis features
            context: Additional context

        Returns:
            AgentOutput with mean reversion analysis
        """

        # Calculate BB extension score
        bb_score = self._calculate_bb_extension(features)

        # Calculate Keltner extension score
        keltner_score = self._calculate_keltner_extension(features)

        # Calculate RSI extreme score
        rsi_score = self._calculate_rsi_extremes(features)

        # Calculate VWAP distance score
        vwap_score = self._calculate_vwap_distance(features)

        # Calculate volatility adjustment
        vol_adjustment = self._calculate_volatility_adjustment(features)

        # Combine scores for final mean reversion score
        dir_score = self._combine_reversion_scores(bb_score, keltner_score, rsi_score, vwap_score, vol_adjustment)

        # Calculate confidence
        confidence = self.calculate_confidence(features, context)

        # Generate rationale
        rationale = self._generate_rationale(bb_score, keltner_score, rsi_score, vwap_score, vol_adjustment, dir_score)

        # Prepare evidence
        evidence = {
            'bb_score': bb_score,
            'keltner_score': keltner_score,
            'rsi_score': rsi_score,
            'vwap_score': vwap_score,
            'vol_adjustment': vol_adjustment,
            'bb_percent_b': features.bb_percent_b,
            'keltner_width': features.keltner_width,
            'rsi_14': features.rsi_14,
            'vwap_distance': features.vwap_distance,
            'atr_14': features.atr_14,
            'realized_vol': features.realized_vol
        }

        return AgentOutput(
            timeframe=self.timeframe,
            dir_score=dir_score,
            conf=confidence,
            rationale=rationale,
            evidence=evidence
        )

    def _calculate_bb_extension(self, features: TechnicalFeatures) -> float:
        """Calculate mean reversion score based on Bollinger Band position."""

        bb_percent = features.bb_percent_b

        # BB extension analysis
        if bb_percent > 0.95:  # Extreme upper extension
            bb_score = -0.9
        elif bb_percent > 0.85:  # Upper extension
            bb_score = -0.6
        elif bb_percent < 0.05:  # Extreme lower extension
            bb_score = 0.9
        elif bb_percent < 0.15:  # Lower extension
            bb_score = 0.6
        elif 0.4 < bb_percent < 0.6:  # Middle range - no reversion signal
            bb_score = 0.0
        else:  # Moderate extension
            bb_score = 0.0

        return np.clip(bb_score, -1.0, 1.0)

    def _calculate_keltner_extension(self, features: TechnicalFeatures) -> float:
        """Calculate mean reversion score based on Keltner Channel position.

        Keltner bands = EMA(20) ± 2×ATR(14).  vwap_distance is already in ATR
        units (how many ATRs price is above/below VWAP ≈ EMA).  Dividing by 2.0
        normalises so that ±1.0 = at the upper/lower Keltner band; beyond that
        is a powerful mean-reversion setup.
        """
        # Use the actual BB/KC width ratio — the true Keltner signal.
        # bb_width / keltner_width:
        #   > 1.05 → BBands wider than KC → price broke outside channel → extension
        #   0.80–1.05 → normal expansion → mild signal
        #   < 0.80 → squeeze (BB inside KC) → no directional edge yet
        # Sign: extension on the VWAP side is the reversion direction.
        ratio = features.bb_width / max(features.keltner_width, 1e-9)
        vwap_sign = 1.0 if features.vwap_distance > 0 else -1.0

        if ratio > 1.05:
            # Price has broken outside the Keltner channel — extended, fade expected
            extension = min((ratio - 1.05) / 0.50, 1.0)
            score = -vwap_sign * float(np.clip(0.30 + 0.50 * extension, 0.0, 0.85))
        elif ratio < 0.80:
            # Squeeze: energy coiling, direction unknown — wait
            score = 0.0
        else:
            # Mild expansion: weak reversion lean toward the KC centre
            score = -vwap_sign * 0.10

        return float(np.clip(score, -1.0, 1.0))

    def _calculate_rsi_extremes(self, features: TechnicalFeatures) -> float:
        """Calculate mean reversion score based on RSI extremes."""

        rsi = features.rsi_14

        # RSI extreme analysis
        if rsi > 80:  # Extreme overbought
            rsi_score = -0.8
        elif rsi > 70:  # Overbought
            rsi_score = -0.4
        elif rsi < 20:  # Extreme oversold
            rsi_score = 0.8
        elif rsi < 30:  # Oversold
            rsi_score = 0.4
        else:  # Neutral
            rsi_score = 0.0

        return np.clip(rsi_score, -1.0, 1.0)

    def _calculate_vwap_distance(self, features: TechnicalFeatures) -> float:
        """Calculate mean reversion score based on VWAP distance."""

        vwap_dist = features.vwap_distance

        # VWAP distance analysis.
        # vwap_distance is already in ATR units (±1 = 1 ATR from VWAP).
        # Mean-reversion logic: price far ABOVE VWAP → expect DOWN → negative score.
        # Negate the distance and use 1.0 divisor so ±1 ATR ≈ tanh(±1) ≈ ±0.76.
        vwap_score = -np.tanh(vwap_dist)  # sign inverted: above VWAP = bearish signal

        return vwap_score

    def _calculate_volatility_adjustment(self, features: TechnicalFeatures) -> float:
        """Calculate volatility adjustment for mean reversion signals.

        realized_vol is annualised (e.g. 0.08 = 8% pa).
        High annual vol (> 20%) reduces mean reversion probability because
        price can stay extended longer in trending/volatile regimes.
        """
        realized_vol = features.realized_vol

        if realized_vol > 0.20:   # high annual vol: indices/gold in volatile periods
            vol_adjustment = -0.3
        elif realized_vol < 0.05: # very low annual vol: quiet FX sessions
            vol_adjustment = 0.2
        else:                     # normal
            vol_adjustment = 0.0

        return vol_adjustment

    def _combine_reversion_scores(self, bb_score: float, keltner_score: float,
                                 rsi_score: float, vwap_score: float, 
                                 vol_adjustment: float) -> float:
        """Combine individual reversion scores into final score."""

        # Weighted combination of indicator scores
        raw = (
            bb_score * 0.4 +
            rsi_score * 0.3 +
            vwap_score * 0.2 +
            keltner_score * 0.1
        )

        # vol_adjustment (±0.3 or 0) scales how much we trust the reversion signal.
        # High vol (indices) → *0.85 (reversion less reliable).
        # Very low vol (quiet FX) → *1.10 (reversion very reliable).
        # Using a multiplier rather than an additive offset prevents pushing the
        # score away from zero in instruments that are not extended at all.
        combined_score = raw * (1.0 + vol_adjustment * 0.50)

        return float(np.clip(combined_score, -1.0, 1.0))

    def _generate_rationale(self, bb_score: float, keltner_score: float,
                           rsi_score: float, vwap_score: float, 
                           vol_adjustment: float, dir_score: float) -> str:
        """Generate human-readable rationale for the mean reversion analysis."""

        # BB description
        if bb_score < -0.5:
            bb_desc = "extreme upper BB extension"
        elif bb_score < -0.2:
            bb_desc = "upper BB extension"
        elif bb_score > 0.5:
            bb_desc = "extreme lower BB extension"
        elif bb_score > 0.2:
            bb_desc = "lower BB extension"
        else:
            bb_desc = "neutral BB position"

        # RSI description
        if rsi_score < -0.5:
            rsi_desc = "extreme overbought RSI"
        elif rsi_score < -0.2:
            rsi_desc = "overbought RSI"
        elif rsi_score > 0.5:
            rsi_desc = "extreme oversold RSI"
        elif rsi_score > 0.2:
            rsi_desc = "oversold RSI"
        else:
            rsi_desc = "neutral RSI"

        # Overall reversion signal
        if dir_score > 0.5:
            reversion = "strong bullish reversion signal"
        elif dir_score > 0.2:
            reversion = "moderate bullish reversion signal"
        elif dir_score < -0.5:
            reversion = "strong bearish reversion signal"
        elif dir_score < -0.2:
            reversion = "moderate bearish reversion signal"
        else:
            reversion = "no clear reversion signal"

        rationale = (
            f"BB shows {bb_desc}, RSI indicates {rsi_desc}. "
            f"Overall mean reversion analysis shows {reversion} "
            f"(BB: {bb_score:.2f}, RSI: {rsi_score:.2f}, VWAP: {vwap_score:.2f}, Vol: {vol_adjustment:.2f})"
        )

        return rationale

    def calculate_confidence(self, features: TechnicalFeatures, context: Dict[str, Any] = None) -> float:
        """Calculate confidence score for mean reversion analysis."""

        # Start with base confidence
        confidence = 0.5

        # Higher confidence with extreme BB positions
        if features.bb_percent_b > 0.9 or features.bb_percent_b < 0.1:
            confidence += 0.2

        # Higher confidence with extreme RSI values
        if features.rsi_14 > 75 or features.rsi_14 < 25:
            confidence += 0.2

        # Higher confidence with clear VWAP distance
        if abs(features.vwap_distance) > 0.02:
            confidence += 0.1

        # Lower confidence in high volatility
        if features.realized_vol > 0.03:
            confidence -= 0.1

        # Ensure confidence is within bounds
        return min(max(confidence, 0.0), 1.0) 