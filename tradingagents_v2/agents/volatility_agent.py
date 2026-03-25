"""
Volatility Agent - Risk-based entry sizing and volatility analysis.
"""

import numpy as np
from typing import Dict, Any

from ..core.agent_base import BaseAgent
from ..core.types import AgentOutput, TechnicalFeatures, Timeframe


class VolatilityAgent(BaseAgent):
    """
    Volatility Agent: Risk-based entry sizing and volatility analysis.

    Goal: Risk-based entry sizing and volatility analysis.
    Features: ATR, realized vol, BB width, Keltner width, volatility regime.
    Signal: Volatility-adjusted risk score.
    """

    name: str = "VolatilityAgent"
    timeframe: Timeframe = Timeframe.SHORT

    def get_required_features(self) -> list:
        """Return list of required feature names."""
        return [
            'atr_14', 'atr_5', 'realized_vol', 'bb_width', 'keltner_width',
            'atr_price_ratio'
        ]

    async def analyze(self, features: TechnicalFeatures, context: Dict[str, Any] = None) -> AgentOutput:
        """
        Analyze volatility for risk-based entry sizing.

        Args:
            features: Technical analysis features
            context: Additional context

        Returns:
            AgentOutput with volatility analysis
        """

        # Calculate ATR-based volatility score
        atr_score = self._calculate_atr_volatility(features)

        # Calculate realized volatility score
        realized_vol_score = self._calculate_realized_volatility(features)

        # Calculate band width score
        band_width_score = self._calculate_band_width(features)

        # Calculate volatility regime score
        regime_score = self._calculate_volatility_regime(features)

        # Combine scores for final volatility score
        dir_score = self._combine_volatility_scores(atr_score, realized_vol_score, band_width_score, regime_score)

        # Calculate confidence
        confidence = self.calculate_confidence(features, context)

        # Generate rationale
        rationale = self._generate_rationale(atr_score, realized_vol_score, band_width_score, regime_score, dir_score)

        # Prepare evidence
        evidence = {
            'atr_score': atr_score,
            'realized_vol_score': realized_vol_score,
            'band_width_score': band_width_score,
            'regime_score': regime_score,
            'atr_14': features.atr_14,
            'atr_5': features.atr_5,
            'realized_vol': features.realized_vol,
            'bb_width': features.bb_width,
            'keltner_width': features.keltner_width,
            'atr_price_ratio': features.atr_price_ratio
        }

        return AgentOutput(
            timeframe=self.timeframe,
            dir_score=dir_score,
            conf=confidence,
            rationale=rationale,
            evidence=evidence
        )

    def _calculate_atr_volatility(self, features: TechnicalFeatures) -> float:
        """Directional ATR expansion signal.

        ATR5 > ATR14 means volatility is EXPANDING (impulse move underway).
        ATR5 < ATR14 means volatility is CONTRACTING (consolidation).
        We combine this with price direction (ROC) to get a directional score:
          expanding + price up   → bullish confirmation
          expanding + price down → bearish confirmation
          contracting            → weak signal (near-neutral)
        """
        atr_14 = features.atr_14
        atr_5  = features.atr_5
        roc    = features.roc_10  # price direction proxy

        if atr_14 == 0:
            return 0.0

        expansion_ratio = (atr_5 - atr_14) / atr_14   # positive = expanding
        price_direction = np.sign(roc) if abs(roc) > 1e-5 else 0.0

        if expansion_ratio > 0.10:       # clearly expanding
            atr_score = price_direction * 0.6
        elif expansion_ratio > 0.02:     # mildly expanding
            atr_score = price_direction * 0.3
        elif expansion_ratio < -0.15:    # clearly contracting
            atr_score = price_direction * 0.1   # choppy — dampen
        else:                            # neutral expansion
            atr_score = price_direction * 0.2

        return float(np.clip(atr_score, -1.0, 1.0))

    def _calculate_realized_volatility(self, features: TechnicalFeatures) -> float:
        """Realized volatility regime modifier.

        Very high realized vol → uncertainty, pull toward neutral.
        Very low realized vol  → regime may break out, mild directional lean.
        Normal realized vol    → pass through price direction unmodified.
        Score is always paired with price direction (ROC sign).
        """
        realized_vol = features.realized_vol
        roc = features.roc_10
        price_direction = float(np.sign(roc)) if abs(roc) > 1e-5 else 0.0

        if realized_vol > 0.04:    # Very high vol — noisy, dampen
            realized_score = price_direction * 0.1
        elif realized_vol > 0.02:  # Elevated vol — trending environment
            realized_score = price_direction * 0.5
        elif realized_vol > 0.01:  # Normal
            realized_score = price_direction * 0.4
        else:                       # Low vol — pre-breakout; mild direction
            realized_score = price_direction * 0.2

        return float(np.clip(realized_score, -1.0, 1.0))

    def _calculate_band_width(self, features: TechnicalFeatures) -> float:
        """Band-width squeeze / expansion directional signal.

        BB-width expanding above Keltner-width suggests a Bollinger squeeze
        breakout in progress — confirm direction with price ROC.
        """
        bb_width = features.bb_width
        keltner_width = features.keltner_width
        roc = features.roc_10
        price_direction = float(np.sign(roc)) if abs(roc) > 1e-5 else 0.0

        if bb_width > 0 and keltner_width > 0:
            ratio = bb_width / keltner_width
            if ratio > 1.2:   # BB wider than KC — expansion / breakout
                band_score = price_direction * 0.5
            elif ratio < 0.8: # BB inside KC — squeeze, pre-breakout
                band_score = price_direction * 0.2
            else:              # Normal
                band_score = price_direction * 0.3
        else:
            band_score = 0.0

        return float(np.clip(band_score, -1.0, 1.0))

    def _calculate_volatility_regime(self, features: TechnicalFeatures) -> float:
        """Volatility regime suitability score.

        For trend-following: we want moderate-to-high vol, trending regime.
        Score reflects how favourable the vol environment is for the
        current price direction (ROC).
        """
        atr_ratio    = features.atr_price_ratio
        realized_vol = features.realized_vol
        roc = features.roc_10
        price_direction = float(np.sign(roc)) if abs(roc) > 1e-5 else 0.0

        # Elevated & rising vol = good for trend-following entry
        if atr_ratio > 0.005 and realized_vol > 0.01:
            suitability = 0.5
        elif atr_ratio > 0.002 and realized_vol > 0.005:
            suitability = 0.2
        else:   # Very low vol — poor trending conditions
            suitability = 0.0

        return float(np.clip(price_direction * suitability, -1.0, 1.0))

    def _combine_volatility_scores(self, atr_score: float, realized_vol_score: float,
                                  band_width_score: float, regime_score: float) -> float:
        """Combine individual volatility scores into final score."""

        # Weighted combination
        # ATR and realized vol get highest weights as they're most reliable
        combined_score = (
            atr_score * 0.4 +
            realized_vol_score * 0.3 +
            regime_score * 0.2 +
            band_width_score * 0.1
        )

        return np.clip(combined_score, -1.0, 1.0)

    def _generate_rationale(self, atr_score: float, realized_vol_score: float,
                           band_width_score: float, regime_score: float, dir_score: float) -> str:
        """Generate human-readable rationale for the volatility analysis."""

        # ATR description
        if atr_score > 0.5:
            atr_desc = "high ATR volatility"
        elif atr_score > 0.2:
            atr_desc = "moderate ATR volatility"
        elif atr_score < -0.5:
            atr_desc = "low ATR volatility"
        elif atr_score < -0.2:
            atr_desc = "moderate-low ATR volatility"
        else:
            atr_desc = "normal ATR volatility"

        # Realized volatility description
        if realized_vol_score > 0.5:
            realized_desc = "high realized volatility"
        elif realized_vol_score > 0.2:
            realized_desc = "moderate realized volatility"
        elif realized_vol_score < -0.5:
            realized_desc = "low realized volatility"
        elif realized_vol_score < -0.2:
            realized_desc = "moderate-low realized volatility"
        else:
            realized_desc = "normal realized volatility"

        # Overall volatility
        if dir_score > 0.5:
            volatility = "high volatility regime"
        elif dir_score > 0.2:
            volatility = "moderate-high volatility regime"
        elif dir_score < -0.5:
            volatility = "low volatility regime"
        elif dir_score < -0.2:
            volatility = "moderate-low volatility regime"
        else:
            volatility = "normal volatility regime"

        rationale = (
            f"ATR shows {atr_desc}, realized volatility indicates {realized_desc}. "
            f"Overall market is in a {volatility} "
            f"(ATR: {atr_score:.2f}, Realized: {realized_vol_score:.2f}, "
            f"Bands: {band_width_score:.2f}, Regime: {regime_score:.2f})"
        )

        return rationale

    def calculate_confidence(self, features: TechnicalFeatures, context: Dict[str, Any] = None) -> float:
        """Calculate confidence score for volatility analysis."""

        # Start with base confidence
        confidence = 0.5

        # Higher confidence with clear volatility signals
        if features.atr_price_ratio > 0.02:
            confidence += 0.2

        # Higher confidence with clear realized volatility
        if features.realized_vol > 0.015:
            confidence += 0.2

        # Higher confidence with clear band widths
        if features.bb_width > 0.05 or features.keltner_width > 0.04:
            confidence += 0.1

        # Ensure confidence is within bounds
        return min(max(confidence, 0.0), 1.0) 