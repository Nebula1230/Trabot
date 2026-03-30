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
            'atr_price_ratio', 'roc_10'
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

        # NaN guard: if any critical feature is NaN/Inf, return neutral
        _critical = [
            features.atr_14, features.atr_5, features.realized_vol,
            features.bb_width, features.keltner_width,
            features.atr_price_ratio, features.roc_10,
        ]
        if any(not np.isfinite(v) for v in _critical):
            self.logger.debug("[CALC] VolatilityAgent NaN/Inf in features — returning neutral")
            return AgentOutput(
                timeframe=self.timeframe, dir_score=0.0, conf=0.1,
                rationale="Insufficient data (NaN detected)", evidence={},
            )

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

        self.logger.debug(
            f"[CALC] VolatilityAgent atr_sc={atr_score:+.3f} rv_sc={realized_vol_score:+.3f} "
            f"bw_sc={band_width_score:+.3f} reg_sc={regime_score:+.3f} "
            f"→ dir={dir_score:+.4f} conf={confidence:.3f} | "
            f"atr14={features.atr_14:.6f} atr5={features.atr_5:.6f} "
            f"rv={features.realized_vol:.5f} roc10={features.roc_10:.5f}"
        )

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

    def _apr_floor(self, features: TechnicalFeatures) -> float:
        """Adaptive atr_price_ratio floor.

        FX 1m bars have atr_price_ratio ~0.0003; a fixed 0.001 floor would
        be 3× the real value and crush all directional scoring.  Use 50% of
        the actual ratio (min 1e-6) so the floor only guards against
        degenerate near-zero values, not normal FX levels.
        """
        return max(features.atr_price_ratio * 0.5, 1e-6)

    def _calculate_atr_volatility(self, features: TechnicalFeatures) -> float:
        """Directional ATR expansion signal.

        ATR5/ATR14 ratio measures volatility momentum.  We combine expansion
        with price direction using continuous scoring via tanh, not binary
        sign gating, so weak expansions produce proportionally weak signals.
        """
        atr_14 = features.atr_14
        atr_5  = features.atr_5
        roc    = features.roc_10

        if atr_14 == 0:
            return 0.0

        expansion_ratio = (atr_5 - atr_14) / atr_14
        # Continuous price direction: tanh(roc / atr_price_ratio) → [-1, +1]
        apr = self._apr_floor(features)
        price_dir = float(np.tanh(roc / apr))

        # Expansion multiplier: expanding vol amplifies, contracting dampens
        if expansion_ratio > 0:
            vol_mult = min(0.3 + expansion_ratio * 3.0, 0.8)
        else:
            vol_mult = max(0.1, 0.3 + expansion_ratio * 2.0)

        return float(np.clip(price_dir * vol_mult, -1.0, 1.0))

    def _calculate_realized_volatility(self, features: TechnicalFeatures) -> float:
        """Realized volatility regime modifier.

        Uses continuous scoring instead of binary sign gating.
        Elevated vol + clear direction = confirming move.
        Very high vol dampens (noise), very low vol dampens (no energy).
        """
        realized_vol = features.realized_vol
        roc = features.roc_10
        apr = self._apr_floor(features)
        price_dir = float(np.tanh(roc / apr))

        # Bell-curve vol suitability: best in 0.06–0.15 annual range
        if 0.06 <= realized_vol <= 0.15:
            vol_mult = 0.5   # sweet spot
        elif 0.04 <= realized_vol < 0.06 or 0.15 < realized_vol <= 0.25:
            vol_mult = 0.35  # acceptable
        elif realized_vol > 0.25:
            vol_mult = 0.10  # too noisy
        else:
            vol_mult = 0.20  # too quiet

        return float(np.clip(price_dir * vol_mult, -1.0, 1.0))

    def _calculate_band_width(self, features: TechnicalFeatures) -> float:
        """Band-width squeeze / expansion directional signal.

        BB-width vs Keltner-width ratio indicates squeeze state.
        Direction confirmed by continuous ROC scoring.
        """
        bb_width = features.bb_width
        keltner_width = features.keltner_width
        roc = features.roc_10
        apr = self._apr_floor(features)
        price_dir = float(np.tanh(roc / apr))

        if bb_width > 0 and keltner_width > 0:
            ratio = bb_width / keltner_width
            if ratio > 1.2:   # BB wider than KC — expansion / breakout
                band_score = price_dir * 0.5
            elif ratio < 0.8: # BB inside KC — squeeze, pre-breakout
                band_score = price_dir * 0.2
            else:
                band_score = price_dir * 0.3
        else:
            band_score = 0.0

        return float(np.clip(band_score, -1.0, 1.0))

    def _calculate_volatility_regime(self, features: TechnicalFeatures) -> float:
        """Volatility regime suitability score.

        For trend-following: moderate-to-high vol in a trending regime is ideal.
        Uses continuous direction scoring for smooth output.
        """
        atr_ratio    = features.atr_price_ratio
        realized_vol = features.realized_vol
        roc = features.roc_10
        apr = self._apr_floor(features)
        price_dir = float(np.tanh(roc / apr))

        # Suitability scales with vol and trend quality.
        # Adaptive thresholds: FX (atr_ratio ~0.0003) needs much lower
        # cutoffs than indices/crypto (~0.01).
        _hi = 0.002 if atr_ratio < 0.005 else 0.005
        _md = 0.001 if atr_ratio < 0.005 else 0.003
        _lo = 0.0003 if atr_ratio < 0.005 else 0.001

        if atr_ratio > _hi and realized_vol > 0.06:
            suitability = 0.5
        elif atr_ratio > _md and realized_vol > 0.03:
            suitability = 0.3
        elif atr_ratio > _lo:
            suitability = 0.15
        else:
            suitability = 0.0

        return float(np.clip(price_dir * suitability, -1.0, 1.0))

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
        import math
        confidence = 0.40

        # Proportional ATR/price ratio boost.
        # Adaptive divisor: FX (apr ~0.0003) vs indices/crypto (apr ~0.01).
        apr = features.atr_price_ratio
        _apr_div = 0.001 if apr < 0.005 else 0.003
        confidence += math.tanh(apr / _apr_div) * 0.20

        # Proportional realized vol boost (forex typical: 0.03–0.25)
        rv = features.realized_vol
        confidence += min(rv / 0.20, 1.0) * 0.15

        # Band width clarity (BB or KC expansion → clearer signal).
        # Adaptive divisor: FX bb_width ~0.001, indices ~0.03.
        bw = max(features.bb_width, features.keltner_width)
        _bw_div = 0.003 if bw < 0.01 else 0.01
        confidence += min(bw / _bw_div, 1.0) * 0.10

        # Ensure confidence is within bounds
        return min(max(confidence, 0.0), 1.0) 