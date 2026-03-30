"""
Breadth Agent - Market context and breadth analysis.
"""

import numpy as np
from typing import Dict, Any

from ..core.agent_base import BaseAgent
from ..core.types import AgentOutput, TechnicalFeatures, Timeframe


class BreadthAgent(BaseAgent):
    """
    Breadth Agent: Market context and breadth analysis.

    Goal: Market context and breadth analysis.
    Features: Index trend, advance/decline, % above EMAs, sector direction.
    Signal: Market breadth score.
    """

    name: str = "BreadthAgent"
    timeframe: Timeframe = Timeframe.LONG

    def get_required_features(self) -> list:
        """Return list of required feature names."""
        return [
            'index_trend', 'advance_decline', 'above_50ema_pct', 'above_200ema_pct', 'sector_direction'
        ]

    async def analyze(self, features: TechnicalFeatures, context: Dict[str, Any] = None) -> AgentOutput:
        """
        Analyze market breadth and context.

        Args:
            features: Technical analysis features
            context: Additional context

        Returns:
            AgentOutput with breadth analysis
        """

        # NaN guard: if any critical feature is NaN/Inf, return neutral
        _critical = [
            features.index_trend, features.advance_decline,
            features.above_50ema_pct, features.above_200ema_pct,
            features.sector_direction,
        ]
        if any(not np.isfinite(v) for v in _critical):
            self.logger.debug("[CALC] BreadthAgent NaN/Inf in features — returning neutral")
            return AgentOutput(
                timeframe=self.timeframe, dir_score=0.0, conf=0.1,
                rationale="Insufficient data (NaN detected)", evidence={},
            )

        # Calculate index trend score
        index_score = self._calculate_index_trend(features)

        # Calculate advance/decline score
        ad_score = self._calculate_advance_decline(features)

        # Calculate EMA breadth score
        ema_score = self._calculate_ema_breadth(features)

        # Calculate sector direction score
        sector_score = self._calculate_sector_direction(features)

        # Combine scores for final breadth score
        dir_score = self._combine_breadth_scores(index_score, ad_score, ema_score, sector_score)

        # Calculate confidence
        confidence = self.calculate_confidence(features, context)

        self.logger.debug(
            f"[CALC] BreadthAgent idx={index_score:+.3f} ad={ad_score:+.3f} "
            f"ema={ema_score:+.3f} sec={sector_score:+.3f} "
            f"\u2192 dir={dir_score:+.4f} conf={confidence:.3f}"
        )

        # Generate rationale
        rationale = self._generate_rationale(index_score, ad_score, ema_score, sector_score, dir_score)

        # Prepare evidence
        evidence = {
            'index_score': index_score,
            'ad_score': ad_score,
            'ema_score': ema_score,
            'sector_score': sector_score,
            'index_trend': features.index_trend,
            'advance_decline': features.advance_decline,
            'above_50ema_pct': features.above_50ema_pct,
            'above_200ema_pct': features.above_200ema_pct,
            'sector_direction': features.sector_direction
        }

        return AgentOutput(
            timeframe=self.timeframe,
            dir_score=dir_score,
            conf=confidence,
            rationale=rationale,
            evidence=evidence
        )

    def _calculate_index_trend(self, features: TechnicalFeatures) -> float:
        """Calculate score based on index trend."""

        index_trend = features.index_trend

        # Index trend analysis (normalize to [-1, 1])
        # Assuming index_trend is already in [-1, 1] range
        index_score = np.clip(index_trend, -1.0, 1.0)

        return index_score

    def _calculate_advance_decline(self, features: TechnicalFeatures) -> float:
        """Calculate score based on advance/decline ratio."""

        ad_ratio = features.advance_decline

        # Advance/decline: centred at 0.  Continuous tanh mapping.
        # yfinance range: [-1, 1] (RSP-SPY ratio).  MT5 fallback: [-0.5, 0.5].
        # Use ×3 so moderate values (±0.3) map to ±0.71 without early saturation.
        ad_score = float(np.tanh(ad_ratio * 3.0))

        return np.clip(ad_score, -1.0, 1.0)

    def _calculate_ema_breadth(self, features: TechnicalFeatures) -> float:
        """Calculate score based on percentage of stocks above EMAs."""

        above_50 = features.above_50ema_pct
        above_200 = features.above_200ema_pct

        # EMA breadth analysis.
        # above_50ema_pct / above_200ema_pct are stored as (close - ema) / ema,
        # already centred at 0 (positive = price above EMA).
        # Real values: ±0.001–0.05 for FX, ±0.01–0.10 for indices.
        # Use ×5 so a 2% distance (±0.02) maps to tanh(0.1) ≈ 0.10,
        # and a 10% distance (±0.10) maps to tanh(0.5) ≈ 0.46.
        ema_50_score = np.tanh(above_50 * 5)
        ema_200_score = np.tanh(above_200 * 5)

        # Combine EMA scores
        ema_score = (ema_50_score + ema_200_score) / 2.0

        return np.clip(ema_score, -1.0, 1.0)

    def _calculate_sector_direction(self, features: TechnicalFeatures) -> float:
        """Calculate score based on sector direction."""

        sector_dir = features.sector_direction

        # Sector direction analysis (normalize to [-1, 1])
        # Assuming sector_direction is already in [-1, 1] range
        sector_score = np.clip(sector_dir, -1.0, 1.0)

        return sector_score

    def _combine_breadth_scores(self, index_score: float, ad_score: float,
                               ema_score: float, sector_score: float) -> float:
        """Combine individual breadth scores into final score."""

        # Weighted combination
        combined_score = (
            index_score * 0.35 +
            ad_score * 0.35 +
            ema_score * 0.2 +
            sector_score * 0.1
        )

        # Breadth divergence: index bullish but A/D + EMA bearish → dampen
        breadth_avg = (ad_score + ema_score) / 2.0
        if index_score > 0.2 and breadth_avg < -0.1:
            combined_score *= 0.5   # bullish price, bearish breadth → halve
        elif index_score < -0.2 and breadth_avg > 0.1:
            combined_score *= 0.5   # bearish price, bullish breadth → halve

        return np.clip(combined_score, -1.0, 1.0)

    def _generate_rationale(self, index_score: float, ad_score: float,
                           ema_score: float, sector_score: float, dir_score: float) -> str:
        """Generate human-readable rationale for the breadth analysis."""

        # Index trend description
        if index_score > 0.5:
            index_desc = "strong bullish index trend"
        elif index_score > 0.2:
            index_desc = "moderate bullish index trend"
        elif index_score < -0.5:
            index_desc = "strong bearish index trend"
        elif index_score < -0.2:
            index_desc = "moderate bearish index trend"
        else:
            index_desc = "neutral index trend"

        # Advance/decline description
        if ad_score > 0.5:
            ad_desc = "strong advance/decline ratio"
        elif ad_score > 0.2:
            ad_desc = "moderate advance/decline ratio"
        elif ad_score < -0.5:
            ad_desc = "strong decline/advance ratio"
        elif ad_score < -0.2:
            ad_desc = "moderate decline/advance ratio"
        else:
            ad_desc = "neutral advance/decline ratio"

        # Overall breadth
        if dir_score > 0.5:
            breadth = "strong bullish market breadth"
        elif dir_score > 0.2:
            breadth = "moderate bullish market breadth"
        elif dir_score < -0.5:
            breadth = "strong bearish market breadth"
        elif dir_score < -0.2:
            breadth = "moderate bearish market breadth"
        else:
            breadth = "neutral market breadth"

        rationale = (
            f"Index shows {index_desc}, advance/decline indicates {ad_desc}. "
            f"Overall market breadth is {breadth} "
            f"(Index: {index_score:.2f}, A/D: {ad_score:.2f}, "
            f"EMA: {ema_score:.2f}, Sector: {sector_score:.2f})"
        )

        return rationale

    def calculate_confidence(self, features: TechnicalFeatures, context: Dict[str, Any] = None) -> float:
        """Calculate confidence score for breadth analysis."""
        confidence = 0.45

        # Proportional boost from index trend clarity (0.0–0.2)
        idx = abs(features.index_trend)
        confidence += min(idx / 0.8, 1.0) * 0.20

        # Proportional boost from advance/decline magnitude (0.0–0.15)
        ad = abs(features.advance_decline)
        confidence += min(ad / 0.5, 1.0) * 0.15

        # Proportional boost from EMA breadth (0.0–0.10)
        ema_b = abs(features.above_50ema_pct)
        confidence += min(ema_b / 0.15, 1.0) * 0.10

        return min(max(confidence, 0.0), 0.95) 