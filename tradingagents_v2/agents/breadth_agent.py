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

        # Advance/decline analysis.
        # advance_decline is stored as (up_days/20) - 0.5, i.e. centred at 0.
        # Positive = more up-days, negative = more down-days.
        # Thresholds are offset by 0.5 relative to the old [0,1] convention.
        if ad_ratio > 0.10:   # Strong advance (>60% up-days)
            ad_score = 0.8
        elif ad_ratio > 0.05:  # Moderate advance (>55% up-days)
            ad_score = 0.4
        elif ad_ratio < -0.10: # Strong decline (<40% up-days)
            ad_score = -0.8
        elif ad_ratio < -0.05: # Moderate decline (<45% up-days)
            ad_score = -0.4
        else:  # Neutral
            ad_score = 0.0

        return np.clip(ad_score, -1.0, 1.0)

    def _calculate_ema_breadth(self, features: TechnicalFeatures) -> float:
        """Calculate score based on percentage of stocks above EMAs."""

        above_50 = features.above_50ema_pct
        above_200 = features.above_200ema_pct

        # EMA breadth analysis.
        # above_50ema_pct / above_200ema_pct are stored as (close - ema) / ema,
        # already centred at 0 (positive = price above EMA).  Do NOT subtract 0.5
        # (that assumed a [0,1] range and would make every value appear bearish).
        ema_50_score = np.tanh(above_50 * 10)   # ±1% relative → tanh(±0.1)≈±0.10
        ema_200_score = np.tanh(above_200 * 10)

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
        # Index trend and advance/decline get highest weights
        combined_score = (
            index_score * 0.35 +
            ad_score * 0.35 +
            ema_score * 0.2 +
            sector_score * 0.1
        )

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

        # Start with base confidence
        confidence = 0.5

        # Higher confidence with clear index trend
        if abs(features.index_trend) > 0.3:
            confidence += 0.2

        # Higher confidence with clear advance/decline
        if abs(features.advance_decline - 0.5) > 0.1:
            confidence += 0.2

        # Higher confidence with clear EMA breadth
        if abs(features.above_50ema_pct - 0.5) > 0.15:
            confidence += 0.1

        # Ensure confidence is within bounds
        return min(max(confidence, 0.0), 1.0) 