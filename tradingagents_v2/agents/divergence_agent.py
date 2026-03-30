"""
Divergence Agent — detects RSI divergences between price and momentum.

Regular divergences signal high-probability reversals; hidden divergences
confirm trend continuations at pullbacks. Both are among the highest win-rate
setups in technical analysis when confirmed by other signals.

Timeframe: MID (1H bars, where divergences are most actionable).
"""

import numpy as np
from typing import Dict, Any

from ..core.agent_base import BaseAgent
from ..core.types import AgentOutput, TechnicalFeatures, Timeframe


class DivergenceAgent(BaseAgent):
    """
    RSI Divergence detection agent.

    Reads pre-computed divergence scores from TechnicalFeatures
    (computed in DataLoader._divergence_score on 1H bars).

    Signal logic:
      bull_div_score > bear_div_score + margin → LONG (bullish divergence)
      bear_div_score > bull_div_score + margin → SHORT (bearish divergence)
      Neither strong enough                    → NEUTRAL (0.0)

    Confidence boost when:
      - RSI < 35 during bullish divergence  (oversold + divergence = conviction)
      - RSI > 65 during bearish divergence  (overbought + divergence = conviction)
      - MACD histogram confirms direction
    """

    name: str      = "DivergenceAgent"
    timeframe: Timeframe = Timeframe.MID

    # Minimum score to recognise a divergence at all
    _MIN_SCORE: float = 0.15
    # Edge to prefer one side (avoids acting on coin-flip situations)
    _EDGE_MARGIN: float = 0.05

    def get_required_features(self) -> list:
        return ["bull_div_score", "bear_div_score", "rsi_14", "rsi_4", "macd_hist", "macd_hist_delta"]

    async def analyze(
        self, features: TechnicalFeatures, context: Dict[str, Any] = None
    ) -> AgentOutput:

        bull = features.bull_div_score
        bear = features.bear_div_score
        rsi  = features.rsi_14
        rsi4 = features.rsi_4
        macd = features.macd_hist
        macd_d = features.macd_hist_delta

        # Guard against NaN/Inf
        _critical = [bull, bear, rsi, rsi4, macd, macd_d]
        if any(not np.isfinite(v) for v in _critical):
            return AgentOutput(
                timeframe=self.timeframe, dir_score=0.0, conf=0.1,
                rationale="Insufficient data (NaN detected)", evidence={},
            )

        # Momentum direction: MACD histogram delta is a true derivative
        # (positive = momentum accelerating up, negative = decelerating).
        # Avoids the bullish bias of comparing RSI-4 vs RSI-14 directly.
        mom_rising = macd_d > 0

        # ── Direction decision ─────────────────────────────────────────
        if bull >= self._MIN_SCORE and bull > (bear + self._EDGE_MARGIN):
            dir_score = float(np.clip(bull, 0.0, 1.0))
            sig_type  = "bullish"
        elif bear >= self._MIN_SCORE and bear > (bull + self._EDGE_MARGIN):
            dir_score = float(np.clip(-bear, -1.0, 0.0))
            sig_type  = "bearish"
        else:
            dir_score = 0.0
            sig_type  = "none"

        # ── RSI slope confirmation ─────────────────────────────────────
        # Bullish divergence + RSI now rising = confirmed reversal.
        # Bearish divergence + RSI now falling = confirmed reversal.
        # If RSI slope opposes the divergence, dampen the signal.
        if sig_type == "bullish" and not mom_rising:
            dir_score *= 0.6   # unconfirmed: momentum still falling
        elif sig_type == "bearish" and mom_rising:
            dir_score *= 0.6   # unconfirmed: momentum still rising

        # ── Confidence ───────────────────────────────────────────────────────
        conf = 0.30   # base: signal exists but worth low weight until confirmed

        if sig_type == "bullish":
            active_score = bull
            if rsi < 35:
                conf += 0.25   # oversold + divergence = high-quality reversal
            elif rsi < 45:
                conf += 0.10
            if macd > 0:       # MACD histogram turning positive agrees
                conf += 0.10
            if macd_d > 0:     # MACD histogram accelerating upward
                conf += 0.05
            if mom_rising:     # momentum slope confirms
                conf += 0.10
        elif sig_type == "bearish":
            active_score = bear
            if rsi > 65:
                conf += 0.25   # overbought + divergence = high-quality reversal
            elif rsi > 55:
                conf += 0.10
            if macd < 0:       # MACD histogram turning negative agrees
                conf += 0.10
            if macd_d < 0:     # MACD histogram accelerating downward
                conf += 0.05
            if not mom_rising: # momentum slope confirms
                conf += 0.10
        else:
            active_score = 0.0

        # Scale conf with signal strength (stronger divergence = higher conf)
        if active_score > 0:
            conf += active_score * 0.20

        conf = float(np.clip(conf, 0.0, 0.95))

        self.logger.debug(
            f"[CALC] DivergenceAgent sig={sig_type} bull={bull:.3f} bear={bear:.3f} "
            f"\u2192 dir={dir_score:+.4f} conf={conf:.3f} | "
            f"rsi14={rsi:.1f} rsi4={rsi4:.1f} macd_h={macd:.6f} mom_rising={mom_rising}"
        )

        # \u2500\u2500 Rationale \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if sig_type == "none":
            rationale = (
                f"No significant divergence (bull={bull:.2f} bear={bear:.2f} "
                f"threshold={self._MIN_SCORE})"
            )
        else:
            rationale = (
                f"RSI {sig_type} divergence: "
                f"bull={bull:.2f} bear={bear:.2f} "
                f"rsi={rsi:.1f} macd_hist={macd:.6f}"
            )

        evidence = {
            "bull_div_score":  bull,
            "bear_div_score":  bear,
            "signal_type":     sig_type,
            "rsi_14":          rsi,
            "macd_hist":       macd,
        }

        return AgentOutput(
            timeframe=self.timeframe,
            dir_score=dir_score,
            conf=conf,
            rationale=rationale,
            evidence=evidence,
        )

    def calculate_confidence(
        self, features: TechnicalFeatures, context: Dict[str, Any] = None
    ) -> float:
        """Fallback confidence (used by BaseAgent.run if analyze is not overridden)."""
        return 0.40
