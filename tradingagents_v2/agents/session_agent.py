"""
Session Breakout Agent — Asian range breakout confirmation for forex / index CFDs.

Forex markets consolidate during the Asian session (00:00–08:00 UTC) in a
well-defined range. The London open (08:00 UTC) typically produces a directional
move that breaks out of that range. This is one of the most reliable and
widely-traded intraday patterns in spot FX.

The score is pre-computed by DataLoader._session_break_score() using H1 bars
and stored in TechnicalFeatures.session_break_score:

  +1  →  strong bullish break above Asian high during London/NY session
  -1  →  strong bearish break below Asian low during London/NY session
   0  →  price inside range, or signal not valid (Asian session forming)

The agent applies this directly with a confidence that accounts for:
  - Magnitude of the breakout (how far beyond the range)
  - Session timing (London > NY > Asian)
"""

import numpy as np
from typing import Dict, Any

from ..core.agent_base import BaseAgent
from ..core.types import AgentOutput, TechnicalFeatures, Timeframe


class SessionBreakoutAgent(BaseAgent):
    """
    Session Breakout Agent: Asian-range breakout confirmation.

    Timeframe: MID (uses H1 bars; London/NY intraday timing).
    Output:  directional score based on price relative to the Asian session range.
    """

    name: str = "SessionBreakoutAgent"
    timeframe: Timeframe = Timeframe.MID

    def get_required_features(self) -> list:
        return ["session_break_score", "adx_14", "ema20_slope", "atr_14"]

    async def analyze(
        self, features: TechnicalFeatures, context: Dict[str, Any] = None
    ) -> AgentOutput:

        score = features.session_break_score   # pre-computed in loader, already in [-1, 1]
        adx   = features.adx_14
        slope = features.ema20_slope
        atr   = max(features.atr_14, 1e-9)

        # Guard against NaN/Inf
        if any(not np.isfinite(v) for v in [score, adx, slope, features.atr_14]):
            return AgentOutput(
                timeframe=self.timeframe, dir_score=0.0, conf=0.1,
                rationale="Insufficient data (NaN detected)", evidence={},
            )

        dir_score = float(np.clip(score, -1.0, 1.0))
        slope_norm = float(np.clip(slope / (atr * 0.2), -1.0, 1.0))

        # Confidence scales with breakout magnitude; low when near boundary or inside range
        magnitude = abs(dir_score)
        if magnitude < 0.05:
            confidence = 0.30   # inside range / no signal
        elif magnitude < 0.20:
            confidence = 0.45   # near range boundary / weak signal
        else:
            confidence = float(np.clip(0.45 + magnitude * 0.45, 0.45, 0.90))

        # Corroboration: ADX confirms trend strength at breakout
        if magnitude > 0.05:
            if adx > 25:
                confidence = min(1.0, confidence + 0.10)
            elif adx < 15:
                confidence *= 0.70   # weak trend = suspect breakout
                dir_score  *= 0.70

            # Slope agreement: slope should point same way as breakout
            # Use current (possibly dampened) dir_score for slope check
            slope_agrees = (dir_score > 0 and slope_norm > 0.1) or \
                           (dir_score < 0 and slope_norm < -0.1)
            if slope_agrees:
                confidence = min(1.0, confidence + 0.05)
            elif (dir_score > 0 and slope_norm < -0.2) or \
                 (dir_score < 0 and slope_norm > 0.2):
                # Slope actively disagrees
                dir_score  *= 0.6
                confidence *= 0.75

        confidence = float(np.clip(confidence, 0.25, 1.0))

        # Rationale
        if dir_score > 0.35:
            label = f"above Asian high (bullish break, score={dir_score:+.2f})"
        elif dir_score < -0.35:
            label = f"below Asian low (bearish break, score={dir_score:+.2f})"
        elif abs(dir_score) < 0.08:
            label = "inside Asian range (no breakout)"
        else:
            label = f"near Asian range boundary (weak, score={dir_score:+.2f})"

        rationale = f"Session: {label}"

        self.logger.debug(
            f"[CALC] SessionBreakout break_sc={score:+.3f} slope_n={slope_norm:+.3f} "
            f"adx={adx:.1f} \u2192 dir={dir_score:+.4f} conf={confidence:.3f}"
        )

        return AgentOutput(
            timeframe=self.timeframe,
            dir_score=dir_score,
            conf=confidence,
            rationale=rationale,
            evidence={"session_break_score": score, "adx_14": adx, "slope_norm": round(slope_norm, 3)},
        )
