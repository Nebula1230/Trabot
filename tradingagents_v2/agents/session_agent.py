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
        return ["session_break_score"]

    async def analyze(
        self, features: TechnicalFeatures, context: Dict[str, Any] = None
    ) -> AgentOutput:

        score = features.session_break_score   # pre-computed in loader, already in [-1, 1]

        dir_score = float(np.clip(score, -1.0, 1.0))

        # Confidence scales with breakout magnitude; low when near boundary or inside range
        magnitude = abs(dir_score)
        if magnitude < 0.05:
            confidence = 0.30   # inside range / no signal
        elif magnitude < 0.20:
            confidence = 0.45   # near range boundary / weak signal
        else:
            confidence = float(np.clip(0.45 + magnitude * 0.45, 0.45, 0.90))

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

        return AgentOutput(
            timeframe=self.timeframe,
            dir_score=dir_score,
            conf=confidence,
            rationale=rationale,
            evidence={"session_break_score": score},
        )
