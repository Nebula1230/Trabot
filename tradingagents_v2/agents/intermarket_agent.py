"""
Intermarket Agent — macro-driven directional bias for forex and index CFDs.

Uses pre-computed macro signals stored in TechnicalFeatures (populated by DataLoader
via yfinance). Each signal is already mapped to this symbol's directional bias
(positive = bullish for the pair, negative = bearish).

Macro factors:
  dxy_dir    — USD Index trend  (DX-Y.NYB 5-day momentum)
  vix_dir    — VIX fear/greed  (^VIX level + direction)
  crude_dir  — WTI crude trend (CL=F 5-day momentum)
  yield_dir  — US 10Y yield   (^TNX 5-day momentum)

Signal weights reflect each factor's typical contribution:
  DXY    0.75 — primary USD driver; affects all forex pairs
  VIX    0.65 — risk-on/risk-off; critical for AUD/NZD/JPY/indices
  Crude  0.35 — mainly relevant for USDCAD; minor for others
  Yield  0.55 — carry trade / rate differential; key for USDJPY
"""

import numpy as np
from typing import Dict, Any

from ..core.agent_base import BaseAgent
from ..core.types import AgentOutput, TechnicalFeatures, Timeframe


class IntermarketAgent(BaseAgent):
    """
    Intermarket Agent: macro context bias for forex / index CFDs.

    Timeframe: LONG (macro forces are slow-moving, D1 relevant).
    Output: directional score blending DXY, VIX, crude, and yield signals
            that have been pre-weighted per symbol by the DataLoader.
    """

    name: str = "IntermarketAgent"
    timeframe: Timeframe = Timeframe.LONG

    # Absolute importance of each macro factor in the final blend.
    # Higher = that factor gets more vote in the combined score.
    _W_DXY:   float = 0.75
    _W_VIX:   float = 0.65
    _W_CRUDE: float = 0.35
    _W_YIELD: float = 0.55

    def get_required_features(self) -> list:
        return ["dxy_dir", "vix_dir", "crude_dir", "yield_dir"]

    async def analyze(
        self, features: TechnicalFeatures, context: Dict[str, Any] = None
    ) -> AgentOutput:

        d = features.dxy_dir
        v = features.vix_dir
        c = features.crude_dir
        y = features.yield_dir

        # Weighted blend of pre-mapped macro signals
        total_w = self._W_DXY + self._W_VIX + self._W_CRUDE + self._W_YIELD
        raw = (d * self._W_DXY + v * self._W_VIX +
               c * self._W_CRUDE + y * self._W_YIELD) / total_w

        # tanh compression: keeps output in (-1, 1) and penalises extreme single-factor spikes
        dir_score = float(np.tanh(raw * 2.0))

        # Confidence: how many factors are informative (non-zero) AND agree on direction
        active  = [s for s in (d, v, c, y) if abs(s) > 0.05]
        if not active:
            confidence = 0.35   # all neutral — very low confidence
        else:
            sign_std    = float(np.std([np.sign(s) for s in active]))  # 0 = perfect agreement
            agreement   = max(0.0, 1.0 - sign_std * 0.7)
            magnitude   = float(np.mean([abs(s) for s in active]))
            confidence  = float(np.clip(0.35 + magnitude * 0.35 + agreement * 0.30, 0.35, 0.90))

        # Rationale
        parts = []
        if abs(d) > 0.10:
            parts.append(f"DXY {'↑' if d > 0 else '↓'}{abs(d):.2f}")
        if abs(v) > 0.10:
            parts.append(f"VIX {'↑' if v > 0 else '↓'}{abs(v):.2f}")
        if abs(c) > 0.10:
            parts.append(f"Crude {'↑' if c > 0 else '↓'}{abs(c):.2f}")
        if abs(y) > 0.10:
            parts.append(f"Yield {'↑' if y > 0 else '↓'}{abs(y):.2f}")
        direction_label = "bullish" if dir_score > 0.1 else "bearish" if dir_score < -0.1 else "neutral"
        rationale = (
            f"Intermarket: {direction_label} ({', '.join(parts) or 'all signals muted'})"
            f" → score {dir_score:+.2f}"
        )

        return AgentOutput(
            timeframe=self.timeframe,
            dir_score=dir_score,
            conf=confidence,
            rationale=rationale,
            evidence={
                "dxy_dir":   d,
                "vix_dir":   v,
                "crude_dir": c,
                "yield_dir": y,
                "raw_blend": float(raw),
            },
        )
