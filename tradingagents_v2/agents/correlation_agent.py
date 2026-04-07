"""
CorrelationAgent — scores cross-symbol correlation divergences.

Tier: LONG (D1-scale relationships).

Signals consumed (pre-computed by DataLoader and stored on TechnicalFeatures):
  corr_dxy_divergence  — rolling-window divergence from DXY relationship
  corr_pair_divergence — divergence from most-correlated FX peer
  corr_risk_divergence — divergence from risk-appetite proxy (VIX-inverse)

Interpretation:
  Positive divergence = the symbol is STRONGER than its correlations imply
    → bullish bias (momentum continuation or snap-back of the correlated asset).
  Negative divergence = the symbol is WEAKER than expected
    → bearish bias.

The agent is a confirmation/context signal — it rarely overrides price structure
but adds conviction when price action aligns with cross-market evidence.
"""

from typing import Dict, Any
import numpy as np

from ..core.agent_base import BaseAgent
from ..core.types import AgentOutput, TechnicalFeatures, Timeframe


class CorrelationAgent(BaseAgent):
    name: str = "CorrelationAgent"
    timeframe: Timeframe = Timeframe.LONG

    # Component weights (sum to 1.0)
    _W_DXY:  float = 0.45   # DXY correlation divergence (strongest macro driver)
    _W_PAIR: float = 0.30   # peer pair divergence
    _W_RISK: float = 0.25   # risk-appetite divergence

    def get_required_features(self) -> list:
        return [
            "corr_dxy_divergence",
            "corr_pair_divergence",
            "corr_risk_divergence",
        ]

    async def analyze(
        self,
        features: TechnicalFeatures,
        context: Dict[str, Any] = None,
    ) -> AgentOutput:
        d = features.corr_dxy_divergence
        p = features.corr_pair_divergence
        r = features.corr_risk_divergence

        # Guard: all zeros → neutral (no correlation data available)
        if abs(d) < 1e-6 and abs(p) < 1e-6 and abs(r) < 1e-6:
            return AgentOutput(
                timeframe=self.timeframe,
                dir_score=0.0,
                conf=0.1,
                rationale="No correlation data available",
                evidence={"dxy_div": d, "pair_div": p, "risk_div": r},
            )

        # Weighted blend → raw score in [-1, 1] (already clipped by loader)
        raw = d * self._W_DXY + p * self._W_PAIR + r * self._W_RISK
        # Guard NaN from incomplete correlation data
        if not np.isfinite(raw):
            return AgentOutput(
                timeframe=self.timeframe,
                dir_score=0.0,
                conf=0.1,
                rationale="Correlation data contains NaN — returning neutral",
                evidence={"dxy_div": d, "pair_div": p, "risk_div": r},
            )
        # tanh compression keeps score well-bounded
        dir_score = float(np.tanh(raw * 1.8))

        # ── Confidence ──────────────────────────────────────────────────
        active = [s for s in (d, p, r) if abs(s) > 0.05]
        if not active:
            conf = 0.20
        else:
            # Agreement: how many active signals share the same direction?
            pos = sum(1 for s in active if s > 0)
            agreement = max(pos, len(active) - pos) / len(active)
            magnitude = float(np.mean([abs(s) for s in active]))
            conf = 0.30 + agreement * 0.30 + magnitude * 0.25

            # Single-source dampening: one signal alone is weak evidence
            if len(active) == 1:
                conf *= 0.60
                dir_score *= 0.65

            conf = float(np.clip(conf, 0.15, 0.85))

        return AgentOutput(
            timeframe=self.timeframe,
            dir_score=dir_score,
            conf=conf,
            rationale=(
                f"Corr divergence: DXY={d:+.3f} pair={p:+.3f} risk={r:+.3f} "
                f"→ dir={dir_score:+.3f} conf={conf:.2f}"
            ),
            evidence={
                "dxy_div": d,
                "pair_div": p,
                "risk_div": r,
                "raw_blend": raw,
            },
        )
