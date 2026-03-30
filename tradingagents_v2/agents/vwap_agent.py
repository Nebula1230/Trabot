"""
VWAP Scalp Agent — intraday VWAP-based bias for 1-minute index scalping.

Why VWAP matters on indices
---------------------------
Institutional desks benchmark against VWAP.  Every major index (US30, US500,
USTEC, DAX) respects VWAP throughout the regular session; price oscillates
around it, and sustained breaks of VWAP are high-conviction directional moves.

This agent translates the pre-computed `vwap_distance` (already in
TechnicalFeatures, in ATR units) into three distinct scalp regimes:

  REVERSION regime (|vwap_distance| > 2.0 ATR away)
    Price is stretched; odds of VWAP reversion > continuation.
    → Fade the current direction (counter-trend signal).
    Confidence scales with stretch distance and RSI exhaustion.

  BREAKOUT regime (|vwap_distance| 0.5–2.0 ATR, moving away)
    Price has left VWAP and is trending; ride the breakout.
    → Bias in direction of move.
    Confidence requires corroborating slope (EMA20 slope).

  CONSOLIDATION / MAGNET regime (|vwap_distance| < 0.5 ATR)
    Price is at VWAP — neutral; no edge.
    → dir_score ≈ 0, low confidence.

Signal semantics
----------------
  dir_score ∈ [-1, +1]  (positive = long bias)
  conf      ∈ [0,  1]
"""

import numpy as np
from typing import Dict, Any

from ..core.agent_base import BaseAgent
from ..core.types import AgentOutput, TechnicalFeatures, Timeframe


class VwapScalpAgent(BaseAgent):
    """
    VWAP-based intraday bias agent for 1-minute index scalping.

    Operates on the SHORT timeframe so it is counted in the scalp-profile
    fusion alongside ScalpingAgent and SqueezeBreakoutAgent.
    """

    name: str = "VwapScalpAgent"
    timeframe: Timeframe = Timeframe.SHORT

    # Regime thresholds (in ATR units, matching how vwap_distance is stored)
    _REVERSION_THRESHOLD: float = 2.0   # stretched ≥ 2×ATR from VWAP → fade
    _BREAKOUT_MIN: float = 0.5          # ≥ 0.5×ATR from VWAP → directional bias
    _RSI_OB: float = 65.0
    _RSI_OS: float = 35.0

    def get_required_features(self) -> list:
        return ["vwap_distance", "rsi_14", "rsi_4", "ema20_slope", "atr_14", "adx_14"]

    async def analyze(
        self,
        features: TechnicalFeatures,
        context: Dict[str, Any] = None,
    ) -> AgentOutput:

        # Guard against NaN/Inf in critical floats
        _critical = [features.vwap_distance, features.rsi_4, features.rsi_14,
                     features.ema20_slope, features.atr_14, features.adx_14]
        if any(not np.isfinite(v) for v in _critical):
            return AgentOutput(
                timeframe=self.timeframe, dir_score=0.0, conf=0.1,
                rationale="Insufficient data (NaN detected)", evidence={},
            )

        vd = features.vwap_distance       # positive = above VWAP, negative = below
        rsi = features.rsi_4
        rsi14 = features.rsi_14
        slope = features.ema20_slope
        atr = max(features.atr_14, 1e-9)
        adx = features.adx_14

        # Normalise slope to [-1, 1]
        slope_norm = float(np.clip(slope / (atr * 0.2), -1.0, 1.0))

        evidence: Dict[str, Any] = {
            "vwap_distance_atr": round(vd, 3),
            "rsi_4": round(rsi, 1),
            "rsi_14": round(rsi14, 1),
            "ema20_slope_norm": round(slope_norm, 3),
            "adx_14": round(adx, 1),
        }

        abs_vd = abs(vd)
        direction_sign = 1.0 if vd > 0 else -1.0   # which side of VWAP we're on

        # ── REVERSION: stretched far from VWAP ─────────────────────────
        if abs_vd >= self._REVERSION_THRESHOLD:
            # Counter-directional signal: price too far from VWAP, expect snap-back.
            # Strengthen signal when RSI also confirms exhaustion.
            stretch_factor = min(abs_vd / (self._REVERSION_THRESHOLD * 2), 1.0)
            rsi_exhaustion = 0.0
            if direction_sign > 0 and rsi > self._RSI_OB:
                rsi_exhaustion = (rsi - self._RSI_OB) / (100.0 - self._RSI_OB)
            elif direction_sign < 0 and rsi < self._RSI_OS:
                rsi_exhaustion = (self._RSI_OS - rsi) / self._RSI_OS

            # Fade: vote against current direction
            raw_score = -direction_sign * (0.5 + 0.3 * stretch_factor + 0.2 * rsi_exhaustion)
            conf = float(np.clip(0.55 + 0.35 * stretch_factor + 0.10 * rsi_exhaustion, 0.0, 1.0))
            # Strong trend = price can stay stretched longer → dampen reversion
            if adx > 30:
                trend_dampen = min((adx - 30) / 30.0, 0.5)  # up to 50% reduction
                raw_score *= (1.0 - trend_dampen)
                conf *= (1.0 - trend_dampen * 0.5)
            regime = "reversion"

        # ── BREAKOUT: away from VWAP but not exhausted ──────────────────
        elif abs_vd >= self._BREAKOUT_MIN:
            # Directional signal: price is trending away from VWAP.
            # Require slope agreement for confidence; without it, score is weak.
            slope_agrees = (direction_sign > 0 and slope_norm > 0) or \
                           (direction_sign < 0 and slope_norm < 0)
            rsi_not_exhausted = not (
                (direction_sign > 0 and rsi > self._RSI_OB) or
                (direction_sign < 0 and rsi < self._RSI_OS)
            )

            move_factor = min((abs_vd - self._BREAKOUT_MIN) /
                              (self._REVERSION_THRESHOLD - self._BREAKOUT_MIN), 1.0)
            raw_score = direction_sign * (0.3 + 0.4 * move_factor)
            if not slope_agrees:
                raw_score *= 0.4   # conflicting slope = weak signal
            if not rsi_not_exhausted:
                raw_score *= 0.5   # RSI extreme in this direction = caution

            conf_base = 0.40 + 0.30 * move_factor
            if slope_agrees:
                conf_base += 0.15
            if rsi_not_exhausted:
                conf_base += 0.10
            # ADX confirms trend → higher conviction breakout
            if adx > 25:
                conf_base += min((adx - 25) / 50.0, 0.10)
            conf = float(np.clip(conf_base, 0.0, 1.0))
            regime = "breakout"

        # ── CONSOLIDATION: sitting on VWAP ──────────────────────────────
        else:
            # No edge — price is in the VWAP magnet zone.
            raw_score = 0.0
            conf = 0.15    # very low confidence to avoid polluting the vote
            regime = "consolidation"

        dir_score = float(np.clip(raw_score, -1.0, 1.0))
        evidence["regime"] = regime
        evidence["dir_score"] = round(dir_score, 3)

        self.logger.debug(
            f"[CALC] VwapScalpAgent regime={regime} vd={vd:+.3f}ATR "
            f"→ dir={dir_score:+.4f} conf={conf:.3f} | "
            f"rsi4={rsi:.1f} slope_n={slope_norm:+.3f} adx={adx:.1f}"
        )

        direction_word = "LONG" if dir_score > 0.05 else ("SHORT" if dir_score < -0.05 else "FLAT")
        rationale = (
            f"VWAP-scalp {direction_word} [{regime}] | "
            f"dist={vd:+.2f}ATR RSI4={rsi:.0f} slope={slope_norm:+.2f}"
        )

        return AgentOutput(
            timeframe=Timeframe.SHORT,
            dir_score=dir_score,
            conf=conf,
            rationale=rationale,
            evidence=evidence,
        )
