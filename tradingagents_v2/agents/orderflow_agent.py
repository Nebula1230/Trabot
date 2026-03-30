"""
Order Flow Agent — volume-weighted directional pressure for scalping.

The core idea
-------------
On 1-minute bars, not all price moves are equal.  A bar that closes near its
high on *expanding* volume reflects genuine buying pressure — institutional
orders being filled.  A bar that drifts up on shrinking volume is just noise.

This agent uses a "candle body commitment" proxy because tick-level order
flow data is not available through MT5's bar API.  The proxy combines:

  1. Close position in bar range  ≈  bb_percent_b  (already in features)
     - Near 1.0 = close near high = buyers dominated that bar
     - Near 0.0 = close near low  = sellers dominated

  2. Rate of change (roc_10)  ≈  directional velocity
     - Magnitude + direction of recent price momentum

  3. Realized volatility expansion  (realized_vol  relative to atr_14)
     - Vol expanding in direction of move = real conviction
     - Vol contracting despite price move = stop-hunt / noise

  4. VWAP confirmation  (vwap_distance)
     - Buying pressure above VWAP with expanding vol = institutional sponsorship
     - Buying pressure below VWAP = counter-trend (lower quality)

Signal semantics
----------------
  dir_score ∈ [-1, +1]   positive = buying pressure dominant
  conf      ∈ [0,  1]    scales with how many components agree
"""

import numpy as np
from typing import Dict, Any

from ..core.agent_base import BaseAgent
from ..core.types import AgentOutput, TechnicalFeatures, Timeframe


class OrderFlowAgent(BaseAgent):
    """
    Order flow proxy agent for 1-minute scalping.

    Combines bar close-position, rate-of-change, volatility expansion, and
    VWAP alignment to assess whether real directional pressure is behind a
    1m price move.
    """

    name: str = "OrderFlowAgent"
    timeframe: Timeframe = Timeframe.SHORT

    # Thresholds
    _BODY_BULL: float = 0.70   # close in top 30% of range = buyer-dominated bar
    _BODY_BEAR: float = 0.30   # close in bottom 30% = seller-dominated bar
    _VOL_EXPAND: float = 1.20  # realized_vol / (atr_14 / sqrt(14)) > 1.2 = expanding
    _VOL_SHRINK: float = 0.80

    def get_required_features(self) -> list:
        return ["bb_percent_b", "roc_10", "realized_vol", "atr_14",
                "vwap_distance", "macd_hist", "rsi_14"]

    async def analyze(
        self,
        features: TechnicalFeatures,
        context: Dict[str, Any] = None,
    ) -> AgentOutput:

        # Guard against NaN/Inf in critical floats
        _critical = [features.bb_percent_b, features.roc_10, features.realized_vol,
                     features.atr_14, features.vwap_distance, features.macd_hist,
                     features.rsi_14]
        if any(not np.isfinite(v) for v in _critical):
            return AgentOutput(
                timeframe=self.timeframe, dir_score=0.0, conf=0.1,
                rationale="Insufficient data (NaN detected)", evidence={},
            )

        bb_b  = features.bb_percent_b   # close position in rolling BB range
        roc   = features.roc_10         # velocity
        rvol  = features.realized_vol   # realised vol (annualised pct)
        atr   = max(features.atr_14, 1e-9)
        vd    = features.vwap_distance  # ATR-normalised distance from VWAP
        macd  = features.macd_hist
        rsi   = features.rsi_14

        # Volatility expansion ratio: compare current-window realised vol to ATR.
        # Both must be in the same units (per-bar absolute move, not annualised).
        # realized_vol from features is correctly annualised (e.g. 0.08 = 8% pa).
        # De-annualise to per-bar by dividing by sqrt(bars_per_year) which is
        # stored on the feature object by the DataLoader.
        # atr_price_ratio = atr_14 / close, also per-bar fractional.
        # We then compare apples-to-apples: per-bar fractional vol vs per-bar ATR.
        bpy = max(features.bars_per_year, 1.0)   # guard against 0
        per_bar_rvol = rvol / (bpy ** 0.5)   # de-annualised to per-bar
        atr_price_ratio = features.atr_price_ratio if features.atr_price_ratio > 0 else 0.001
        vol_ratio = per_bar_rvol / max(atr_price_ratio, 1e-9)

        scores: list[float] = []
        evidence: Dict[str, Any] = {}

        # ── 1. Bar close-position (body commitment) ─────────────────────
        if bb_b >= self._BODY_BULL:
            body_score = 0.8    # buyers clearly in control
        elif bb_b <= self._BODY_BEAR:
            body_score = -0.8   # sellers clearly in control
        elif bb_b > 0.55:
            body_score = 0.35
        elif bb_b < 0.45:
            body_score = -0.35
        else:
            body_score = 0.0    # neutral body
        evidence["bb_percent_b"] = round(bb_b, 3)
        evidence["body_score"] = round(body_score, 2)
        scores.append(body_score)

        # ── 2. Rate of change — directional velocity ────────────────────
        # roc_10 is (close/close_10barsago - 1) × 100
        # ATR-normalised: 2× ATR-price-ratio is significant on any timeframe
        _atr_ratio = max(features.atr_price_ratio, 0.001)
        roc_norm = float(np.clip(roc / (_atr_ratio * 2.0), -1.0, 1.0))
        evidence["roc_10"] = round(roc, 4)
        evidence["roc_norm"] = round(roc_norm, 3)
        scores.append(roc_norm * 0.7)   # slightly down-weighted vs body score

        # ── 3. Volatility expansion in direction of move ─────────────────
        # If vol is expanding AND move is directional → conviction behind it.
        # If vol contracts despite a move → low commitment (noise).
        roc_dir = np.sign(roc) if abs(roc) > 1e-6 else 0.0
        if vol_ratio >= self._VOL_EXPAND:
            # Vol expanding: amplify the directional score
            vol_score = roc_dir * 0.7
        elif vol_ratio <= self._VOL_SHRINK:
            # Vol shrinking: dampen — move might be low-conviction
            vol_score = roc_dir * 0.1
        else:
            vol_score = roc_dir * 0.35   # neutral expansion
        evidence["vol_ratio"] = round(float(vol_ratio), 2)
        evidence["vol_score"] = round(float(vol_score), 2)
        scores.append(float(vol_score))

        # ── 4. VWAP alignment — institutional sponsorship ───────────────
        # Buying pressure ABOVE VWAP = with institutional flow.
        # Buying pressure below VWAP = counter-trend (lower quality).
        vwap_dir = np.sign(vd) if abs(vd) > 0.1 else 0.0   # which side of VWAP
        if roc_dir == vwap_dir and vwap_dir != 0:
            # Move is aligned with VWAP position — higher quality
            vwap_score = roc_dir * min(abs(vd) / 2.0, 0.6)
        elif abs(vd) > 2.0:
            # Very far from VWAP — any pressure here is extension, not flow
            vwap_score = -roc_dir * 0.3   # slight penalty for over-extension
        else:
            vwap_score = 0.0
        evidence["vwap_distance"] = round(vd, 3)
        evidence["vwap_score"] = round(float(vwap_score), 3)
        scores.append(float(vwap_score))

        # ── 5. MACD histogram as order flow accelerator ─────────────────
        macd_dir = np.sign(macd) if abs(macd) > 1e-9 else 0.0
        macd_score = float(macd_dir) * 0.3
        evidence["macd_hist_dir"] = float(macd_dir)
        scores.append(macd_score)

        # ── 6. RSI-based flow momentum ─────────────────────────────────
        # RSI away from midpoint confirms directional pressure.
        if rsi > 60:
            rsi_flow = min((rsi - 50) / 30, 0.5)
        elif rsi < 40:
            rsi_flow = max((rsi - 50) / 30, -0.5)
        else:
            rsi_flow = 0.0
        evidence["rsi_flow"] = round(rsi_flow, 3)
        scores.append(rsi_flow)

        # ── 7. Aggregate ────────────────────────────────────────────────
        # Weighted: body commitment and vol expansion are most informative.
        _w = [0.25, 0.15, 0.22, 0.15, 0.10, 0.13]  # body, roc, vol, vwap, macd, rsi
        dir_score = float(np.clip(
            sum(s * w for s, w in zip(scores, _w)) / sum(_w),
            -1.0, 1.0
        ))

        # ── 8. Confidence ───────────────────────────────────────────────
        # High when: body + roc + vol all agree AND VWAP aligned
        agreement = 1.0 - float(np.std(scores)) / 0.6
        agreement = float(np.clip(agreement, 0.1, 1.0))

        # Bonus for vol expansion
        if vol_ratio >= self._VOL_EXPAND:
            agreement = min(1.0, agreement * 1.15)

        # Penalty if RSI is in extreme zone opposite the signal
        if dir_score > 0 and rsi > 70:
            agreement *= 0.7   # buying pressure but overbought
        elif dir_score < 0 and rsi < 30:
            agreement *= 0.7

        conf = float(np.clip(agreement * 0.80, 0.0, 1.0))

        self.logger.debug(
            f"[CALC] OrderFlowAgent body={body_score:+.3f} roc_n={roc_norm:+.3f} "
            f"vol={float(vol_score):+.3f} vwap={float(vwap_score):+.3f} "
            f"macd={macd_score:+.3f} rsi_f={rsi_flow:+.3f} "
            f"→ dir={dir_score:+.4f} conf={conf:.3f} | vol_ratio={float(vol_ratio):.2f}"
        )

        direction_word = "LONG" if dir_score > 0.05 else ("SHORT" if dir_score < -0.05 else "FLAT")
        rationale = (
            f"OrderFlow {direction_word} | "
            f"body={bb_b:.2f} ROC={roc:+.3f} vol_r={vol_ratio:.1f} "
            f"vwap={vd:+.2f}ATR MACD={'↑' if macd > 0 else '↓'}"
        )

        return AgentOutput(
            timeframe=Timeframe.SHORT,
            dir_score=dir_score,
            conf=conf,
            rationale=rationale,
            evidence=evidence,
        )
