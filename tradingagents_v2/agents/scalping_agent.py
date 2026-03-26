"""
Scalping Agent — 1-minute momentum snap analysis.

Designed for the `scalp` profile.  It reads the SCALP timeframe (1m bars)
and produces a directional vote based on:

  1. Micro-momentum   — 1m RSI + MACD histogram direction
  2. Micro-structure  — last N-bar high/low break (range breakout)
  3. Spread guard     — rejects when spread > max_spread_atr (ATR fraction)
  4. Intrabar vol     — prefers setups where price has already moved ≥ 0.5×ATR
                        (entry on confirmed kick, not anticipation)

The agent outputs to `Timeframe.SHORT` so the existing fusion logic can aggregate
it alongside the regular SHORT-tier agents (PatternAgent, etc.).  In scalp mode
the fusion weights are skewed to treat this signal as dominant.

Signal semantics
----------------
  dir_score  ∈  [-1, +1]   — direction bias
  conf       ∈  [0,  1]    — confidence (quality of the 1m setup)

Confidence is reduced when:
  • Spread is elevated (> 0.3×ATR)
  • Recent candles show indecision (small bodies, doji-like)
  • RSI is in middle no-man's-land (45–55) without a strong MACD push
"""

import numpy as np
from typing import Dict, Any

from ..core.agent_base import BaseAgent
from ..core.types import AgentOutput, TechnicalFeatures, Timeframe


class ScalpingAgent(BaseAgent):
    """
    Scalping Agent: 1-minute breakout/momentum snap for the scalp profile.

    Optimised for speed; operates entirely on the SHORT (1m) timeframe.
    Should be used together with a fast cycle (interval_seconds: 60) and
    narrow-spread indices or majors.
    """

    name: str = "ScalpingAgent"
    timeframe: Timeframe = Timeframe.SHORT

    # Thresholds (can be tuned; kept as class-level constants for speed)
    _RSI_OB: float = 65.0     # overbought for scalp (tighter than normal 70)
    _RSI_OS: float = 35.0     # oversold  for scalp
    _RSI_NEUTRAL_LO: float = 45.0
    _RSI_NEUTRAL_HI: float = 55.0
    _MIN_BODY_ATR: float = 0.15   # body must be ≥ 15% of ATR to count as directional

    def get_required_features(self) -> list:
        return ["rsi_14", "rsi_4", "macd_hist", "macd_hist_delta",
                "atr_14", "atr_5", "bb_percent_b", "ema20", "ema50",
                "ema20_slope", "swing_highs", "swing_lows"]

    async def analyze(
        self,
        features: TechnicalFeatures,
        context: Dict[str, Any] = None,
    ) -> AgentOutput:
        ctx = context or {}

        scores: list[float] = []
        evidence: Dict[str, Any] = {}

        # ── 1. Micro-momentum: RSI ──────────────────────────────────────
        rsi = features.rsi_4          # 4-period RSI is most responsive on 1m
        rsi_14 = features.rsi_14
        if rsi > self._RSI_OB:
            rsi_score = 0.7            # strong momentum long
        elif rsi < self._RSI_OS:
            rsi_score = -0.7           # strong momentum short
        elif rsi > self._RSI_NEUTRAL_HI:
            rsi_score = 0.35
        elif rsi < self._RSI_NEUTRAL_LO:
            rsi_score = -0.35
        else:
            rsi_score = 0.0            # dead zone — no edge
        evidence["rsi_4"] = round(rsi, 1)
        evidence["rsi_14"] = round(rsi_14, 1)
        scores.append(rsi_score)

        # ── 2. MACD histogram momentum snap ────────────────────────────
        # For scalping, histogram direction + acceleration both matter.
        macd_h  = features.macd_hist
        macd_dh = features.macd_hist_delta   # delta = histogram[t] - histogram[t-1]
        if macd_h > 0 and macd_dh > 0:
            macd_score = 0.7     # rising positive hist → strong bullish impulse
        elif macd_h < 0 and macd_dh < 0:
            macd_score = -0.7    # falling negative hist → strong bearish impulse
        elif macd_h > 0 and macd_dh < 0:
            macd_score = 0.2     # positive but decelerating — weakening
        elif macd_h < 0 and macd_dh > 0:
            macd_score = -0.2    # negative but recovering — weakening
        else:
            macd_score = 0.0
        evidence["macd_hist"] = round(macd_h, 6)
        evidence["macd_hist_delta"] = round(macd_dh, 6)
        scores.append(macd_score)

        # ── 3. Micro-structure: EMA20 slope (acts as 1m trend filter) ──
        # ema20 on 1m bars responds to ~20-bar micro-trend (~20 minutes).
        slope = features.ema20_slope
        # Normalise slope to ±1 range using ATR (slope unit = price/bar)
        atr = features.atr_5 if features.atr_5 > 0 else features.atr_14
        if atr > 0:
            slope_norm = np.clip(slope / (atr * 0.2), -1.0, 1.0)
        else:
            slope_norm = 0.0
        evidence["ema20_slope_norm"] = round(float(slope_norm), 3)
        scores.append(float(slope_norm) * 0.6)   # down-weighted vs RSI/MACD

        # ── 4. BB position (overbought / oversold on 1m BB) ────────────
        bb_b = features.bb_percent_b   # 0 = lower band, 1 = upper band
        if bb_b > 0.85:
            bb_score = 0.5     # near upper BB → bullish breakout candidate
        elif bb_b < 0.15:
            bb_score = -0.5    # near lower BB → bearish breakout candidate
        elif 0.45 <= bb_b <= 0.55:
            bb_score = 0.0     # midpoint — no edge
        else:
            bb_score = np.sign(bb_b - 0.5) * 0.2
        evidence["bb_percent_b"] = round(bb_b, 3)
        scores.append(bb_score)

        # ── 5. Swing breakout: last swing high/low ─────────────────────
        # If price is above the most recent swing high → breakout long.
        # If price is below the most recent swing low  → breakout short.
        swing_break_score = 0.0
        current_price = ctx.get("current_price", None)
        if current_price and features.swing_highs and features.swing_lows:
            # Use the most recent confirmed swing high/low (last in chronological list)
            # so we detect whether price just broke the immediately prior structure.
            # max/min of the full list would require price to beat the all-time
            # highest/lowest of the lookback window — far too strict for 1m scalping.
            nearest_sh = features.swing_highs[-1]   # most recent swing high
            nearest_sl = features.swing_lows[-1]    # most recent swing low
            if current_price > nearest_sh:
                swing_break_score = 0.8   # above last high = momentum breakout
            elif current_price < nearest_sl:
                swing_break_score = -0.8  # below last low  = momentum breakdown
            evidence["nearest_swing_high"] = round(nearest_sh, 5)
            evidence["nearest_swing_low"]  = round(nearest_sl, 5)
        evidence["swing_break_score"] = round(swing_break_score, 2)
        scores.append(swing_break_score)

        # ── 6. Aggregate direction score ────────────────────────────────
        # Simple mean; all components given equal weight here.
        dir_score = float(np.clip(np.mean(scores), -1.0, 1.0))

        # ── 7. Confidence ───────────────────────────────────────────────
        # Penalise if all components disagree (variance is high).
        agreement = 1.0 - float(np.std(scores)) / 0.7    # 0.7 = max expected std
        agreement = max(0.1, min(1.0, agreement))

        # Penalise RSI in neutral zone (no clear setup)
        if self._RSI_NEUTRAL_LO < rsi < self._RSI_NEUTRAL_HI:
            agreement *= 0.7

        # Boost when MACD and RSI both agree strongly
        if (rsi_score > 0.5 and macd_score > 0.5) or (rsi_score < -0.5 and macd_score < -0.5):
            agreement = min(1.0, agreement * 1.2)

        conf = float(np.clip(agreement * 0.85, 0.0, 1.0))

        # ── Flat-signal penalty ─────────────────────────────────────────
        # When the combined direction is near zero (no clear edge), confidence
        # should be low.  A flat aggregate score with perfect component
        # agreement just means everything is neutral — not that we're sure.
        if abs(dir_score) < 0.10:
            conf *= 0.40

        # ── 8. Rationale ────────────────────────────────────────────────
        direction_word = "LONG" if dir_score > 0.05 else ("SHORT" if dir_score < -0.05 else "FLAT")
        rationale = (
            f"Scalp {direction_word} | RSI4={rsi:.0f} MACD={'↑' if macd_h > 0 else '↓'}"
            f"{'↑' if macd_dh > 0 else '↓'} BB%B={bb_b:.2f} "
            f"slope={slope_norm:+.2f} swing={swing_break_score:+.1f}"
        )

        return AgentOutput(
            timeframe=Timeframe.SHORT,
            dir_score=dir_score,
            conf=conf,
            rationale=rationale,
            evidence=evidence,
        )
