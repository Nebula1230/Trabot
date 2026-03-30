"""
Squeeze Breakout Agent — Bollinger Band / Keltner Channel squeeze detection.

The Squeeze concept (John Carter / TTM Squeeze)
------------------------------------------------
A Bollinger Band "squeeze" occurs when the BB narrows inside the Keltner
Channel.  It signals a period of extremely low volatility — the market is
coiling energy.  When the BB expands back outside the Keltner Channel, the
coiled energy is released as a directional breakout.

On 1-minute index CFDs (US30, US500, USTEC, DAX), squeezes resolve fast —
often within 3–10 bars — making them ideal scalp setups.

What this agent does
--------------------
TechnicalFeatures already provides:
  • bb_width       — current Bollinger Band width (in price units, 2σ)
  • keltner_width  — current Keltner Channel width (ATR-based)
  • roc_10         — rate of change (direction of recent move)
  • macd_hist      — histogram (momentum direction at breakout)
  • adx_14         — trend strength (confirms breakout vs fake)
  • ema20_slope    — direction filter

Three states:
  SQUEEZE_ACTIVE (bb_width < keltner_width × 0.85)
    → Coiling.  No entry yet.  dir_score ≈ 0, low conf.

  SQUEEZE_FIRE   (bb_width ≥ keltner_width × 0.85, previously tight)
    → Energy releasing.  Strong breakout signal in direction of momentum.
    Approximated here as: bb_width is close to or above keltner_width
    AND roc_10 is directional AND macd_hist confirms direction.

  NO_SQUEEZE     (bb_width ≥ keltner_width × 1.2)
    → Wide bands — trend already expanded.  Weaker continuation signal.
    The breakout has already happened; chase only if ADX confirms.

Signal semantics
----------------
  dir_score ∈ [-1, +1]
  conf      ∈ [0,  1]
"""

import numpy as np
from typing import Dict, Any, Optional

from ..core.agent_base import BaseAgent
from ..core.types import AgentOutput, TechnicalFeatures, Timeframe


class SqueezeBreakoutAgent(BaseAgent):
    """
    Bollinger Band / Keltner squeeze detector for 1-minute scalp breakouts.

    bb_width < keltner_width  → bands squeezed inside channel = energy coiling.
    bb_width ≥ keltner_width  → bands firing = breakout in progress.
    Direction confirmed by MACD histogram + ROC.

    State memory: only emits high-confidence SQUEEZE_FIRE on the *transition*
    bar out of a squeeze (previous bar was squeeze_active AND current bar has
    ratio ≥ threshold).  Normal BB expansion without a prior squeeze is scored
    as ``expansion_normal`` with lower confidence.
    This prevents spurious high-confidence signals on every normally-spread bar.
    """

    name: str = "SqueezeBreakoutAgent"
    timeframe: Timeframe = Timeframe.SHORT

    # Squeeze / breakout thresholds
    _SQUEEZE_RATIO: float = 0.85   # BB must be < 85% of Keltner width = coiling
    _BREAKOUT_RATIO: float = 1.20  # BB > 120% of Keltner = strong expansion
    _ADX_TREND: float = 20.0       # ADX above this = trending (confirms breakout)

    def model_post_init(self, __context: Any) -> None:
        # Mutable state: tracks whether the previous bar was in squeeze_active.
        # Used by the state machine: SQUEEZE_FIRE is only emitted on the
        # *transition* bar (prev_squeeze_active=True → current ratio >= threshold)
        # to avoid re-emitting the same breakout signal on every subsequent bar.
        object.__setattr__(self, "_prev_squeeze_active", False)

    def get_required_features(self) -> list:
        return ["bb_width", "keltner_width", "roc_10", "macd_hist",
                "macd_hist_delta", "adx_14", "ema20_slope", "atr_14"]

    async def analyze(
        self,
        features: TechnicalFeatures,
        context: Dict[str, Any] = None,
    ) -> AgentOutput:

        # Guard against NaN/Inf in critical floats
        _critical = [features.bb_width, features.keltner_width, features.roc_10,
                     features.macd_hist, features.macd_hist_delta, features.adx_14,
                     features.ema20_slope, features.atr_14]
        if any(not np.isfinite(v) for v in _critical):
            return AgentOutput(
                timeframe=self.timeframe, dir_score=0.0, conf=0.1,
                rationale="Insufficient data (NaN detected)", evidence={},
            )

        bb_w = features.bb_width
        kc_w = max(features.keltner_width, 1e-9)
        ratio = bb_w / kc_w          # > 1 = expanding, < 1 = squeezed

        roc = features.roc_10        # positive = upward momentum, negative = down
        macd = features.macd_hist
        macd_d = features.macd_hist_delta
        adx = features.adx_14
        slope = features.ema20_slope
        atr = max(features.atr_14, 1e-9)
        slope_norm = float(np.clip(slope / (atr * 0.2), -1.0, 1.0))

        evidence: Dict[str, Any] = {
            "bb_kc_ratio": round(ratio, 3),
            "roc_10": round(roc, 4),
            "macd_hist": round(macd, 6),
            "adx_14": round(adx, 1),
        }

        # ── Determine momentum direction ────────────────────────────────
        # ATR-normalised ROC for instrument independence
        roc_norm = float(np.clip(roc / (atr * 0.5), -3.0, 3.0))  # ±3 ATR units
        roc_dir  = np.sign(roc_norm) if abs(roc_norm) > 0.05 else 0.0
        macd_dir = np.sign(macd) if abs(macd) > 1e-9 else 0.0
        macd_accel = np.sign(macd_d) if abs(macd_d) > 1e-9 else 0.0

        # Direction strength: continuous via tanh of normalised magnitude
        roc_strength = float(np.tanh(abs(roc_norm) * 1.5))  # 0-1 smooth

        if roc_dir == macd_dir and macd_dir != 0:
            direction = roc_dir
            direction_strength = min(1.0, roc_strength * 0.7 + 0.3)  # agreement floor 0.3
        elif roc_dir != 0:
            direction = roc_dir
            direction_strength = roc_strength * 0.5   # only ROC votes
        else:
            direction = 0.0
            direction_strength = 0.0

        # MACD acceleration bonus (histogram still moving in direction)
        if macd_accel == direction:
            direction_strength = min(1.0, direction_strength + 0.2)

        # Slope confirmation
        if slope_norm * direction > 0:
            direction_strength = min(1.0, direction_strength + 0.15)

        evidence["direction"] = float(direction)
        evidence["direction_strength"] = round(direction_strength, 2)

        # ── Squeeze state machine ───────────────────────────────────────
        prev_squeeze_active: bool = object.__getattribute__(self, "_prev_squeeze_active")

        if ratio < self._SQUEEZE_RATIO:
            # --- SQUEEZE ACTIVE: coiling, no entry ---
            # Still report direction so other agents can anticipate direction,
            # but with very low confidence (we don't trade the squeeze itself).
            dir_score = float(direction * direction_strength * 0.2)
            conf = 0.15
            regime = "squeeze_active"
            # Update state: still in squeeze
            object.__setattr__(self, "_prev_squeeze_active", True)

        elif ratio < self._BREAKOUT_RATIO:
            # --- TRANSITION / EARLY FIRE: BB reaching Keltner width ---
            # Only emit with high confidence on the *first* transition bar
            # (when previous bar was in squeeze_active).  Subsequent bars in
            # this zone that were never squeezed are normal expansion — lower conf.
            object.__setattr__(self, "_prev_squeeze_active", False)
            if prev_squeeze_active:
                # True squeeze-fire transition: maximum conviction
                dir_score = float(direction * direction_strength * 0.85)
                adx_bonus = min((adx - self._ADX_TREND) / 30.0, 0.15) if adx > self._ADX_TREND else 0.0
                conf = float(np.clip(0.55 + 0.25 * direction_strength + adx_bonus, 0.0, 1.0))
                regime = "squeeze_fire"
            else:
                # Normal BB expansion (no preceding squeeze) — moderate signal
                dir_score = float(direction * direction_strength * 0.50)
                adx_bonus = min((adx - self._ADX_TREND) / 30.0, 0.10) if adx > self._ADX_TREND else 0.0
                conf = float(np.clip(0.35 + 0.15 * direction_strength + adx_bonus, 0.0, 1.0))
                regime = "expansion_normal"

        else:
            # --- BREAKOUT EXPANDED: BB already wide ---
            # Breakout is underway; entry is still valid but we're chasing.
            # Reduce confidence proportional to how extended the expansion is.
            object.__setattr__(self, "_prev_squeeze_active", False)
            extension = min((ratio - self._BREAKOUT_RATIO) / 0.5, 1.0)
            dir_score = float(direction * direction_strength * (0.7 - 0.3 * extension))
            conf = float(np.clip(0.40 - 0.20 * extension + 0.15 * direction_strength, 0.0, 1.0))
            # Further dampened if ADX is weak (false breakout territory)
            if adx < self._ADX_TREND:
                dir_score *= 0.5
                conf *= 0.7
            regime = "breakout_expanded"

        # Clamp
        dir_score = float(np.clip(dir_score, -1.0, 1.0))
        evidence["regime"] = regime

        self.logger.debug(
            f"[CALC] SqueezeBreakout regime={regime} ratio={ratio:.3f} "
            f"dir_str={direction_strength:.3f} → dir={dir_score:+.4f} conf={conf:.3f} | "
            f"roc_n={roc_norm:+.3f} adx={adx:.1f} prev_sq={prev_squeeze_active}"
        )

        direction_word = "LONG" if dir_score > 0.05 else ("SHORT" if dir_score < -0.05 else "FLAT")
        rationale = (
            f"Squeeze {direction_word} [{regime}] | "
            f"BB/KC={ratio:.2f} ROC={roc:+.3f} MACD={'↑' if macd > 0 else '↓'} ADX={adx:.0f}"
        )

        return AgentOutput(
            timeframe=Timeframe.SHORT,
            dir_score=dir_score,
            conf=conf,
            rationale=rationale,
            evidence=evidence,
        )
