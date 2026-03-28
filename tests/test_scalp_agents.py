"""
test_scalp_agents.py — Intensive tests for the scalping agent suite.

Coverage:
  § 1  ScalpingAgent       — unit, edge cases, boundary values, directional polarity
  § 2  VwapScalpAgent      — all three regimes + edge cases
  § 3  SqueezeBreakoutAgent — squeeze_active / fire / expanded + ADX gate
  § 4  OrderFlowAgent      — body, velocity, vol expansion, VWAP alignment combos
  § 5  Cross-agent agreement — all scalp agents agree on obvious bull/bear setups
  § 6  Scalp profile config  — symbol override, interval, magic, agent weights
  § 7  Registry integration  — scalp agents registered & weighted correctly
  § 8  Graph round-trip       — TradingGraph with scalp-only registry, sim features
  § 9  Runner integration     — TradingRunner with scalp config, run_once no crash
  §10  Output contract         — all agents honour [-1,+1] dir_score, [0,1] conf
  §11  Idempotency              — same features ⇒ same output every call
  §12  Degenerate inputs        — zeros, NaN-like, extreme values never crash
"""

import asyncio
import math
import pytest
import numpy as np

from tradingagents_v2.core.types import (
    TechnicalFeatures, AgentOutput, Timeframe, Direction,
    PortfolioState, RiskLimits,
)
from tradingagents_v2.core.agent_base import AgentRegistry
from tradingagents_v2.core.graph import TradingGraph
from tradingagents_v2.execution.mt5_executor import MT5Executor

from tradingagents_v2.agents.scalping_agent import ScalpingAgent
from tradingagents_v2.agents.vwap_agent import VwapScalpAgent
from tradingagents_v2.agents.squeeze_agent import SqueezeBreakoutAgent
from tradingagents_v2.agents.orderflow_agent import OrderFlowAgent

from tradingagents_v2.config.yaml_config import load_config_from_yaml
from tradingagents_v2.runner import TradingRunner
from tradingagents_v2.config.settings import TradingConfig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run(coro):
    return asyncio.run(coro)


def make_features(**overrides) -> TechnicalFeatures:
    """Baseline 1m neutral features — override as needed per test."""
    base = dict(
        swing_highs=[100.0, 102.0, 104.0],
        swing_lows=[98.0, 99.0, 100.5],
        hh_hl_count=2, lh_ll_count=0,
        last_break="bullish",
        ema20=102.0, ema50=100.0, ema200=98.0,
        ema20_slope=0.002, ema50_slope=0.001, ema200_slope=0.0005,
        rsi_14=50.0, rsi_4=52.0,
        macd_hist=0.0, macd_hist_delta=0.0,
        roc_10=0.0, bb_percent_b=0.50,
        atr_14=1.5, atr_5=1.2,
        realized_vol=0.015, bb_width=0.04, keltner_width=0.045,
        adx_14=22.0, rvi=0.1, hurst_exponent=0.5,
        atr_price_ratio=0.015, vwap_distance=0.0,
        index_trend=0.0, advance_decline=0.0,
        above_50ema_pct=0.0, above_200ema_pct=0.0,
        sector_direction=0.0,
    )
    base.update(overrides)
    return TechnicalFeatures(**base)


def make_bull_features() -> TechnicalFeatures:
    """Unambiguously bullish 1m scalp setup."""
    return make_features(
        rsi_4=72.0, rsi_14=62.0,
        macd_hist=0.8, macd_hist_delta=0.3,
        roc_10=0.6,
        bb_percent_b=0.92,
        ema20_slope=0.01,
        vwap_distance=1.0,          # above VWAP → breakout
        bb_width=0.07,              # wider than keltner → squeeze fired
        keltner_width=0.045,
        adx_14=32.0,
        realized_vol=0.025,
        atr_price_ratio=0.015,
        swing_highs=[98.0, 100.0, 102.0],
        swing_lows=[96.0, 97.5, 99.0],
    )


def make_bear_features() -> TechnicalFeatures:
    """Unambiguously bearish 1m scalp setup."""
    return make_features(
        rsi_4=28.0, rsi_14=38.0,
        macd_hist=-0.8, macd_hist_delta=-0.3,
        roc_10=-0.6,
        bb_percent_b=0.08,
        ema20_slope=-0.01,
        vwap_distance=-1.0,         # below VWAP → breakout down
        bb_width=0.07,
        keltner_width=0.045,
        adx_14=32.0,
        realized_vol=0.025,
        atr_price_ratio=0.015,
        swing_highs=[104.0, 102.0, 100.0],
        swing_lows=[101.5, 99.5, 97.0],
    )


def make_scalp_registry() -> AgentRegistry:
    r = AgentRegistry()
    for agent in [ScalpingAgent(), VwapScalpAgent(),
                  SqueezeBreakoutAgent(), OrderFlowAgent()]:
        r.register(agent)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# § 1  ScalpingAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestScalpingAgentBasic:
    @pytest.fixture
    def agent(self):
        return ScalpingAgent()

    # ── name & timeframe ──────────────────────────────────────────────────
    def test_name(self, agent):
        assert agent.name == "ScalpingAgent"

    def test_timeframe_is_short(self, agent):
        assert agent.timeframe == Timeframe.SHORT

    # ── output contract ───────────────────────────────────────────────────
    def test_output_is_agent_output(self, agent):
        out = run(agent.analyze(make_features()))
        assert isinstance(out, AgentOutput)

    def test_dir_score_in_range(self, agent):
        out = run(agent.analyze(make_features()))
        assert -1.0 <= out.dir_score <= 1.0

    def test_conf_in_range(self, agent):
        out = run(agent.analyze(make_features()))
        assert 0.0 <= out.conf <= 1.0

    def test_rationale_is_nonempty_string(self, agent):
        out = run(agent.analyze(make_features()))
        assert isinstance(out.rationale, str) and len(out.rationale) > 0

    def test_evidence_is_dict(self, agent):
        out = run(agent.analyze(make_features()))
        assert isinstance(out.evidence, dict)

    # ── RSI direction ─────────────────────────────────────────────────────
    def test_RSI4_above_OB_gives_positive_score(self, agent):
        f = make_features(rsi_4=70.0, macd_hist=0.0, macd_hist_delta=0.0,
                          ema20_slope=0.0, bb_percent_b=0.5)
        out = run(agent.analyze(f))
        assert out.dir_score > 0

    def test_RSI4_below_OS_gives_negative_score(self, agent):
        f = make_features(rsi_4=30.0, macd_hist=0.0, macd_hist_delta=0.0,
                          ema20_slope=0.0, bb_percent_b=0.5)
        out = run(agent.analyze(f))
        assert out.dir_score < 0

    def test_RSI4_neutral_zone_reduces_confidence(self, agent):
        f_neutral = make_features(rsi_4=50.0)
        f_strong  = make_features(rsi_4=72.0, macd_hist=0.5, macd_hist_delta=0.2)
        out_n = run(agent.analyze(f_neutral))
        out_s = run(agent.analyze(f_strong))
        assert out_s.conf >= out_n.conf

    # ── MACD direction ────────────────────────────────────────────────────
    def test_rising_positive_MACD_adds_bull_score(self, agent):
        f = make_features(rsi_4=50.0, macd_hist=0.5, macd_hist_delta=0.2,
                          bb_percent_b=0.5, ema20_slope=0.0)
        out = run(agent.analyze(f))
        assert out.dir_score > 0

    def test_falling_negative_MACD_adds_bear_score(self, agent):
        f = make_features(rsi_4=50.0, macd_hist=-0.5, macd_hist_delta=-0.2,
                          bb_percent_b=0.5, ema20_slope=0.0)
        out = run(agent.analyze(f))
        assert out.dir_score < 0

    def test_decelerating_positive_MACD_weaker_than_accelerating(self, agent):
        f_accel = make_features(rsi_4=50.0, macd_hist=0.5, macd_hist_delta=+0.2,
                                 bb_percent_b=0.5, ema20_slope=0.0)
        f_decel = make_features(rsi_4=50.0, macd_hist=0.5, macd_hist_delta=-0.2,
                                 bb_percent_b=0.5, ema20_slope=0.0)
        out_a = run(agent.analyze(f_accel))
        out_d = run(agent.analyze(f_decel))
        assert out_a.dir_score > out_d.dir_score

    # ── BB %B ─────────────────────────────────────────────────────────────
    def test_upper_BB_adds_bull_contribution(self, agent):
        f_upper = make_features(bb_percent_b=0.90, rsi_4=50.0,
                                 macd_hist=0.0, macd_hist_delta=0.0, ema20_slope=0.0)
        f_lower = make_features(bb_percent_b=0.10, rsi_4=50.0,
                                 macd_hist=0.0, macd_hist_delta=0.0, ema20_slope=0.0)
        assert run(agent.analyze(f_upper)).dir_score > \
               run(agent.analyze(f_lower)).dir_score

    # ── Swing breakout via context ────────────────────────────────────────
    def test_price_above_swing_high_adds_bull_score(self, agent):
        f = make_features(swing_highs=[100.0], swing_lows=[98.0])
        ctx_above = {"current_price": 101.0}   # above swing high 100
        ctx_below = {"current_price": 99.0}    # inside range
        out_above = run(agent.analyze(f, context=ctx_above))
        out_below = run(agent.analyze(f, context=ctx_below))
        assert out_above.dir_score > out_below.dir_score

    def test_price_below_swing_low_adds_bear_score(self, agent):
        f = make_features(swing_highs=[100.0], swing_lows=[98.0])
        ctx = {"current_price": 97.0}   # below swing low 98
        out = run(agent.analyze(f, context=ctx))
        assert out.dir_score < 0

    # ── Polarity: bull setup → positive, bear setup → negative ───────────
    def test_full_bull_setup_positive(self, agent):
        out = run(agent.analyze(make_bull_features()))
        assert out.dir_score > 0.1

    def test_full_bear_setup_negative(self, agent):
        out = run(agent.analyze(make_bear_features()))
        assert out.dir_score < -0.1

    # ── MACD+RSI boost confidence ─────────────────────────────────────────
    def test_macd_rsi_agreement_boosts_confidence(self, agent):
        f_agree = make_features(rsi_4=70.0, macd_hist=0.8, macd_hist_delta=0.2)
        f_disagr = make_features(rsi_4=70.0, macd_hist=-0.8, macd_hist_delta=-0.2)
        assert run(agent.analyze(f_agree)).conf >= run(agent.analyze(f_disagr)).conf

    # ── Evidence keys ─────────────────────────────────────────────────────
    def test_evidence_contains_rsi_and_macd(self, agent):
        out = run(agent.analyze(make_features()))
        assert "rsi_4" in out.evidence
        assert "macd_hist" in out.evidence
        assert "bb_percent_b" in out.evidence


# ─────────────────────────────────────────────────────────────────────────────
# § 2  VwapScalpAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestVwapScalpAgent:
    @pytest.fixture
    def agent(self):
        return VwapScalpAgent()

    def test_name(self, agent):
        assert agent.name == "VwapScalpAgent"

    def test_timeframe_is_short(self, agent):
        assert agent.timeframe == Timeframe.SHORT

    # ── CONSOLIDATION at VWAP (<0.5 ATR) ──────────────────────────────────
    def test_near_vwap_flat_score(self, agent):
        f = make_features(vwap_distance=0.1, ema20_slope=0.0, rsi_4=50.0)
        out = run(agent.analyze(f))
        assert abs(out.dir_score) < 0.15
        assert out.conf < 0.35

    def test_on_vwap_low_confidence(self, agent):
        f = make_features(vwap_distance=0.0)
        out = run(agent.analyze(f))
        assert out.conf < 0.35

    # ── BREAKOUT (0.5–2×ATR from VWAP) ────────────────────────────────────
    def test_above_vwap_breakout_zone_positive_score(self, agent):
        # vwap_distance is in ATR units; 1.0 ATR above = breakout regime
        f = make_features(vwap_distance=1.0, ema20_slope=0.005, rsi_4=55.0)
        out = run(agent.analyze(f))
        assert out.dir_score > 0

    def test_below_vwap_breakout_zone_negative_score(self, agent):
        f = make_features(vwap_distance=-1.0, ema20_slope=-0.005, rsi_4=45.0)
        out = run(agent.analyze(f))
        assert out.dir_score < 0

    def test_breakout_slope_agreement_increases_confidence(self, agent):
        # Both above VWAP and positive slope should yield higher conf than conflicting slope
        f_agree  = make_features(vwap_distance=1.0, ema20_slope=+0.01, rsi_4=55.0)
        f_contra = make_features(vwap_distance=1.0, ema20_slope=-0.01, rsi_4=55.0)
        out_a = run(agent.analyze(f_agree))
        out_c = run(agent.analyze(f_contra))
        assert out_a.conf >= out_c.conf

    def test_breakout_rsi_exhaustion_reduces_score(self, agent):
        # Above VWAP but RSI extremely overbought → lower confidence
        f_ok  = make_features(vwap_distance=1.0, rsi_4=55.0, rsi_14=52.0, ema20_slope=0.005)
        f_ob  = make_features(vwap_distance=1.0, rsi_4=82.0, rsi_14=78.0, ema20_slope=0.005)
        out_ok = run(agent.analyze(f_ok))
        out_ob = run(agent.analyze(f_ob))
        # Overbought should produce weaker (lower or negative) score than healthy
        assert out_ok.dir_score >= out_ob.dir_score

    # ── REVERSION (≥2×ATR from VWAP) ─────────────────────────────────────
    def test_far_above_vwap_gives_bearish_reversion(self, agent):
        # 3 ATR above VWAP → fade → negative dir_score
        f = make_features(vwap_distance=3.0, rsi_4=50.0, ema20_slope=0.0)
        out = run(agent.analyze(f))
        assert out.dir_score < 0, "Should give reversion (short) signal above 2× ATR"

    def test_far_below_vwap_gives_bullish_reversion(self, agent):
        f = make_features(vwap_distance=-3.0, rsi_4=50.0, ema20_slope=0.0)
        out = run(agent.analyze(f))
        assert out.dir_score > 0, "Should give reversion (long) signal below -2×ATR"

    def test_reversion_rsi_exhaustion_strengthens_signal(self, agent):
        # Far above VWAP AND RSI overbought → stronger reversion signal
        f_plain = make_features(vwap_distance=3.0, rsi_4=50.0, ema20_slope=0.0)
        f_ob    = make_features(vwap_distance=3.0, rsi_4=82.0, rsi_14=78.0, ema20_slope=0.0)
        out_p = run(agent.analyze(f_plain))
        out_ob = run(agent.analyze(f_ob))
        # Both should be negative (bearish reversion); RSI extreme means more confidence
        assert out_p.dir_score < 0 and out_ob.dir_score < 0
        assert out_ob.conf >= out_p.conf

    def test_reversion_high_confidence(self, agent):
        # Extreme stretch → high confidence
        f = make_features(vwap_distance=4.0, rsi_4=85.0, rsi_14=80.0)
        out = run(agent.analyze(f))
        assert out.conf > 0.5

    # ── Regime label in evidence ──────────────────────────────────────────
    def test_evidence_contains_regime(self, agent):
        out = run(agent.analyze(make_features()))
        assert "regime" in out.evidence
        assert out.evidence["regime"] in ("consolidation", "breakout", "reversion")

    def test_correct_regime_near_vwap(self, agent):
        out = run(agent.analyze(make_features(vwap_distance=0.1)))
        assert out.evidence["regime"] == "consolidation"

    def test_correct_regime_breakout(self, agent):
        out = run(agent.analyze(make_features(vwap_distance=1.0)))
        assert out.evidence["regime"] == "breakout"

    def test_correct_regime_reversion(self, agent):
        out = run(agent.analyze(make_features(vwap_distance=2.5)))
        assert out.evidence["regime"] == "reversion"

    # ── Full setups ───────────────────────────────────────────────────────
    def test_full_bull_features_positive(self, agent):
        out = run(agent.analyze(make_bull_features()))
        assert out.dir_score > 0

    def test_full_bear_features_negative(self, agent):
        out = run(agent.analyze(make_bear_features()))
        assert out.dir_score < 0


# ─────────────────────────────────────────────────────────────────────────────
# § 3  SqueezeBreakoutAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestSqueezeBreakoutAgent:
    @pytest.fixture
    def agent(self):
        return SqueezeBreakoutAgent()

    def test_name(self, agent):
        assert agent.name == "SqueezeBreakoutAgent"

    def test_timeframe_is_short(self, agent):
        assert agent.timeframe == Timeframe.SHORT

    # ── SQUEEZE_ACTIVE (BB < 0.85× Keltner) ──────────────────────────────
    def test_squeezed_bb_low_score_and_low_conf(self, agent):
        # BB much narrower than Keltner → coiling, no signal
        f = make_features(bb_width=0.020, keltner_width=0.050,   # ratio = 0.4 < 0.85
                          roc_10=0.0, macd_hist=0.0, macd_hist_delta=0.0, adx_14=15.0)
        out = run(agent.analyze(f))
        assert abs(out.dir_score) < 0.3
        assert out.conf < 0.40
        assert out.evidence["regime"] == "squeeze_active"

    def test_squeezed_preserves_directional_hint(self, agent):
        # Even in squeeze, a directional momentum hint should be present (weak)
        f = make_features(bb_width=0.020, keltner_width=0.050,
                          roc_10=0.5, macd_hist=0.5, macd_hist_delta=0.2, adx_14=20.0)
        out = run(agent.analyze(f))
        # dir_score should be slightly positive because ROC/MACD agree on direction
        assert out.dir_score >= 0

    # ── SQUEEZE_FIRE (0.85 ≤ BB/KC < 1.20) ──────────────────────────────
    def test_squeeze_fire_bull_strong_signal(self, agent):
        # Prime the agent with a squeeze-active bar first (ratio < 0.85)
        run(agent.analyze(make_features(bb_width=0.030, keltner_width=0.050,
                                        roc_10=0.4, macd_hist=0.5, macd_hist_delta=0.2, adx_14=28.0)))
        # Now fire: BB just expanding past Keltner: ratio ≈ 0.956
        f = make_features(bb_width=0.043, keltner_width=0.045,   # ratio ≈ 0.956
                          roc_10=0.4, macd_hist=0.5, macd_hist_delta=0.2, adx_14=28.0,
                          ema20_slope=0.008)
        out = run(agent.analyze(f))
        assert out.dir_score > 0.3
        assert out.conf > 0.45
        assert out.evidence["regime"] == "squeeze_fire"

    def test_squeeze_fire_bear_strong_signal(self, agent):
        # Prime with squeeze active first
        run(agent.analyze(make_features(bb_width=0.030, keltner_width=0.050,
                                        roc_10=-0.4, macd_hist=-0.5, macd_hist_delta=-0.2, adx_14=28.0)))
        f = make_features(bb_width=0.043, keltner_width=0.045,
                          roc_10=-0.4, macd_hist=-0.5, macd_hist_delta=-0.2, adx_14=28.0,
                          ema20_slope=-0.008)
        out = run(agent.analyze(f))
        assert out.dir_score < -0.3
        assert out.conf > 0.45

    def test_squeeze_fire_conflicting_roc_macd_reduces_score(self, agent):
        # ROC bullish but MACD bearish → lower signal
        f_agree = make_features(bb_width=0.043, keltner_width=0.045,
                                roc_10=0.4, macd_hist=0.4, macd_hist_delta=0.1, adx_14=25.0)
        f_conflict = make_features(bb_width=0.043, keltner_width=0.045,
                                   roc_10=0.4, macd_hist=-0.4, macd_hist_delta=-0.1, adx_14=25.0)
        out_a = run(agent.analyze(f_agree))
        out_c = run(agent.analyze(f_conflict))
        assert out_a.dir_score > out_c.dir_score

    def test_adx_above_20_boosts_confidence_at_fire(self, agent):
        base = dict(bb_width=0.043, keltner_width=0.045,
                    roc_10=0.4, macd_hist=0.4, macd_hist_delta=0.1)
        f_strong = make_features(adx_14=35.0, **base)
        f_weak   = make_features(adx_14=12.0, **base)
        out_s = run(agent.analyze(f_strong))
        out_w = run(agent.analyze(f_weak))
        assert out_s.conf >= out_w.conf

    # ── BREAKOUT_EXPANDED (BB > 1.20× Keltner) ───────────────────────────
    def test_expanded_bb_gives_reduced_score(self, agent):
        # ratio=1.5 → already expanded → lower score than fire zone
        f_fire     = make_features(bb_width=0.043, keltner_width=0.045,
                                   roc_10=0.4, macd_hist=0.4, macd_hist_delta=0.1, adx_14=25.0)
        f_expanded = make_features(bb_width=0.090, keltner_width=0.050,  # ratio = 1.8
                                   roc_10=0.4, macd_hist=0.4, macd_hist_delta=0.1, adx_14=25.0)
        out_fire = run(agent.analyze(f_fire))
        out_exp  = run(agent.analyze(f_expanded))
        assert out_fire.conf >= out_exp.conf

    def test_expanded_low_adx_dampens_score(self, agent):
        f = make_features(bb_width=0.090, keltner_width=0.050,   # expanded
                          roc_10=0.3, macd_hist=0.3, macd_hist_delta=0.1, adx_14=12.0)
        out = run(agent.analyze(f))
        # Weak ADX + already expanded → should be modest
        assert out.dir_score < 0.5

    def test_regime_label_expanded(self, agent):
        f = make_features(bb_width=0.090, keltner_width=0.050,
                          roc_10=0.3, macd_hist=0.3, macd_hist_delta=0.1, adx_14=25.0)
        out = run(agent.analyze(f))
        assert out.evidence["regime"] == "breakout_expanded"

    # ── Direction symmetry ─────────────────────────────────────────────────
    def test_bull_fire_positive_bear_fire_negative(self, agent):
        base = dict(bb_width=0.043, keltner_width=0.045, adx_14=25.0)
        out_bull = run(agent.analyze(make_features(
            roc_10=0.4, macd_hist=0.4, macd_hist_delta=0.1,
            ema20_slope=0.01, **base)))
        out_bear = run(agent.analyze(make_features(
            roc_10=-0.4, macd_hist=-0.4, macd_hist_delta=-0.1,
            ema20_slope=-0.01, **base)))
        assert out_bull.dir_score > 0
        assert out_bear.dir_score < 0

    # ── Full setups ───────────────────────────────────────────────────────
    def test_full_bull_features_positive(self, agent):
        out = run(agent.analyze(make_bull_features()))
        assert out.dir_score > 0

    def test_full_bear_features_negative(self, agent):
        out = run(agent.analyze(make_bear_features()))
        assert out.dir_score < 0

    # ── Evidence keys ─────────────────────────────────────────────────────
    def test_evidence_keys(self, agent):
        out = run(agent.analyze(make_features()))
        for k in ["bb_kc_ratio", "roc_10", "macd_hist", "adx_14", "regime"]:
            assert k in out.evidence


# ─────────────────────────────────────────────────────────────────────────────
# § 4  OrderFlowAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderFlowAgent:
    @pytest.fixture
    def agent(self):
        return OrderFlowAgent()

    def test_name(self, agent):
        assert agent.name == "OrderFlowAgent"

    def test_timeframe_is_short(self, agent):
        assert agent.timeframe == Timeframe.SHORT

    # ── Body score (bb_percent_b as close-position proxy) ─────────────────
    def test_close_near_high_bullish_body(self, agent):
        f = make_features(bb_percent_b=0.92, roc_10=0.0, realized_vol=0.015,
                          vwap_distance=0.0, macd_hist=0.0)
        out = run(agent.analyze(f))
        assert out.dir_score > 0

    def test_close_near_low_bearish_body(self, agent):
        f = make_features(bb_percent_b=0.08, roc_10=0.0, realized_vol=0.015,
                          vwap_distance=0.0, macd_hist=0.0)
        out = run(agent.analyze(f))
        assert out.dir_score < 0

    # ── ROC velocity ──────────────────────────────────────────────────────
    def test_positive_roc_adds_bull_contribution(self, agent):
        f_up   = make_features(roc_10=+0.4, bb_percent_b=0.5, realized_vol=0.015,
                               vwap_distance=0.0, macd_hist=0.0, macd_hist_delta=0.0)
        f_down = make_features(roc_10=-0.4, bb_percent_b=0.5, realized_vol=0.015,
                               vwap_distance=0.0, macd_hist=0.0, macd_hist_delta=0.0)
        assert run(agent.analyze(f_up)).dir_score > run(agent.analyze(f_down)).dir_score

    def test_roc_zero_neutral(self, agent):
        f = make_features(roc_10=0.0, bb_percent_b=0.5, realized_vol=0.015,
                          vwap_distance=0.0, macd_hist=0.0)
        out = run(agent.analyze(f))
        # With all neutral inputs dir_score should be close to zero
        assert abs(out.dir_score) < 0.4

    # ── Volatility expansion ─────────────────────────────────────────────
    def test_vol_expansion_amplifies_directional_score(self, agent):
        base = dict(bb_percent_b=0.85, roc_10=0.3, vwap_distance=0.0, macd_hist=0.3)
        # High vol_ratio = expanding
        f_expand  = make_features(realized_vol=0.035, atr_price_ratio=0.010, **base)
        # Low vol_ratio = contracting
        f_contract = make_features(realized_vol=0.006, atr_price_ratio=0.015, **base)
        out_e = run(agent.analyze(f_expand))
        out_c = run(agent.analyze(f_contract))
        # Expanding vol should give higher confidence/stronger directional score
        assert out_e.conf >= out_c.conf

    # ── VWAP alignment ────────────────────────────────────────────────────
    def test_buying_pressure_above_vwap_higher_quality(self, agent):
        # Bullish body + ROC + above VWAP vs same but below VWAP
        f_with = make_features(bb_percent_b=0.82, roc_10=0.3, vwap_distance=+1.0,
                               realized_vol=0.020, macd_hist=0.3, macd_hist_delta=0.1)
        f_against = make_features(bb_percent_b=0.82, roc_10=0.3, vwap_distance=-1.0,
                                   realized_vol=0.020, macd_hist=0.3, macd_hist_delta=0.1)
        out_with    = run(agent.analyze(f_with))
        out_against = run(agent.analyze(f_against))
        # Same body+ROC, different VWAP side → with-flow should score >= against-flow
        assert out_with.dir_score >= out_against.dir_score

    def test_selling_pressure_below_vwap_higher_quality(self, agent):
        f_with    = make_features(bb_percent_b=0.08, roc_10=-0.3, vwap_distance=-1.0,
                                   realized_vol=0.020, macd_hist=-0.3, macd_hist_delta=-0.1)
        f_against = make_features(bb_percent_b=0.08, roc_10=-0.3, vwap_distance=+1.0,
                                   realized_vol=0.020, macd_hist=-0.3, macd_hist_delta=-0.1)
        out_with    = run(agent.analyze(f_with))
        out_against = run(agent.analyze(f_against))
        assert out_with.dir_score <= out_against.dir_score

    # ── RSI extreme penalty ───────────────────────────────────────────────
    def test_buying_but_rsi_extreme_overbought_reduces_confidence(self, agent):
        f_ok = make_features(bb_percent_b=0.85, roc_10=0.3, rsi_14=58.0,
                              realized_vol=0.02, vwap_distance=0.5, macd_hist=0.3)
        f_ob = make_features(bb_percent_b=0.85, roc_10=0.3, rsi_14=75.0,
                              realized_vol=0.02, vwap_distance=0.5, macd_hist=0.3)
        assert run(agent.analyze(f_ok)).conf >= run(agent.analyze(f_ob)).conf

    # ── Full setups ───────────────────────────────────────────────────────
    def test_full_bull_features_positive(self, agent):
        out = run(agent.analyze(make_bull_features()))
        assert out.dir_score > 0

    def test_full_bear_features_negative(self, agent):
        out = run(agent.analyze(make_bear_features()))
        assert out.dir_score < 0

    # ── Evidence keys ─────────────────────────────────────────────────────
    def test_evidence_contains_required_keys(self, agent):
        out = run(agent.analyze(make_features()))
        for k in ["bb_percent_b", "roc_10", "vol_ratio", "vwap_distance", "body_score"]:
            assert k in out.evidence


# ─────────────────────────────────────────────────────────────────────────────
# § 5  Cross-agent directional agreement
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossAgentAgreement:
    """On unambiguous bull/bear setups all four scalp agents should agree in direction."""

    @pytest.fixture(autouse=True)
    def agents(self):
        self.scalp   = ScalpingAgent()
        self.vwap    = VwapScalpAgent()
        self.squeeze = SqueezeBreakoutAgent()
        self.flow    = OrderFlowAgent()

    def _run_all(self, features):
        return {
            "scalp":   run(self.scalp.analyze(features)),
            "vwap":    run(self.vwap.analyze(features)),
            "squeeze": run(self.squeeze.analyze(features)),
            "flow":    run(self.flow.analyze(features)),
        }

    def test_all_positive_on_bull_setup(self):
        outputs = self._run_all(make_bull_features())
        for name, out in outputs.items():
            assert out.dir_score > 0, f"{name} should be bullish on bull setup, got {out.dir_score}"

    def test_all_negative_on_bear_setup(self):
        outputs = self._run_all(make_bear_features())
        for name, out in outputs.items():
            assert out.dir_score < 0, f"{name} should be bearish on bear setup, got {out.dir_score}"

    def test_majority_positive_on_bull(self):
        """Even if one agent hedges, ≥3 of 4 should be positive."""
        outputs = self._run_all(make_bull_features())
        positive = sum(1 for o in outputs.values() if o.dir_score > 0)
        assert positive >= 3

    def test_majority_negative_on_bear(self):
        outputs = self._run_all(make_bear_features())
        negative = sum(1 for o in outputs.values() if o.dir_score < 0)
        assert negative >= 3

    def test_all_output_contract_bull(self):
        for out in self._run_all(make_bull_features()).values():
            assert -1.0 <= out.dir_score <= 1.0
            assert  0.0 <= out.conf      <= 1.0

    def test_all_output_contract_bear(self):
        for out in self._run_all(make_bear_features()).values():
            assert -1.0 <= out.dir_score <= 1.0
            assert  0.0 <= out.conf      <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# § 6  Scalp profile config
# ─────────────────────────────────────────────────────────────────────────────

class TestScalpProfileConfig:
    @pytest.fixture(scope="class")
    def cfg(self):
        return load_config_from_yaml("config.demo.yaml", profile="scalp")

    def test_profile_name(self, cfg):
        assert cfg.profile == "scalp"

    def test_symbols_include_liquid_instruments(self, cfg):
        # Scalp now trades all liquid instruments: forex majors, gold, tradable indices
        expected = {
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
            "XAUUSD", "EURJPY", "GBPJPY",
            "DAX", "UK100",
            "US30", "US500", "USTEC",
        }
        assert set(cfg.symbols) == expected

    def test_interval_is_60_seconds(self, cfg):
        assert cfg.interval_seconds == 60

    def test_magic_number_distinct_from_balanced(self, cfg):
        balanced = load_config_from_yaml("config.demo.yaml", profile="balanced")
        assert cfg.mt5.magic_number != balanced.mt5.magic_number

    def test_magic_number_is_434243(self, cfg):
        assert cfg.mt5.magic_number == 434243

    def test_risk_pct_matches_profile(self, cfg):
        assert cfg.risk.base_risk_pct == pytest.approx(0.10)

    def test_max_concurrent_4(self, cfg):
        # Tightened from 8 → 4 after live-trading overtrading analysis
        assert cfg.risk.max_concurrent_trades == 4

    def test_surveillance_20s(self, cfg):
        assert cfg.model_dump().get("realtime", {}).get("surveillance_interval_seconds") == 20

    def test_scalping_agent_weight_3(self, cfg):
        assert cfg.agents.agent_weights.get("ScalpingAgent") == pytest.approx(3.0)

    def test_vwap_agent_weight_2_5(self, cfg):
        assert cfg.agents.agent_weights.get("VwapScalpAgent") == pytest.approx(2.5)

    def test_squeeze_agent_weight_2_5(self, cfg):
        assert cfg.agents.agent_weights.get("SqueezeBreakoutAgent") == pytest.approx(2.5)

    def test_orderflow_agent_weight_2(self, cfg):
        assert cfg.agents.agent_weights.get("OrderFlowAgent") == pytest.approx(2.0)

    def test_momentum_agent_weight_1_5(self, cfg):
        assert cfg.agents.agent_weights.get("MomentumAgent") == pytest.approx(1.5)

    def test_structural_agents_suppressed(self, cfg):
        for name in ["TrendAgent", "RegimeAgent", "BreadthAgent"]:
            w = cfg.agents.agent_weights.get(name, 1.0)
            assert w <= 0.5, f"{name} weight {w} should be ≤ 0.5 in scalp"

    def test_scale_in_disabled(self, cfg):
        assert cfg.model_dump().get("scale_in", {}).get("enabled") is False

    def test_tp_extend_disabled(self, cfg):
        assert cfg.model_dump().get("trailing", {}).get("tp_extend_enabled") is False

    def test_config_validates_without_error(self, cfg):
        assert cfg.validate_config() is True

    def test_min_win_prob_allowed(self, cfg):
        # Tightened from 0.42 → 0.52 after live-trading overtrading analysis
        assert cfg.probability.min_win_prob == pytest.approx(0.52)

    def test_all_four_profiles_distinct_magic(self):
        magics = set()
        for p in ["safe", "balanced", "risky", "scalp", "hft"]:
            cfg = load_config_from_yaml("config.demo.yaml", profile=p)
            magics.add(cfg.mt5.magic_number)
        assert len(magics) == 4, "All four profiles must have distinct magic numbers"


# ─────────────────────────────────────────────────────────────────────────────
# § 7  Registry integration
# ─────────────────────────────────────────────────────────────────────────────

class TestScalpRegistryIntegration:
    def test_scalp_agents_all_registered_and_named(self):
        r = make_scalp_registry()
        names = {a.name for a in r.get_all_agents()}
        assert {"ScalpingAgent", "VwapScalpAgent",
                "SqueezeBreakoutAgent", "OrderFlowAgent"} == names

    def test_scalp_agents_all_short_timeframe(self):
        r = make_scalp_registry()
        for a in r.get_all_agents():
            assert a.timeframe == Timeframe.SHORT, \
                f"{a.name} should be SHORT timeframe"

    def test_all_scalp_agents_appear_in_short_tier(self):
        r = make_scalp_registry()
        short_agents = {a.name for a in r.get_agents_by_timeframe(Timeframe.SHORT)}
        for name in ["ScalpingAgent", "VwapScalpAgent",
                     "SqueezeBreakoutAgent", "OrderFlowAgent"]:
            assert name in short_agents

    def test_weight_multiplier_applied(self):
        r = make_scalp_registry()
        r.set_agent_weight("ScalpingAgent", 3.0)
        assert r.get_agent("ScalpingAgent").weight == 3.0

    def test_disable_and_reenable(self):
        r = make_scalp_registry()
        r.disable_agent("SqueezeBreakoutAgent")
        assert not r.get_agent("SqueezeBreakoutAgent").enabled
        r.enable_agent("SqueezeBreakoutAgent")
        assert r.get_agent("SqueezeBreakoutAgent").enabled

    def test_disabled_agent_not_in_run_outputs(self):
        r = make_scalp_registry()
        r.disable_agent("VwapScalpAgent")
        g = TradingGraph(r, {}, MT5Executor({"simulation": True}))
        f = make_features()
        state = run(g.run("US30", f))
        assert "VwapScalpAgent" not in state["agent_outputs"]

    def test_full_runner_registers_all_scalp_agents(self):
        """TradingRunner (simulation) must include all 4 scalp agents in registry."""
        from tradingagents_v2.runner import _build_registry
        cfg = load_config_from_yaml("config.demo.yaml", profile="scalp")
        registry = _build_registry(cfg)
        names = {a.name for a in registry.get_all_agents()}
        for expected in ["ScalpingAgent", "VwapScalpAgent",
                         "SqueezeBreakoutAgent", "OrderFlowAgent"]:
            assert expected in names


# ─────────────────────────────────────────────────────────────────────────────
# § 8  Graph round-trip (scalp registry)
# ─────────────────────────────────────────────────────────────────────────────

class TestScalpGraphRoundTrip:
    @pytest.fixture(scope="class")
    def graph(self):
        return TradingGraph(make_scalp_registry(), {}, MT5Executor({"simulation": True}))

    def test_graph_compiles(self, graph):
        assert graph.graph is not None

    def test_run_returns_dict(self, graph):
        state = run(graph.run("US30", make_features()))
        assert isinstance(state, dict)

    def test_symbol_preserved(self, graph):
        state = run(graph.run("USTEC", make_features()))
        assert state["symbol"] == "USTEC"

    def test_no_errors_on_neutral_features(self, graph):
        state = run(graph.run("US30", make_features()))
        assert state["errors"] == []

    def test_all_4_agents_ran(self, graph):
        state = run(graph.run("US30", make_features()))
        assert len(state["agent_outputs"]) == 4

    def test_all_agent_scores_in_range(self, graph):
        state = run(graph.run("US30", make_features()))
        for name, out in state["agent_outputs"].items():
            assert -1.0 <= out.dir_score <= 1.0, f"{name} dir_score out of range"
            assert  0.0 <= out.conf <= 1.0,      f"{name} conf out of range"

    def test_decision_valid(self, graph):
        state = run(graph.run("US30", make_features()))
        assert state["decision"] in ("stop", "continue", "approved", "rejected")

    def test_bull_setup_decision_not_stop(self):
        """A strongly bullish short-tier setup should pass alignment with scalp config.

        The scalp registry has only SHORT-tier agents, so dir_long=0 and dir_mid=0.
        The scalp profile sets long_min_score=0.0 and mid_min_score=0.0 so those
        empty tiers don't block the signal — this mirrors live-run behaviour.
        """
        scalp_cfg = load_config_from_yaml("config.demo.yaml", profile="scalp")
        g = TradingGraph(make_scalp_registry(),
                         scalp_cfg.model_dump(),
                         MT5Executor({"simulation": True}))
        state = run(g.run("US30", make_bull_features()))
        assert state["decision"] in ("approved", "rejected", "continue"), \
            f"Expected alignment to pass but got: {state['decision']}"

    def test_empty_registry_decision_stop(self):
        g = TradingGraph(AgentRegistry(), {}, MT5Executor({"simulation": True}))
        state = run(g.run("US30", make_features()))
        assert state["decision"] == "stop"

    def test_timeframe_fusion_populated(self, graph):
        state = run(graph.run("US30", make_features()))
        fusion = state.get("timeframe_fusion")
        assert fusion is not None
        assert -1.0 <= fusion.dir_short <= 1.0
        assert  0.0 <= fusion.conf_short <= 1.0

    def test_metadata_is_dict(self, graph):
        state = run(graph.run("US30", make_features()))
        assert isinstance(state["metadata"], dict)


# ─────────────────────────────────────────────────────────────────────────────
# § 9  Runner integration (scalp profile)
# ─────────────────────────────────────────────────────────────────────────────

class TestScalpRunnerIntegration:
    @pytest.fixture(scope="class")
    def runner(self):
        cfg = load_config_from_yaml("config.demo.yaml", profile="scalp")
        return TradingRunner(config=cfg, simulation=True)

    def test_runner_simulation_mode(self, runner):
        assert runner.executor.simulation_mode is True

    def test_runner_magic_is_scalp(self, runner):
        assert runner.executor.magic_number == 434243

    def test_run_once_returns_all_scalp_symbols(self, runner):
        results = run(runner.run_once())
        expected = {
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
            "XAUUSD", "EURJPY", "GBPJPY",
            "DAX", "UK100",
            "US30", "US500", "USTEC",
        }
        assert set(results.keys()) == expected

    def test_run_once_no_errors(self, runner):
        results = run(runner.run_once())
        for sym, r in results.items():
            assert r["errors"] == [], f"{sym}: unexpected errors {r['errors']}"

    def test_decisions_all_valid(self, runner):
        results = run(runner.run_once())
        for sym, r in results.items():
            assert r["decision"] in ("stop", "continue", "approved", "rejected", "cooldown", "news_blackout"), \
                f"{sym}: unexpected decision '{r['decision']}'"

    def test_executed_is_bool(self, runner):
        results = run(runner.run_once())
        for r in results.values():
            assert isinstance(r.get("executed", False), bool)

    def test_order_comment_contains_scalp(self, runner):
        assert "scp" in runner.executor._order_comment

    def test_result_has_required_keys(self, runner):
        results = run(runner.run_once())
        for r in results.values():
            assert "decision" in r
            assert "errors"   in r
            assert "executed" in r


# ─────────────────────────────────────────────────────────────────────────────
# § 10  Output contract (parametrised over all scalp agents + many feature sets)
# ─────────────────────────────────────────────────────────────────────────────

ALL_SCALP_AGENTS = [
    ScalpingAgent(),
    VwapScalpAgent(),
    SqueezeBreakoutAgent(),
    OrderFlowAgent(),
]

FEATURE_VARIANTS = [
    make_features(),                    # neutral baseline
    make_bull_features(),               # strong bull
    make_bear_features(),               # strong bear
    make_features(atr_14=0.001),        # tiny ATR (micro-pip market)
    make_features(atr_14=500.0),        # huge ATR (e.g. US30 points)
    make_features(vwap_distance=0.0, bb_width=0.0, keltner_width=0.0),  # all zeros
    make_features(rsi_4=100.0, rsi_14=100.0, macd_hist=999.0),          # extreme high
    make_features(rsi_4=0.0,   rsi_14=0.0,   macd_hist=-999.0),         # extreme low
    make_features(adx_14=0.0),           # no trend
    make_features(adx_14=100.0),         # max trend
    make_features(bb_percent_b=0.0),     # at lower band
    make_features(bb_percent_b=1.0),     # at upper band
    make_features(realized_vol=0.0001),  # ultra-low vol
    make_features(realized_vol=1.0),     # absurdly high vol
]


@pytest.mark.parametrize("agent", ALL_SCALP_AGENTS, ids=lambda a: a.name)
@pytest.mark.parametrize("features", FEATURE_VARIANTS,
                          ids=[f"f{i}" for i in range(len(FEATURE_VARIANTS))])
def test_output_contract(agent, features):
    """dir_score ∈ [-1,+1] and conf ∈ [0,1] for every agent × feature combination."""
    out = run(agent.analyze(features))
    assert -1.0 <= out.dir_score <= 1.0, \
        f"{agent.name}: dir_score={out.dir_score} out of [-1,1]"
    assert  0.0 <= out.conf      <= 1.0, \
        f"{agent.name}: conf={out.conf} out of [0,1]"
    assert math.isfinite(out.dir_score), f"{agent.name}: dir_score is not finite"
    assert math.isfinite(out.conf),      f"{agent.name}: conf is not finite"
    assert isinstance(out.rationale, str) and len(out.rationale) > 0


# ─────────────────────────────────────────────────────────────────────────────
# § 11  Idempotency — same inputs → same outputs
# ─────────────────────────────────────────────────────────────────────────────

class TestIdempotency:
    def _assert_idempotent(self, agent, features):
        out1 = run(agent.analyze(features))
        out2 = run(agent.analyze(features))
        assert out1.dir_score == pytest.approx(out2.dir_score, abs=1e-9)
        assert out1.conf      == pytest.approx(out2.conf,      abs=1e-9)

    def test_scalping_agent_idempotent(self):
        self._assert_idempotent(ScalpingAgent(), make_bull_features())

    def test_vwap_agent_idempotent(self):
        self._assert_idempotent(VwapScalpAgent(), make_features(vwap_distance=1.5))

    def test_squeeze_agent_idempotent(self):
        f = make_features(bb_width=0.043, keltner_width=0.045,
                          roc_10=0.3, macd_hist=0.3, macd_hist_delta=0.1)
        self._assert_idempotent(SqueezeBreakoutAgent(), f)

    def test_orderflow_agent_idempotent(self):
        self._assert_idempotent(OrderFlowAgent(), make_bull_features())

    def test_graph_idempotent(self):
        g = TradingGraph(make_scalp_registry(), {}, MT5Executor({"simulation": True}))
        f = make_bull_features()
        s1 = run(g.run("US30", f))
        s2 = run(g.run("US30", f))
        assert s1["decision"] == s2["decision"]


# ─────────────────────────────────────────────────────────────────────────────
# § 12  Degenerate / adversarial inputs — must never crash
# ─────────────────────────────────────────────────────────────────────────────

class TestDegenerateInputs:
    """All agents must handle edge-case financials gracefully without raising."""

    @pytest.mark.parametrize("agent", ALL_SCALP_AGENTS, ids=lambda a: a.name)
    def test_all_zeros_no_crash(self, agent):
        f = make_features(
            atr_14=0.0, atr_5=0.0, rsi_4=0.0, rsi_14=0.0,
            macd_hist=0.0, macd_hist_delta=0.0, roc_10=0.0,
            bb_percent_b=0.0, ema20_slope=0.0, vwap_distance=0.0,
            bb_width=0.0, keltner_width=0.0, realized_vol=0.0,
            atr_price_ratio=0.0,
        )
        out = run(agent.analyze(f))
        assert out is not None
        assert math.isfinite(out.dir_score)
        assert math.isfinite(out.conf)

    @pytest.mark.parametrize("agent", ALL_SCALP_AGENTS, ids=lambda a: a.name)
    def test_atr_zero_no_division_error(self, agent):
        f = make_features(atr_14=0.0, atr_5=0.0, atr_price_ratio=0.0)
        out = run(agent.analyze(f))
        assert out is not None

    @pytest.mark.parametrize("agent", ALL_SCALP_AGENTS, ids=lambda a: a.name)
    def test_keltner_zero_no_division_error(self, agent):
        f = make_features(keltner_width=0.0, bb_width=0.0)
        out = run(agent.analyze(f))
        assert out is not None

    @pytest.mark.parametrize("agent", ALL_SCALP_AGENTS, ids=lambda a: a.name)
    def test_very_large_values_no_crash(self, agent):
        f = make_features(
            atr_14=100000.0, rsi_4=100.0, rsi_14=100.0,
            macd_hist=50000.0, macd_hist_delta=10000.0,
            roc_10=500.0, vwap_distance=999.0, bb_percent_b=1.0,
            bb_width=5000.0, keltner_width=3000.0, realized_vol=100.0,
        )
        out = run(agent.analyze(f))
        assert -1.0 <= out.dir_score <= 1.0
        assert  0.0 <= out.conf      <= 1.0

    @pytest.mark.parametrize("agent", ALL_SCALP_AGENTS, ids=lambda a: a.name)
    def test_empty_swing_lists_no_crash(self, agent):
        f = make_features(swing_highs=[], swing_lows=[])
        out = run(agent.analyze(f))
        assert out is not None

    @pytest.mark.parametrize("agent", ALL_SCALP_AGENTS, ids=lambda a: a.name)
    def test_single_swing_point_no_crash(self, agent):
        f = make_features(swing_highs=[100.0], swing_lows=[99.0])
        out = run(agent.analyze(f))
        assert out is not None

    @pytest.mark.parametrize("agent", ALL_SCALP_AGENTS, ids=lambda a: a.name)
    def test_negative_bb_width_no_crash(self, agent):
        # Pathological: BB width computed as negative (shouldn't happen in practice)
        f = make_features(bb_width=-0.001, keltner_width=0.04)
        out = run(agent.analyze(f))
        assert out is not None

    @pytest.mark.parametrize("agent", ALL_SCALP_AGENTS, ids=lambda a: a.name)
    def test_rsi_exactly_at_thresholds(self, agent):
        for rsi_val in [0.0, 35.0, 45.0, 50.0, 55.0, 65.0, 100.0]:
            f = make_features(rsi_4=rsi_val, rsi_14=rsi_val)
            out = run(agent.analyze(f))
            assert math.isfinite(out.dir_score)

    @pytest.mark.parametrize("agent", ALL_SCALP_AGENTS, ids=lambda a: a.name)
    def test_vwap_distance_exactly_at_thresholds(self, agent):
        for vd in [-2.0, -0.5, -0.1, 0.0, 0.1, 0.5, 2.0]:
            f = make_features(vwap_distance=vd)
            out = run(agent.analyze(f))
            assert math.isfinite(out.dir_score)

    @pytest.mark.parametrize("agent", ALL_SCALP_AGENTS, ids=lambda a: a.name)
    def test_bb_kc_ratio_at_thresholds(self, agent):
        kc = 0.050
        for ratio in [0.0, 0.85, 1.0, 1.20, 2.0]:
            f = make_features(bb_width=kc * ratio, keltner_width=kc)
            out = run(agent.analyze(f))
            assert math.isfinite(out.dir_score)
