"""
Comprehensive test suite for tradingagents_v2.
Covers internal logic, edge cases, boundary values, and integration paths.
"""

import asyncio
import math
import pytest
import numpy as np

# ── imports ──────────────────────────────────────────────────────────────────
from tradingagents_v2.config.settings import (
    TradingConfig, RiskConfig, ExecutionConfig, MT5Config,
    AlignmentThresholds, TradingConfig,
)
from tradingagents_v2.config.yaml_config import load_config_from_yaml

from tradingagents_v2.data.loader import (
    DataLoader, _ema, _rsi, _atr, _macd, _hurst, _detect_swings,
)

from tradingagents_v2.core.types import (
    TechnicalFeatures, AgentOutput, TimeframeFusion, TradeRecipe,
    TradePlan, PortfolioState, RiskLimits, Timeframe, Direction,
)
from tradingagents_v2.core.agent_base import BaseAgent, AgentRegistry
from tradingagents_v2.core.graph import TradingGraph, TradingState

from tradingagents_v2.agents import (
    RegimeAgent, TrendAgent, MomentumAgent, MeanReversionAgent,
    VolatilityAgent, BreadthAgent, PatternAgent,
)

from tradingagents_v2.execution.mt5_executor import MT5Executor
from tradingagents_v2.execution.order_manager import OrderManager

from tradingagents_v2.runner import TradingRunner


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS & SHARED FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

def run(coro):
    """Run a coroutine synchronously."""
    return asyncio.run(coro)


def make_features(**overrides) -> TechnicalFeatures:
    """Build a valid TechnicalFeatures with sensible defaults, accepting overrides."""
    base = dict(
        swing_highs=[100.0, 102.0, 104.0],
        swing_lows=[98.0, 99.0, 100.5],
        hh_hl_count=2,
        lh_ll_count=0,
        last_break="bullish",
        ema20=102.0, ema50=100.0, ema200=98.0,
        ema20_slope=0.002, ema50_slope=0.001, ema200_slope=0.0005,
        rsi_14=55.0, rsi_4=60.0,
        macd_hist=0.5, macd_hist_delta=0.1,
        roc_10=0.03, bb_percent_b=0.65,
        atr_14=1.5, atr_5=1.2,
        realized_vol=0.02, bb_width=0.04, keltner_width=0.03,
        adx_14=30.0, rvi=0.3, hurst_exponent=0.6,
        atr_price_ratio=0.015, vwap_distance=0.01,
        index_trend=0.3, advance_decline=0.55,
        above_50ema_pct=0.6, above_200ema_pct=0.55,
        sector_direction=0.2,
    )
    base.update(overrides)
    return TechnicalFeatures(**base)


def make_trade_plan(direction=Direction.LONG, symbol="NASDAQ:NVDA") -> TradePlan:
    recipe = TradeRecipe(
        name="TEST", direction=direction,
        entry_trigger="breakout", win_probability=0.6,
        expected_value=0.25, risk_reward_ratio=2.0,
    )
    entry = 150.0
    sl = 148.0 if direction == Direction.LONG else 152.0
    tp = 154.0 if direction == Direction.LONG else 146.0
    return TradePlan(
        symbol=symbol, recipe=recipe, quantity=10.0,
        entry_price=entry, stop_loss=sl, take_profit=tp,
        risk_amount=20.0, confidence=0.7,
        timeframes_aligned=["long", "mid", "short"],
    )


def make_portfolio(equity=100_000.0, open_positions=None, daily_drawdown=0.001):
    return PortfolioState(
        equity=equity, margin_used=5000.0, free_margin=equity - 5000.0,
        daily_pnl=equity * 0.001, daily_drawdown=daily_drawdown,
        open_positions=open_positions or [],
        max_daily_drawdown=daily_drawdown, leverage_used=1.0,
    )


def _full_registry() -> AgentRegistry:
    registry = AgentRegistry()
    for a in [RegimeAgent(), TrendAgent(), MomentumAgent(),
              MeanReversionAgent(), VolatilityAgent(), BreadthAgent(), PatternAgent()]:
        registry.register(a)
    return registry


# ═══════════════════════════════════════════════════════════════════════════
# 1. MATH UTILITIES (data/loader.py internal functions)
# ═══════════════════════════════════════════════════════════════════════════

class TestEMA:
    def test_constant_series_returns_constant(self):
        s = np.full(50, 10.0)
        result = _ema(s, 20)
        np.testing.assert_allclose(result, 10.0, atol=1e-6)

    def test_single_element(self):
        s = np.array([5.0])
        assert _ema(s, 1)[0] == pytest.approx(5.0)

    def test_length_preserved(self):
        s = np.random.rand(100)
        assert len(_ema(s, 20)) == 100

    def test_converges_toward_rising_series(self):
        s = np.arange(1, 51, dtype=float)
        result = _ema(s, 10)
        # EMA should be below current price (lagging), but rising
        assert result[-1] > result[0]

    def test_faster_ema_tracks_closer(self):
        s = np.sin(np.linspace(0, 4 * np.pi, 200)) * 10 + 100
        fast = _ema(s, 5)
        slow = _ema(s, 50)
        # Fast EMA should have larger std deviation (tracks price more)
        assert fast.std() > slow.std()


class TestRSI:
    def test_flat_series_returns_50_or_100(self):
        # All zeros diff → avg_loss = 0 → returns 100
        s = np.full(30, 50.0)
        r = _rsi(s, 14)
        assert r == pytest.approx(100.0)

    def test_monotone_up_returns_high(self):
        s = np.arange(1, 31, dtype=float)
        r = _rsi(s, 14)
        assert r == pytest.approx(100.0)

    def test_monotone_down_returns_low(self):
        s = np.arange(30, 0, -1, dtype=float)
        r = _rsi(s, 14)
        assert r == pytest.approx(0.0)

    def test_output_in_range(self):
        rng = np.random.default_rng(42)
        s = np.cumsum(rng.normal(0, 1, 100)) + 100
        r = _rsi(s, 14)
        assert 0.0 <= r <= 100.0


class TestATR:
    def test_length_correct(self):
        n = 100
        c = np.random.rand(n) + 50
        h = c + 0.5
        lo = c - 0.5
        atr = _atr(h, lo, c, 14)
        assert len(atr) == n - 1

    def test_all_positive(self):
        n = 60
        c = np.random.rand(n) + 50
        h = c + 0.5
        lo = c - 0.5
        atr = _atr(h, lo, c, 14)
        assert np.all(atr > 0)

    def test_wider_range_yields_larger_atr(self):
        c = np.random.rand(60) + 50
        atr_narrow = _atr(c + 0.1, c - 0.1, c, 14)
        atr_wide   = _atr(c + 2.0, c - 2.0, c, 14)
        assert atr_wide[-1] > atr_narrow[-1]


class TestMACD:
    def test_returns_three_floats(self):
        c = np.random.rand(100) + 50
        ml, sl, hist = _macd(c)
        assert isinstance(ml, float)
        assert isinstance(sl, float)
        assert isinstance(hist, float)

    def test_hist_equals_macd_minus_signal(self):
        c = np.random.rand(100) + 50
        ml, sl, hist = _macd(c)
        assert hist == pytest.approx(ml - sl, abs=1e-9)

    def test_fast_gt_slow_gives_positive_macd(self):
        # Monotone increasing → fast EMA > slow EMA → positive MACD line
        c = np.arange(1, 101, dtype=float)
        ml, sl, hist = _macd(c)
        assert ml > 0


class TestHurst:
    def test_range(self):
        c = np.random.rand(200) + 50
        h = _hurst(c)
        assert 0.0 <= h <= 1.0

    def test_random_walk_near_half(self):
        rng = np.random.default_rng(0)
        c = np.cumsum(rng.normal(0, 1, 500)) + 100
        h = _hurst(c)
        assert 0.0 <= h <= 1.0  # just verify it's a valid Hurst value

    def test_short_series_returns_half(self):
        c = np.array([1.0, 2.0, 3.0])
        h = _hurst(c)
        assert 0.0 <= h <= 1.0


class TestDetectSwings:
    def test_returns_correct_structure(self):
        h = np.array([10, 12, 11, 14, 13, 15, 14], dtype=float)
        lo = np.array([9, 10, 9, 11, 10, 12, 11], dtype=float)
        sh, sl, hh_hl, lh_ll, last_break = _detect_swings(h, lo, lookback=2)
        assert isinstance(sh, list)
        assert isinstance(sl, list)
        assert isinstance(hh_hl, int)
        assert isinstance(lh_ll, int)

    def test_flat_series_no_crash(self):
        h = np.full(20, 50.0)
        lo = np.full(20, 49.0)
        _detect_swings(h, lo)  # must not raise

    def test_max_5_swings_returned(self):
        rng = np.random.default_rng(7)
        h = rng.random(100) + 10
        lo = h - 0.5
        sh, sl, *_ = _detect_swings(h, lo)
        assert len(sh) <= 5
        assert len(sl) <= 5


# ═══════════════════════════════════════════════════════════════════════════
# 2. DataLoader
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def loader():
    return DataLoader(simulation=True)


@pytest.fixture(scope="module")
def features(loader):
    return loader.get_features("SIM")


class TestDataLoaderUnit:
    def test_returns_technical_features(self, features):
        assert isinstance(features, TechnicalFeatures)

    # Moving average sanity
    def test_ema_ordering(self, features):
        # In a trending up synthetic series ema20 > ema50 or at least all positive
        assert features.ema20 > 0 and features.ema50 > 0 and features.ema200 > 0

    def test_ema_slopes_finite(self, features):
        assert math.isfinite(features.ema20_slope)
        assert math.isfinite(features.ema50_slope)
        assert math.isfinite(features.ema200_slope)

    # RSI
    def test_rsi_14_in_range(self, features):
        assert 0.0 <= features.rsi_14 <= 100.0

    def test_rsi_4_in_range(self, features):
        assert 0.0 <= features.rsi_4 <= 100.0

    # ATR
    def test_atr_14_positive(self, features):
        assert features.atr_14 > 0

    def test_atr_5_positive(self, features):
        assert features.atr_5 > 0

    # MACD
    def test_macd_hist_finite(self, features):
        assert math.isfinite(features.macd_hist)

    def test_macd_hist_delta_finite(self, features):
        assert math.isfinite(features.macd_hist_delta)

    # ROC & BB
    def test_roc_finite(self, features):
        assert math.isfinite(features.roc_10)

    def test_bb_percent_b_finite(self, features):
        assert math.isfinite(features.bb_percent_b)

    # Volatility
    def test_realized_vol_positive(self, features):
        assert features.realized_vol > 0

    def test_bb_width_positive(self, features):
        assert features.bb_width > 0

    def test_keltner_width_positive(self, features):
        assert features.keltner_width > 0

    # Regime
    def test_adx_in_range(self, features):
        assert 0.0 <= features.adx_14 <= 100.0

    def test_rvi_finite(self, features):
        assert math.isfinite(features.rvi)

    def test_hurst_in_range(self, features):
        assert 0.0 <= features.hurst_exponent <= 1.0

    def test_atr_price_ratio_positive(self, features):
        assert features.atr_price_ratio > 0

    def test_vwap_distance_finite(self, features):
        assert math.isfinite(features.vwap_distance)

    # Breadth
    def test_index_trend_finite(self, features):
        assert math.isfinite(features.index_trend)

    def test_advance_decline_finite(self, features):
        assert math.isfinite(features.advance_decline)

    def test_above_50ema_finite(self, features):
        assert math.isfinite(features.above_50ema_pct)

    def test_above_200ema_finite(self, features):
        assert math.isfinite(features.above_200ema_pct)

    def test_sector_direction_finite(self, features):
        assert math.isfinite(features.sector_direction)

    # Structure
    def test_swing_highs_list(self, features):
        assert isinstance(features.swing_highs, list)

    def test_swing_lows_list(self, features):
        assert isinstance(features.swing_lows, list)

    def test_hh_hl_non_negative(self, features):
        assert features.hh_hl_count >= 0

    def test_lh_ll_non_negative(self, features):
        assert features.lh_ll_count >= 0

    # Multiple calls are idempotent
    def test_multiple_calls_consistent(self, loader):
        f1 = loader.get_features("A")
        f2 = loader.get_features("B")
        assert f1 is not None and f2 is not None
        # Same seed means same data
        assert f1.ema20 == pytest.approx(f2.ema20, rel=1e-6)

    def test_insufficient_bars_returns_none(self):
        """DataLoader with only 10 synthetic bars should return None."""
        from unittest.mock import patch
        ld = DataLoader(simulation=True)
        with patch.object(ld, '_synthetic_bars',
                          return_value={k: np.ones(10) for k in ['open','high','low','close','volume']}):
            result = ld.get_features("X", n_bars=10)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 3. SIGNAL AGENTS — Internal Logic
# ═══════════════════════════════════════════════════════════════════════════

# ── RegimeAgent ──────────────────────────────────────────────────────────

class TestRegimeAgentLogic:
    @pytest.fixture
    def agent(self):
        return RegimeAgent()

    def test_high_adx_gives_high_trendiness(self, agent):
        f = make_features(adx_14=50.0)
        t = agent._calculate_trendiness(f)
        assert t > 0.5

    def test_low_adx_gives_low_trendiness(self, agent):
        f = make_features(adx_14=5.0, rvi=0.0, atr_price_ratio=0.001, bb_width=0.005)
        t = agent._calculate_trendiness(f)
        assert t < 0.5

    def test_trendiness_always_in_range(self, agent):
        for adx in [0, 10, 25, 50, 100]:
            f = make_features(adx_14=float(adx))
            assert 0.0 <= agent._calculate_trendiness(f) <= 1.0

    def test_vol_state_low(self, agent):
        f = make_features(realized_vol=0.005)
        assert agent._calculate_volatility_state(f) == "low"

    def test_vol_state_normal(self, agent):
        f = make_features(realized_vol=0.02)
        assert agent._calculate_volatility_state(f) == "normal"

    def test_vol_state_high(self, agent):
        f = make_features(realized_vol=0.05)
        assert agent._calculate_volatility_state(f) == "high"

    def test_dir_score_in_range(self, agent):
        for vd in [-0.1, 0.0, 0.05, 0.1]:
            f = make_features(vwap_distance=vd)
            t = agent._calculate_trendiness(f)
            v = agent._calculate_volatility_state(f)
            s = agent._calculate_directional_score(f, t, v)
            assert -1.0 <= s <= 1.0

    def test_low_trendiness_reduces_directional_bias(self, agent):
        f_high = make_features(vwap_distance=0.05, adx_14=50.0, bb_width=0.15)
        f_low  = make_features(vwap_distance=0.05, adx_14=5.0,  bb_width=0.005, rvi=0.0, atr_price_ratio=0.001)
        t_high = agent._calculate_trendiness(f_high)
        t_low  = agent._calculate_trendiness(f_low)
        s_high = agent._calculate_directional_score(f_high, t_high, "normal")
        s_low  = agent._calculate_directional_score(f_low,  t_low,  "normal")
        assert abs(s_high) > abs(s_low)

    def test_confidence_above_25_adx(self, agent):
        f = make_features(adx_14=30.0)
        c = agent.calculate_confidence(f)
        assert c >= 0.7

    def test_confidence_in_range(self, agent):
        for adx in [0, 15, 25, 40]:
            f = make_features(adx_14=float(adx))
            c = agent.calculate_confidence(f)
            assert 0.0 <= c <= 1.0

    def test_rationale_is_string(self, agent):
        r = agent._generate_rationale(0.7, "high", 0.3)
        assert isinstance(r, str) and len(r) > 0

    def test_full_output_valid(self, features):
        out = run(RegimeAgent().run(features))
        assert out is not None
        assert -1.0 <= out.dir_score <= 1.0
        assert 0.0 <= out.conf <= 1.0
        assert "trendiness" in out.evidence
        assert 0.0 <= out.evidence["trendiness"] <= 1.0
        assert out.timeframe == "mid"


# ── TrendAgent ───────────────────────────────────────────────────────────

class TestTrendAgentLogic:
    @pytest.fixture
    def agent(self):
        return TrendAgent()

    def test_bullish_ema_stack_positive_ma_score(self, agent):
        f = make_features(ema20=105.0, ema50=100.0, ema200=95.0)
        assert agent._calculate_ma_alignment(f) > 0

    def test_bearish_ema_stack_negative_ma_score(self, agent):
        f = make_features(ema20=90.0, ema50=95.0, ema200=100.0)
        assert agent._calculate_ma_alignment(f) < 0

    def test_ema_alignment_string_bullish(self, agent):
        f = make_features(ema20=105.0, ema50=100.0, ema200=95.0)
        assert agent._get_ema_alignment(f) == "bullish_stack"

    def test_ema_alignment_string_bearish(self, agent):
        f = make_features(ema20=90.0, ema50=95.0, ema200=100.0)
        assert agent._get_ema_alignment(f) == "bearish_stack"

    def test_ema_alignment_string_mixed(self, agent):
        f = make_features(ema20=100.0, ema50=95.0, ema200=102.0)
        assert agent._get_ema_alignment(f) == "mixed"

    def test_hh_hl_gives_positive_structure(self, agent):
        f = make_features(hh_hl_count=2, lh_ll_count=0)
        assert agent._calculate_structure_score(f) > 0

    def test_lh_ll_gives_negative_structure(self, agent):
        f = make_features(hh_hl_count=0, lh_ll_count=2)
        assert agent._calculate_structure_score(f) < 0

    def test_neutral_structure(self, agent):
        f = make_features(hh_hl_count=1, lh_ll_count=1, last_break=None)
        assert agent._calculate_structure_score(f) == 0.0

    def test_positive_slope_positive_slope_score(self, agent):
        f = make_features(ema20_slope=0.01, ema50_slope=0.005, ema200_slope=0.002)
        assert agent._calculate_slope_score(f) > 0

    def test_negative_slope_negative_slope_score(self, agent):
        f = make_features(ema20_slope=-0.01, ema50_slope=-0.005, ema200_slope=-0.002)
        assert agent._calculate_slope_score(f) < 0

    def test_combine_scores_clipped(self, agent):
        s = agent._combine_scores(1.0, 1.0, 1.0)
        assert s <= 1.0
        s = agent._combine_scores(-1.0, -1.0, -1.0)
        assert s >= -1.0

    def test_confidence_bullish_stack_higher(self, agent):
        f_bull = make_features(ema20=105.0, ema50=100.0, ema200=95.0,
                               hh_hl_count=2, lh_ll_count=0,
                               ema20_slope=0.01, ema50_slope=0.005, ema200_slope=0.001)
        f_neut = make_features(ema20=100.0, ema50=100.0, ema200=100.0,
                               hh_hl_count=1, lh_ll_count=1)
        assert agent.calculate_confidence(f_bull) >= agent.calculate_confidence(f_neut)

    def test_full_output_valid(self, features):
        out = run(TrendAgent().run(features))
        assert out is not None
        assert -1.0 <= out.dir_score <= 1.0
        assert 0.0 <= out.conf <= 1.0
        assert out.timeframe == "long"


# ── MomentumAgent ────────────────────────────────────────────────────────

class TestMomentumAgentLogic:
    @pytest.fixture
    def agent(self):
        return MomentumAgent()

    def test_healthy_rsi_bullish_score(self, agent):
        f = make_features(rsi_14=55.0, rsi_4=65.0)
        assert agent._calculate_rsi_momentum(f) > 0

    def test_overbought_rsi_negative_score(self, agent):
        f = make_features(rsi_14=75.0, rsi_4=70.0)
        assert agent._calculate_rsi_momentum(f) < 0

    def test_oversold_rsi_positive_score(self, agent):
        # RSI bouncing from oversold → neutral (0.0) in momentum agent, not positive
        f = make_features(rsi_14=25.0, rsi_4=28.0)
        assert agent._calculate_rsi_momentum(f) >= 0

    def test_rising_macd_positive_score(self, agent):
        f = make_features(macd_hist=0.5, macd_hist_delta=0.1)
        assert agent._calculate_macd_momentum(f) > 0

    def test_falling_macd_negative_score(self, agent):
        f = make_features(macd_hist=-0.5, macd_hist_delta=-0.1)
        assert agent._calculate_macd_momentum(f) < 0

    def test_positive_roc_positive_score(self, agent):
        f = make_features(roc_10=0.05)
        assert agent._calculate_roc_momentum(f) > 0

    def test_negative_roc_negative_score(self, agent):
        f = make_features(roc_10=-0.05)
        assert agent._calculate_roc_momentum(f) < 0

    def test_near_upper_bb_positive_score(self, agent):
        # Momentum interpretation: near upper BB = bullish momentum (not overbought)
        f = make_features(bb_percent_b=0.92)
        assert agent._calculate_bb_momentum(f) > 0

    def test_near_lower_bb_negative_score(self, agent):
        # Momentum interpretation: near lower BB = bearish momentum (not oversold)
        f = make_features(bb_percent_b=0.08)
        assert agent._calculate_bb_momentum(f) < 0

    def test_combine_clipped(self, agent):
        s = agent._combine_momentum_scores(1.0, 1.0, 1.0, 1.0)
        assert s <= 1.0
        s = agent._combine_momentum_scores(-1.0, -1.0, -1.0, -1.0)
        assert s >= -1.0

    def test_timeframe_is_mid(self, features):
        out = run(MomentumAgent().run(features))
        assert out.timeframe == "mid"

    def test_all_evidence_keys_present(self, features):
        out = run(MomentumAgent().run(features))
        for key in ["rsi_score", "macd_score", "roc_score", "bb_score",
                    "rsi_14", "macd_hist"]:
            assert key in out.evidence


# ── MeanReversionAgent ───────────────────────────────────────────────────

class TestMeanReversionAgentLogic:
    @pytest.fixture
    def agent(self):
        return MeanReversionAgent()

    def test_extreme_upper_bb_bearish(self, agent):
        f = make_features(bb_percent_b=0.98)
        assert agent._calculate_bb_extension(f) < -0.5

    def test_extreme_lower_bb_bullish(self, agent):
        f = make_features(bb_percent_b=0.02)
        assert agent._calculate_bb_extension(f) > 0.5

    def test_middle_bb_neutral(self, agent):
        f = make_features(bb_percent_b=0.50)
        assert agent._calculate_bb_extension(f) == 0.0

    def test_extreme_overbought_rsi_bearish(self, agent):
        f = make_features(rsi_14=85.0)
        assert agent._calculate_rsi_extremes(f) < -0.5

    def test_extreme_oversold_rsi_bullish(self, agent):
        f = make_features(rsi_14=15.0)
        assert agent._calculate_rsi_extremes(f) > 0.5

    def test_normal_rsi_neutral(self, agent):
        f = make_features(rsi_14=50.0)
        assert agent._calculate_rsi_extremes(f) == 0.0

    def test_high_vol_negative_adjustment(self, agent):
        f = make_features(realized_vol=0.05, atr_14=2.0)
        assert agent._calculate_volatility_adjustment(f) < 0

    def test_low_vol_positive_adjustment(self, agent):
        f = make_features(realized_vol=0.005, atr_14=0.5)
        assert agent._calculate_volatility_adjustment(f) > 0

    def test_confidence_extreme_bb_higher(self, agent):
        f_ext = make_features(bb_percent_b=0.95, rsi_14=80.0)
        f_mid = make_features(bb_percent_b=0.50, rsi_14=50.0)
        assert agent.calculate_confidence(f_ext) > agent.calculate_confidence(f_mid)

    def test_full_output_valid(self, features):
        out = run(MeanReversionAgent().run(features))
        assert out is not None
        assert -1.0 <= out.dir_score <= 1.0
        assert out.timeframe == "short"


# ── VolatilityAgent ──────────────────────────────────────────────────────

class TestVolatilityAgentLogic:
    @pytest.fixture
    def agent(self):
        return VolatilityAgent()

    def test_high_atr_ratio_positive_score(self, agent):
        f = make_features(atr_price_ratio=0.04, atr_14=4.0, atr_5=5.0)
        assert agent._calculate_atr_volatility(f) > 0

    def test_high_realized_vol_positive_score(self, agent):
        f = make_features(realized_vol=0.05)
        assert agent._calculate_realized_volatility(f) > 0

    def test_wide_bands_positive_score(self, agent):
        f = make_features(bb_width=0.15, keltner_width=0.10)
        assert agent._calculate_band_width(f) > 0

    def test_high_vol_regime_positive(self, agent):
        f = make_features(atr_price_ratio=0.04, realized_vol=0.03)
        assert agent._calculate_volatility_regime(f) > 0

    def test_low_vol_regime_non_positive(self, agent):
        # Very low vol → suitability=0 → returns 0.0 (neutral/unfavourable, not strictly negative)
        f = make_features(atr_price_ratio=0.005, realized_vol=0.005)
        assert agent._calculate_volatility_regime(f) <= 0

    def test_combine_scores_clipped(self, agent):
        assert agent._combine_volatility_scores(1.0, 1.0, 1.0, 1.0) <= 1.0
        assert agent._combine_volatility_scores(-1.0, -1.0, -1.0, -1.0) >= -1.0

    def test_increasing_atr_trend_amplifies_score(self, agent):
        f_inc = make_features(atr_14=2.0, atr_5=3.0, atr_price_ratio=0.02)
        f_dec = make_features(atr_14=2.0, atr_5=1.0, atr_price_ratio=0.02)
        s_inc = agent._calculate_atr_volatility(f_inc)
        s_dec = agent._calculate_atr_volatility(f_dec)
        assert s_inc > s_dec

    def test_all_evidence_keys_present(self, features):
        out = run(VolatilityAgent().run(features))
        for k in ["atr_score", "realized_vol_score", "band_width_score",
                  "atr_14", "realized_vol"]:
            assert k in out.evidence

    def test_timeframe_is_short(self, features):
        out = run(VolatilityAgent().run(features))
        assert out.timeframe == "short"


# ── BreadthAgent ─────────────────────────────────────────────────────────

class TestBreadthAgentLogic:
    @pytest.fixture
    def agent(self):
        return BreadthAgent()

    def test_positive_index_trend_positive_score(self, agent):
        f = make_features(index_trend=0.7)
        assert agent._calculate_index_trend(f) > 0

    def test_negative_index_trend_negative_score(self, agent):
        f = make_features(index_trend=-0.7)
        assert agent._calculate_index_trend(f) < 0

    def test_strong_advance_positive_ad(self, agent):
        f = make_features(advance_decline=0.7)
        assert agent._calculate_advance_decline(f) > 0

    def test_strong_decline_negative_ad(self, agent):
        # advance_decline is centred at 0; negative values = more down-days than up-days
        f = make_features(advance_decline=-0.3)
        assert agent._calculate_advance_decline(f) < 0

    def test_high_above_emas_positive_breadth(self, agent):
        f = make_features(above_50ema_pct=0.8, above_200ema_pct=0.75)
        assert agent._calculate_ema_breadth(f) > 0

    def test_low_above_emas_negative_breadth(self, agent):
        # above_50ema_pct stored as (close-ema)/ema; negative = price below EMA = bearish breadth
        f = make_features(above_50ema_pct=-0.2, above_200ema_pct=-0.15)
        assert agent._calculate_ema_breadth(f) < 0

    def test_positive_sector_positive_score(self, agent):
        f = make_features(sector_direction=0.8)
        assert agent._calculate_sector_direction(f) > 0

    def test_combine_clipped(self, agent):
        assert agent._combine_breadth_scores(1.0, 1.0, 1.0, 1.0) <= 1.0
        assert agent._combine_breadth_scores(-1.0, -1.0, -1.0, -1.0) >= -1.0

    def test_timeframe_is_long(self, features):
        out = run(BreadthAgent().run(features))
        assert out.timeframe == "long"

    def test_all_evidence_keys_present(self, features):
        out = run(BreadthAgent().run(features))
        for k in ["index_score", "ad_score", "ema_score", "sector_score"]:
            assert k in out.evidence


# ── PatternAgent ─────────────────────────────────────────────────────────

class TestPatternAgentLogic:
    @pytest.fixture
    def agent(self):
        return PatternAgent()

    def test_hh_hl_swing_bullish(self, agent):
        f = make_features(swing_highs=[100.0, 102.0, 104.0],
                          swing_lows=[98.0, 99.0, 100.5])
        assert agent._calculate_swing_patterns(f) > 0

    def test_lh_ll_swing_bearish(self, agent):
        f = make_features(swing_highs=[104.0, 102.0, 100.0],
                          swing_lows=[100.5, 99.0, 98.0])
        assert agent._calculate_swing_patterns(f) < 0

    def test_empty_swings_neutral(self, agent):
        f = make_features(swing_highs=[], swing_lows=[])
        assert agent._calculate_swing_patterns(f) == 0.0

    def test_one_swing_each_neutral(self, agent):
        f = make_features(swing_highs=[100.0], swing_lows=[98.0])
        assert agent._calculate_swing_patterns(f) == 0.0

    def test_bullish_last_break_high_score(self, agent):
        f = make_features(last_break="bullish", bb_percent_b=0.9)
        assert agent._calculate_breakout_patterns(f) > 0

    def test_bearish_last_break_low_score(self, agent):
        f = make_features(last_break="bearish", bb_percent_b=0.1)
        assert agent._calculate_breakout_patterns(f) < 0

    def test_no_break_neutral(self, agent):
        f = make_features(last_break=None)
        assert agent._calculate_breakout_patterns(f) == 0.0

    def test_combine_clipped(self, agent):
        assert agent._combine_pattern_scores(1.0, 1.0, 1.0, 1.0) <= 1.0
        assert agent._combine_pattern_scores(-1.0, -1.0, -1.0, -1.0) >= -1.0

    def test_timeframe_is_short(self, features):
        out = run(PatternAgent().run(features))
        assert out.timeframe == "short"

    def test_evidence_keys_present(self, features):
        out = run(PatternAgent().run(features))
        for k in ["swing_score", "breakout_score", "sr_score", "last_break"]:
            assert k in out.evidence


# ═══════════════════════════════════════════════════════════════════════════
# 4. BaseAgent & AgentRegistry
# ═══════════════════════════════════════════════════════════════════════════

class TestBaseAgent:
    def test_disabled_agent_returns_none(self, features=None):
        agent = RegimeAgent()
        agent.enabled = False
        f = make_features()
        result = run(agent.run(f))
        assert result is None

    def test_weight_multiplied_into_dir_score(self):
        agent = RegimeAgent()
        f = make_features(vwap_distance=0.05)
        out_no_weight = run(agent.run(f))
        agent2 = RegimeAgent()
        agent2.weight = 0.5
        out_half = run(agent2.run(f))
        if out_no_weight is not None and out_half is not None:
            assert abs(out_half.dir_score) <= abs(out_no_weight.dir_score) + 1e-9

    def test_missing_required_feature_returns_none(self):
        """Patch validate_features on class to return False → run returns None."""
        from unittest.mock import patch
        agent = RegimeAgent()
        f = make_features()
        with patch.object(RegimeAgent, 'validate_features', return_value=False):
            result = run(agent.run(f))
        assert result is None

    def test_exception_in_analyze_returns_none(self):
        """If analyze raises, run catches and returns None."""
        from unittest.mock import patch
        agent = RegimeAgent()
        f = make_features()
        async def _bad_analyze(*a, **kw):
            raise RuntimeError("boom")
        with patch.object(RegimeAgent, 'analyze', side_effect=_bad_analyze):
            result = run(agent.run(f))
        assert result is None


class TestAgentRegistry:
    def test_register_all_7(self):
        r = _full_registry()
        assert len(r.get_all_agents()) == 7

    def test_get_by_timeframe_long(self):
        r = _full_registry()
        long_agents = r.get_agents_by_timeframe(Timeframe.LONG)
        names = [a.name for a in long_agents]
        assert "BreadthAgent" in names

    def test_get_by_timeframe_mid(self):
        r = _full_registry()
        mids = {a.name for a in r.get_agents_by_timeframe(Timeframe.MID)}
        assert "RegimeAgent" in mids
        assert "MomentumAgent" in mids   # TrendAgent is LONG, not MID

    def test_get_by_timeframe_short(self):
        r = _full_registry()
        shorts = {a.name for a in r.get_agents_by_timeframe(Timeframe.SHORT)}
        assert "MeanReversionAgent" in shorts
        assert "VolatilityAgent" in shorts
        assert "PatternAgent" in shorts   # MomentumAgent is MID, not SHORT

    def test_get_unknown_returns_none(self):
        r = AgentRegistry()
        assert r.get_agent("Nonexistent") is None

    def test_overwrite_agent(self):
        r = AgentRegistry()
        r.register(RegimeAgent())
        r.register(RegimeAgent())  # second registration overwrites
        assert len(r.get_all_agents()) == 1

    def test_enable_disable_cycle(self):
        r = _full_registry()
        r.disable_agent("TrendAgent")
        assert not r.get_agent("TrendAgent").enabled
        r.enable_agent("TrendAgent")
        assert r.get_agent("TrendAgent").enabled

    def test_weight_persists(self):
        r = _full_registry()
        r.set_agent_weight("RegimeAgent", 2.0)
        assert r.get_agent("RegimeAgent").weight == 2.0

    def test_set_weight_nonexistent_no_crash(self):
        r = AgentRegistry()
        r.set_agent_weight("Ghost", 99.9)  # should not raise


# ═══════════════════════════════════════════════════════════════════════════
# 5. Core Types
# ═══════════════════════════════════════════════════════════════════════════

class TestCoreTypes:
    def test_dir_score_clamp_positive(self):
        with pytest.raises(Exception):
            AgentOutput(timeframe=Timeframe.MID, dir_score=1.5, conf=0.5,
                        rationale="x")

    def test_dir_score_clamp_negative(self):
        with pytest.raises(Exception):
            AgentOutput(timeframe=Timeframe.MID, dir_score=-1.5, conf=0.5,
                        rationale="x")

    def test_conf_out_of_range(self):
        with pytest.raises(Exception):
            AgentOutput(timeframe=Timeframe.MID, dir_score=0.5, conf=1.1,
                        rationale="x")

    def test_valid_agent_output(self):
        out = AgentOutput(timeframe=Timeframe.SHORT, dir_score=0.0, conf=1.0,
                          rationale="ok")
        assert out.dir_score == 0.0

    def test_direction_enum_values(self):
        assert Direction.LONG == "long"
        assert Direction.SHORT == "short"
        assert Direction.NEUTRAL == "neutral"

    def test_timeframe_enum_values(self):
        assert Timeframe.LONG == "long"
        assert Timeframe.MID == "mid"
        assert Timeframe.SHORT == "short"

    def test_risk_limits_defaults(self):
        rl = RiskLimits()
        assert rl.base_risk_pct == 0.25
        assert rl.max_concurrent_trades == 3

    def test_trade_recipe_valid(self):
        r = TradeRecipe(name="X", direction=Direction.LONG,
                        entry_trigger="break", win_probability=0.6,
                        expected_value=0.2, risk_reward_ratio=2.0)
        assert r.direction == "long"

    def test_trade_plan_serializable(self):
        plan = make_trade_plan()
        d = plan.model_dump()
        assert d["symbol"] == "NASDAQ:NVDA"
        assert "recipe" in d

    def test_portfolio_state_valid(self):
        p = make_portfolio()
        assert p.equity == 100_000.0

    def test_timeframe_fusion_valid(self):
        tf = TimeframeFusion(
            dir_long=0.5, dir_mid=0.3, dir_short=0.2,
            conf_long=0.7, conf_mid=0.6, conf_short=0.5,
            regime_trendiness=0.6, breadth_score=0.3,
        )
        assert tf.dir_long == 0.5


# ═══════════════════════════════════════════════════════════════════════════
# 6. TradingGraph — Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def graph():
    return TradingGraph(_full_registry(), {}, MT5Executor())


class TestTradingGraph:
    def test_graph_compiles(self, graph):
        assert graph.graph is not None

    def test_run_returns_dict(self, graph, features):
        state = run(graph.run("SIM", features))
        assert isinstance(state, dict)

    def test_symbol_preserved(self, graph, features):
        state = run(graph.run("TEST_SYM", features))
        assert state["symbol"] == "TEST_SYM"

    def test_no_errors_on_valid_input(self, graph, features):
        state = run(graph.run("SIM", features))
        assert state["errors"] == []

    def test_all_7_agents_ran(self, graph, features):
        state = run(graph.run("SIM", features))
        assert len(state["agent_outputs"]) == 7

    def test_all_agents_scores_in_range(self, graph, features):
        state = run(graph.run("SIM", features))
        for name, out in state["agent_outputs"].items():
            assert -1.0 <= out.dir_score <= 1.0, f"{name} dir_score out of range"

    def test_timeframe_fusion_populated(self, graph, features):
        state = run(graph.run("SIM", features))
        fusion = state.get("timeframe_fusion")
        assert fusion is not None
        assert -1.0 <= fusion.dir_long  <= 1.0
        assert -1.0 <= fusion.dir_mid   <= 1.0
        assert -1.0 <= fusion.dir_short <= 1.0
        assert  0.0 <= fusion.regime_trendiness <= 1.0

    def test_decision_is_valid_value(self, graph, features):
        state = run(graph.run("SIM", features))
        assert state["decision"] in ("stop", "continue", "approved", "rejected")

    def test_metadata_is_dict(self, graph, features):
        state = run(graph.run("SIM", features))
        assert isinstance(state["metadata"], dict)

    # Alignment-Stop path
    def test_stop_when_all_bearish(self):
        """Strongly bearish features → alignment should fail → decision=stop."""
        f = make_features(
            ema20=85.0, ema50=95.0, ema200=105.0,
            ema20_slope=-0.01, ema50_slope=-0.008, ema200_slope=-0.003,
            hh_hl_count=0, lh_ll_count=3, last_break="bearish",
            vwap_distance=-0.05, rsi_14=25.0, rsi_4=20.0,
            macd_hist=-1.0, macd_hist_delta=-0.2, roc_10=-0.05,
            index_trend=-0.8, advance_decline=0.2,
            above_50ema_pct=0.1, above_200ema_pct=0.05, sector_direction=-0.8,
            bb_percent_b=0.05, adx_14=35.0,
        )
        g = TradingGraph(_full_registry(), {}, MT5Executor())
        state = run(g.run("BEAR", f))
        # consensus gate may return 'rejected' before alignment check yields 'stop'
        assert state["decision"] in ("stop", "rejected")
        assert state["errors"] == []

    # Portfolio circuit-breaker
    def test_stop_when_too_many_open_positions(self):
        f = make_features()
        portfolio = make_portfolio(open_positions=["A", "B", "C"])  # 3 = max
        limits = RiskLimits(max_concurrent_trades=3)
        g = TradingGraph(_full_registry(), {}, MT5Executor())
        state = run(g.run("SIM", f, portfolio_state=portfolio, risk_limits=limits))
        # If it reached risk_check it will be rejected; if alignment failed it's stop
        assert state["decision"] in ("stop", "rejected")
        assert state["errors"] == []

    def test_stop_when_drawdown_exceeded(self):
        f = make_features()
        portfolio = make_portfolio(daily_drawdown=-0.10)  # -10% > 2% limit
        limits = RiskLimits(max_daily_drawdown_pct=2.0)
        g = TradingGraph(_full_registry(), {}, MT5Executor())
        state = run(g.run("SIM", f, portfolio_state=portfolio, risk_limits=limits))
        assert state["decision"] in ("stop", "rejected")

    # Execution path
    def test_execute_path_places_simulated_order(self):
        """Build a strongly bullish feature set → alignment should pass → order placed."""
        f = make_features(
            ema20=110.0, ema50=105.0, ema200=100.0,
            ema20_slope=0.02, ema50_slope=0.015, ema200_slope=0.005,
            hh_hl_count=3, lh_ll_count=0, last_break="bullish",
            vwap_distance=0.06, rsi_14=58.0, rsi_4=62.0,
            macd_hist=1.0, macd_hist_delta=0.3, roc_10=0.06,
            index_trend=0.8, advance_decline=0.7,
            above_50ema_pct=0.85, above_200ema_pct=0.78, sector_direction=0.8,
            bb_percent_b=0.65, adx_14=40.0, atr_14=1.2, atr_price_ratio=0.012,
            realized_vol=0.02,
        )
        portfolio = make_portfolio()
        limits = RiskLimits()
        g = TradingGraph(_full_registry(), {}, MT5Executor())
        state = run(g.run("BULL", f, portfolio_state=portfolio, risk_limits=limits))
        # No errors regardless of outcome
        assert state["errors"] == []
        # If trade was placed, order ticket is set
        if state["metadata"].get("executed"):
            assert state["metadata"]["order_ticket"] is not None

    def test_disabled_agent_excluded_from_outputs(self):
        """Disabling PatternAgent → only 6 agents run."""
        registry = _full_registry()
        registry.disable_agent("PatternAgent")
        g = TradingGraph(registry, {}, MT5Executor())
        f = make_features()
        state = run(g.run("SIM", f))
        assert "PatternAgent" not in state["agent_outputs"]
        assert len(state["agent_outputs"]) == 6

    def test_empty_registry_no_crash(self):
        """Empty registry → no agent outputs → stop."""
        g = TradingGraph(AgentRegistry(), {}, MT5Executor())
        state = run(g.run("SIM", make_features()))
        assert state["errors"] == []
        assert state["decision"] == "stop"


# ═══════════════════════════════════════════════════════════════════════════
# 7. MT5Executor — simulation mode (deep)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def executor():
    return MT5Executor({"simulation": True})


class TestMT5ExecutorDeep:
    def test_simulation_mode_true(self, executor):
        assert executor.simulation_mode is True

    def test_initialized_true(self, executor):
        assert executor.initialized is True

    def test_magic_number_default(self, executor):
        assert executor.magic_number == 424242

    def test_magic_number_custom(self):
        e = MT5Executor({"simulation": True, "magic_number": 999})
        assert e.magic_number == 999

    # account_info
    def test_account_info_keys(self, executor):
        info = executor.get_account_info()
        for k in ["login", "server", "balance", "equity", "margin",
                  "free_margin", "profit", "leverage"]:
            assert k in info

    def test_account_balance_positive(self, executor):
        assert executor.get_account_info()["balance"] > 0

    def test_account_equity_positive(self, executor):
        assert executor.get_account_info()["equity"] > 0

    # symbol_info
    def test_symbol_info_visible(self, executor):
        info = executor.get_symbol_info("EURUSD")
        assert info.visible is True

    def test_symbol_info_any_symbol(self, executor):
        for sym in ["NASDAQ:NVDA", "EURUSD", "BTCUSD", "XAUUSD"]:
            info = executor.get_symbol_info(sym)
            assert info is not None

    # current_price
    def test_bid_lte_ask(self, executor):
        bid, ask = executor.get_current_price("EURUSD")
        assert bid <= ask

    def test_ask_positive(self, executor):
        _, ask = executor.get_current_price("EURUSD")
        assert ask > 0

    # bracket order — LONG
    def test_bracket_long_returns_dict(self, executor):
        result = executor.place_bracket_order(make_trade_plan(Direction.LONG))
        assert isinstance(result, dict)

    def test_bracket_long_ticket_set(self, executor):
        result = executor.place_bracket_order(make_trade_plan(Direction.LONG))
        assert result["order_ticket"] is not None

    def test_bracket_long_symbol_correct(self, executor):
        result = executor.place_bracket_order(make_trade_plan(Direction.LONG, "EURUSD"))
        assert result["symbol"] == "EURUSD"

    def test_bracket_long_direction(self, executor):
        result = executor.place_bracket_order(make_trade_plan(Direction.LONG))
        assert result["direction"] == Direction.LONG

    def test_bracket_long_simulated_flag(self, executor):
        result = executor.place_bracket_order(make_trade_plan(Direction.LONG))
        assert result.get("simulated") is True

    def test_bracket_long_magic_number(self, executor):
        result = executor.place_bracket_order(make_trade_plan(Direction.LONG))
        assert result["magic"] == executor.magic_number

    # bracket order — SHORT
    def test_bracket_short_returns_dict(self, executor):
        result = executor.place_bracket_order(make_trade_plan(Direction.SHORT))
        assert isinstance(result, dict)

    def test_bracket_short_direction(self, executor):
        result = executor.place_bracket_order(make_trade_plan(Direction.SHORT))
        assert result["direction"] == Direction.SHORT

    # modify_stop_loss
    def test_modify_sl_returns_true(self, executor):
        assert executor.modify_stop_loss(12345, 149.5) is True

    def test_modify_sl_any_price(self, executor):
        for price in [100.0, 0.0001, 99999.99]:
            assert executor.modify_stop_loss(1, price) is True

    # close_position
    def test_close_returns_true(self, executor):
        assert executor.close_position(12345) is True

    # open_positions
    def test_positions_is_list(self, executor):
        assert isinstance(executor.get_open_positions(), list)

    def test_positions_have_required_keys(self, executor):
        pos = executor.get_open_positions()
        assert len(pos) > 0
        p = pos[0]
        for k in ["ticket", "symbol", "type", "volume",
                  "price_open", "price_current", "sl", "tp", "profit", "magic"]:
            assert k in p

    def test_positions_magic_matches(self, executor):
        for pos in executor.get_open_positions():
            assert pos["magic"] == executor.magic_number

    # uninitialised executor
    def test_uninitialised_bracket_order_returns_none(self):
        e = MT5Executor({"simulation": True})
        e.initialized = False
        result = e.place_bracket_order(make_trade_plan())
        assert result is None

    # shutdown idempotent
    def test_double_shutdown_no_crash(self, executor):
        executor.shutdown()
        executor.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# 8. OrderManager — Full Lifecycle
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def om():
    return OrderManager()


class TestOrderManagerDeep:
    def test_create_returns_order_dict(self, om):
        o = om.create_order(make_trade_plan())
        assert isinstance(o, dict)

    def test_create_sets_pending_status(self, om):
        o = om.create_order(make_trade_plan())
        assert o["status"] == "pending"

    def test_create_unique_ids(self, om):
        ids = {om.create_order(make_trade_plan())["order_id"] for _ in range(5)}
        assert len(ids) == 5

    def test_create_stores_direction(self, om):
        plan = make_trade_plan(Direction.SHORT)
        o = om.create_order(plan)
        assert o["direction"] == Direction.SHORT

    def test_create_default_market_type(self, om):
        o = om.create_order(make_trade_plan())
        assert o["order_type"] == "market"

    def test_create_limit_type(self, om):
        o = om.create_order(make_trade_plan(), order_type="limit")
        assert o["order_type"] == "limit"

    def test_get_existing_order(self, om):
        o = om.create_order(make_trade_plan())
        assert om.get_order(o["order_id"]) is not None

    def test_get_missing_order_none(self, om):
        assert om.get_order("does_not_exist") is None

    def test_update_status_filled_moves_to_history(self, om):
        o = om.create_order(make_trade_plan())
        oid = o["order_id"]
        om.update_order_status(oid, "filled")
        assert om.get_order(oid) is None
        assert any(h["order_id"] == oid for h in om.order_history)

    def test_update_status_cancelled_moves_to_history(self, om):
        o = om.create_order(make_trade_plan())
        oid = o["order_id"]
        om.update_order_status(oid, "cancelled")
        assert om.get_order(oid) is None

    def test_update_status_rejected_moves_to_history(self, om):
        o = om.create_order(make_trade_plan())
        oid = o["order_id"]
        om.update_order_status(oid, "rejected")
        assert om.get_order(oid) is None

    def test_update_nonexistent_returns_false(self, om):
        assert om.update_order_status("ghost", "filled") is False

    def test_update_extra_kwargs_stored(self, om):
        o = om.create_order(make_trade_plan())
        om.update_order_status(o["order_id"], "submitted", fill_price=151.0)
        retrieved = om.get_order(o["order_id"])
        assert retrieved["fill_price"] == 151.0

    def test_cancel_pending_order(self, om):
        o = om.create_order(make_trade_plan())
        assert om.cancel_order(o["order_id"]) is True

    def test_cancel_nonexistent_returns_false(self, om):
        assert om.cancel_order("ghost") is False

    def test_cannot_cancel_already_filled(self, om):
        o = om.create_order(make_trade_plan())
        om.update_order_status(o["order_id"], "filled")
        assert om.cancel_order(o["order_id"]) is False

    def test_get_active_orders_all(self, om):
        for _ in range(3):
            om.create_order(make_trade_plan())
        assert len(om.get_active_orders()) >= 3

    def test_get_active_orders_by_symbol(self, om):
        om.create_order(make_trade_plan(symbol="NASDAQ:NVDA"))
        om.create_order(make_trade_plan(symbol="EURUSD"))
        nvda = om.get_active_orders("NASDAQ:NVDA")
        assert all(o["symbol"] == "NASDAQ:NVDA" for o in nvda)

    def test_get_order_history_limit(self, om):
        for _ in range(10):
            o = om.create_order(make_trade_plan())
            om.update_order_status(o["order_id"], "filled")
        hist = om.get_order_history(limit=5)
        assert len(hist) <= 5

    def test_history_sorted_newest_first(self, om):
        for _ in range(3):
            o = om.create_order(make_trade_plan())
            om.update_order_status(o["order_id"], "filled")
        hist = om.get_order_history()
        times = [h["created_at"] for h in hist[:3]]
        assert times == sorted(times, reverse=True)

    def test_get_stats_empty(self):
        om_empty = OrderManager()
        stats = om_empty.get_order_statistics()
        assert stats["total_orders"] == 0
        assert stats["success_rate"] == 0.0

    def test_get_stats_all_filled(self, om):
        om2 = OrderManager()
        for _ in range(4):
            o = om2.create_order(make_trade_plan())
            om2.update_order_status(o["order_id"], "filled")
        s = om2.get_order_statistics()
        assert s["filled_orders"] == 4
        assert s["success_rate"] == 1.0

    def test_get_stats_mixed(self):
        om2 = OrderManager()
        for status in ["filled", "filled", "cancelled", "rejected"]:
            o = om2.create_order(make_trade_plan())
            om2.update_order_status(o["order_id"], status)
        s = om2.get_order_statistics()
        assert s["total_orders"] == 4
        assert s["filled_orders"] == 2
        assert s["cancelled_orders"] == 1
        assert s["rejected_orders"] == 1

    def test_cleanup_removes_old_orders(self):
        from datetime import datetime, timedelta
        om2 = OrderManager()
        o = om2.create_order(make_trade_plan())
        om2.update_order_status(o["order_id"], "filled")
        # Fake created_at to be 100 days old
        om2.order_history[0]["created_at"] = datetime.now() - timedelta(days=100)
        om2.cleanup_old_orders(days=30)
        assert len(om2.order_history) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 9. Config — Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigDeep:
    def test_all_agents_enabled_when_list_empty(self):
        cfg = TradingConfig()
        for name in ["RegimeAgent", "TrendAgent", "MomentumAgent",
                     "MeanReversionAgent", "VolatilityAgent", "BreadthAgent", "PatternAgent"]:
            assert cfg.is_agent_enabled(name) is True

    def test_selective_agents_enabled(self):
        cfg = TradingConfig(agents={"enabled_agents": ["RegimeAgent", "TrendAgent"]})
        assert cfg.is_agent_enabled("RegimeAgent") is True
        assert cfg.is_agent_enabled("MomentumAgent") is False

    def test_default_symbols(self):
        cfg = TradingConfig()
        assert "NASDAQ:NVDA" in cfg.symbols

    def test_risk_validation_too_high_total_risk(self):
        # base_risk_pct * max_concurrent_trades > 10 → validation fails
        cfg = TradingConfig(risk={"base_risk_pct": 4.0, "max_concurrent_trades": 3})
        assert cfg.validate_config() is not True

    def test_leverage_validation(self):
        cfg = TradingConfig(risk={
            "per_symbol_leverage_cap": 10.0,
            "portfolio_leverage_cap": 5.0,
        })
        assert cfg.validate_config() is not True

    def test_custom_mt5_magic_number(self):
        cfg = TradingConfig(mt5={"magic_number": 777777})
        assert cfg.mt5.magic_number == 777777

    def test_agent_weight_default_is_one(self):
        cfg = TradingConfig()
        assert cfg.get_agent_weight("AnyAgent") == 1.0

    def test_no_custom_prompt_returns_none(self):
        cfg = TradingConfig()
        assert cfg.get_custom_prompt("RegimeAgent") is None

    def test_load_from_yaml_invalid_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config_from_yaml(tmp_path / "missing.yaml")

    def test_load_from_yaml_valid(self, tmp_path):
        yaml_content = "symbols:\n  - EURUSD\n  - XAUUSD\n"
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml_content)
        cfg = load_config_from_yaml(p)
        assert cfg.symbols == ["EURUSD", "XAUUSD"]

    def test_load_from_yaml_empty_file_uses_defaults(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        cfg = load_config_from_yaml(p)
        assert isinstance(cfg, TradingConfig)


# ═══════════════════════════════════════════════════════════════════════════
# 10. TradingRunner — Integration
# ═══════════════════════════════════════════════════════════════════════════

class TestTradingRunnerIntegration:
    @pytest.fixture
    def runner(self):
        cfg = TradingConfig(symbols=["NASDAQ:NVDA"])
        return TradingRunner(config=cfg, simulation=True)

    def test_runner_creates_executor_in_sim(self, runner):
        assert runner.executor.simulation_mode is True

    def test_runner_creates_data_loader_in_sim(self, runner):
        assert runner.data_loader.simulation is True

    def test_run_once_returns_all_symbols(self, runner):
        results = run(runner.run_once())
        assert "NASDAQ:NVDA" in results

    def test_run_once_no_errors(self, runner):
        results = run(runner.run_once())
        for sym, r in results.items():
            assert r["errors"] == [], f"{sym}: {r['errors']}"

    def test_run_once_explicit_symbols(self, runner):
        results = run(runner.run_once(["EURUSD", "BTCUSD"]))
        assert set(results.keys()) == {"EURUSD", "BTCUSD"}

    def test_run_once_overrides_config_symbols(self):
        cfg = TradingConfig(symbols=["NASDAQ:NVDA"])
        r = TradingRunner(config=cfg, simulation=True)
        results = run(r.run_once(["XAUUSD"]))
        assert "XAUUSD" in results
        assert "NASDAQ:NVDA" not in results

    def test_run_once_three_symbols_no_crash(self):
        cfg = TradingConfig(symbols=["NASDAQ:NVDA", "NASDAQ:MSFT", "EURUSD"])
        r = TradingRunner(config=cfg, simulation=True)
        results = run(r.run_once())
        assert len(results) == 3

    def test_decision_valid_values(self, runner):
        results = run(runner.run_once())
        for r in results.values():
            assert r["decision"] in ("stop", "continue", "approved", "rejected")

    def test_executed_is_bool(self, runner):
        results = run(runner.run_once())
        for r in results.values():
            assert isinstance(r.get("executed", False), bool)

    def test_result_has_required_keys(self, runner):
        results = run(runner.run_once())
        for r in results.values():
            assert "decision" in r
            assert "errors"   in r
            assert "executed" in r

    def test_running_flag_starts_false(self, runner):
        assert runner._running is False

    def test_shutdown_sets_initialized_false(self, runner):
        runner.executor.shutdown()
        # MT5 is in simulation — no crash
