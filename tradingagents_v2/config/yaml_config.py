"""
Load TradingConfig from a YAML file.
"""

from pathlib import Path
from typing import Union
import copy
import yaml

from .settings import TradingConfig

# ──────────────────────────────────────────────────────────────────────────────
# Risk-profile presets
# ──────────────────────────────────────────────────────────────────────────────
# Each preset is a deep-merge patch applied on top of the user's YAML before
# TradingConfig is constructed.  Keys that exist in the user YAML are
# OVERRIDDEN by the profile; keys absent from the profile are untouched.
#
# safe     — preserve capital; very few trades, tight filters, half risk
# balanced — default production settings (no change)
# risky    — higher throughput; looser filters, larger risk, more positions
# scalp    — 1-minute cycle on 2-4 index CFDs; tight risk per trade, extremely
#            loose signal filters (only the ScalpingAgent + Momentum vote matters)
#
# Design rules enforced by the presets:
#   • 'risky' never exceeds 0.30% per trade (3× 'safe')
#   • Max daily drawdown scales: safe=0.5%, balanced=1.0%, risky=2.5%
#   • Correlation limit: safe=1, balanced=2, risky=3
#   • All profiles keep VIX risk scaling enabled (floor only varies)
# ──────────────────────────────────────────────────────────────────────────────
_PROFILES = {
    "safe": {
        "mt5": {
            "magic_number": 434241,   # distinct from balanced (434242) for isolation
        },
        "risk": {
            "base_risk_pct":           0.05,   # 0.05% of equity per trade
            "max_daily_drawdown_pct":  0.50,   # halt at -0.5% intraday
            "max_concurrent_trades":   3,
            "per_symbol_leverage_cap": 2.0,
            "portfolio_leverage_cap":  3.0,
            "max_correlated_positions": 1,     # never more than 1 correlated USD trade
            # Overtrading guards: low per-trade risk doesn't protect against
            # spread erosion from excessive frequency.
            "entry_cooldown_minutes":  20,     # min 20 min between entries on same symbol
            "max_daily_trades":        10,     # hard daily cap across all symbols
            # Weekly circuit breaker: 5 days × 0.49% (just below daily halt) = 2.45%.
            # Cap the week at 1.5% to stop a slow losing streak before it compounds.
            "max_weekly_drawdown_pct": 1.50,
            # Pivot buffer: balanced inherits 0.50×ATR from config.demo.yaml.
            # For safe US30 SHORT the support pivot was consistently 0.36×ATR away,
            # permanently blocking the entry.  0.30 is tight enough to still
            # prevent entries into key levels while allowing trades above 0.30×ATR.
            "pivot_buffer_atr": 0.30,
        },
        "alignment": {
            # Thresholds calibrated so safe trades ~1-3×/day on clear D1 moves.
            # The original 0.35/0.28/0.20 required all agents to agree strongly —
            # dir_long > 0.35 almost never occurs on a normal day (weighted avg
            # of ~10 agents rarely exceeds ~0.25 even in a clean uptrend).
            "long_min_score":   0.25,   # D1 conviction — achievable on clear trends; balanced=0.20
            "mid_min_score":    0.18,   # 1H conviction — stricter than balanced (0.10)
            "short_min_score":  0.12,   # 15m conviction — stricter than balanced (0.08)
            "min_win_prob":     0.52,   # higher win-prob bar (balanced=0.47) — kept strict
            "min_ev":           0.12,   # real edge gate (balanced=0.08) — kept strict
            "min_confidence":   0.50,   # agent confidence gate — tighter than risky (0.42)
            "pullback_tolerance": 0.25, # tighter pullback acceptance
            # 1.60 made dead-zone thresholds 0.35×1.60=0.56 — never fires.
            # 1.40 gives 0.25×1.40=0.35, still strict + matches balanced/risky.
            "dead_zone_factor": 1.40,   # was 1.60 — too strict; now matches balanced
            # The 4 scalp-specific agents (ScalpingAgent, VwapScalpAgent,
            # SqueezeBreakoutAgent, OrderFlowAgent) run on 15m/1H bars in the
            # safe profile and routinely output near-zero scores on D1 setups —
            # they effectively add 4 abstaining voters without contributing signal.
            # The comment in config.demo.yaml says "10 directional agents → need 5"
            # but we have 13 (10 swing + 4 scalp - BreadthAgent).  4/13 is already
            # a solid majority of the 10 swing agents; require 4 absolute votes.
            "min_agent_consensus": 4,   # override balanced=5; 4 swing agents = real signal
        },
        "scale_in": {
            "enabled":                 False,  # no pyramiding in safe mode
        },
        "exit_rules": {
            "conviction_fade_threshold":      0.15,   # close sooner when D1 weakens
            "tighten_fade_threshold":         0.25,
            "mid_short_opposition_threshold": 0.30,
            # Time-stop: close any trade that has been open this long without
            # hitting TP.  Prevents capital sitting in sideways/stalled trades.
            # 0 = disabled.  Safe profile: 8h (we entered on a D1 signal — if it
            # hasn't moved in 8h the thesis is wrong).
            "max_trade_duration_hours":       8,
        },
        # Signal freshness: safe profile cycles every 5 min on 21 symbols.
        # The last symbol in the list could be executing a 3-5 min old plan.
        # 120s gives enough headroom while blocking truly stale decisions.
        "execution": {
            "signal_max_age_seconds": 120,
            # Pre-news blackout: wider window than balanced to protect the
            # tightest SLs from gap-through on high-impact events (NFP, Fed, CPI).
            "news_blackout_minutes":  30,
        },
        # Weekend gap protection: close all positions before Friday market close
        # so no exposure remains over the Sunday gap open.
        "realtime": {
            "weekend_close_enabled":     True,
            "weekend_close_utc_hour":    20,   # 20:00 UTC = after London session
        },
        # Partial TP: safe profile locks in MORE of the gain early (60%)
        # because capital preservation > letting winners run.
        "trailing": {
            "partial_tp_enabled":   True,
            "partial_tp_fraction":  0.60,   # close 60% at +1R (more conservative than balanced)
            "partial_tp2_enabled":  True,
            "partial_tp2_r_mult":   2.0,    # fire 2nd partial at +2R
            "partial_tp2_fraction": 0.50,   # close 50% of remainder at +2R
            "windfall_exit_enabled": True,
            "windfall_r_mult":      3.0,    # close ALL at +3R (safe: protect early)
        },
        "vix_risk_scaling": {
            "enabled":   True,
            "threshold": 0.30,   # start reducing earlier (lower VIX tolerance)
            "floor":     0.35,   # cut to 35% of base at peak fear
        },
        "regime_weighting": {
            "trend_threshold": 0.60,   # require stronger trend signal before amplifying
            "range_threshold": 0.35,
        },
        # ── Per-symbol risk caps ──────────────────────────────────────────────
        # Gold (and silver) are macro/geopolitical assets that behave very
        # differently from forex pairs.  D1-based agents are not calibrated for
        # them.  If XAUUSD is ever re-added to safe's symbol list, cap its risk
        # at 25% of base (0.05% × 0.25 = 0.0125% per trade).
        "symbol_risk_multipliers": {
            "XAUUSD": 0.25,   # gold: 25% of base risk on safe profile
            "XAGUSD": 0.25,   # silver: same — high ATR, macro-driven
        },
        # ── Agent weights: suppress scalp agents in safe profile ─────────────
        # ScalpingAgent, VwapScalpAgent, SqueezeBreakoutAgent, and OrderFlowAgent
        # are designed for 1m/3m microstructure.  On 15m/1H/D1 bars they produce
        # near-noise dir_scores that dilute the fusion signal.  Suppressing to 0.1
        # ensures the D1-based swing agents (Trend, Momentum, Regime…) dominate
        # the fusion without entirely silencing intraday confirmation signals.
        "agents": {
            "agent_weights": {
                "ScalpingAgent":        0.10,
                "VwapScalpAgent":       0.10,
                "SqueezeBreakoutAgent": 0.10,
                "OrderFlowAgent":       0.10,
            },
        },
    },

    "balanced": {
        # No overrides — balanced is the default: all settings come from user YAML.
    },

    "risky": {
        # ── Isolation ──────────────────────────────────────────────────────────
        "mt5": {
            "magic_number": 434244,   # distinct from balanced (434242) — order isolation
        },
        # ── Position sizing & hard limits ─────────────────────────────────────
        "risk": {
            "base_risk_pct":           0.20,   # 0.20% equity per trade (2× balanced)
            "max_daily_drawdown_pct":  2.00,   # halt at -2.0% intraday (lowered from 2.5)
            "max_concurrent_trades":   8,
            "per_symbol_leverage_cap": 4.0,
            "portfolio_leverage_cap":  8.0,
            # Correlation cap: 0.20% × 2 = 0.40% max in one macro direction.
            # 3 was too loose (0.60%+ correlated exposure during adverse events).
            "max_correlated_positions": 2,
            # ── Overtrading guards ────────────────────────────────────────────
            # With a 5-min cycle on 21 symbols the bot can place 100+ trades/day
            # without these two caps.  Values are intentionally 2× the scalp caps
            # (risky is allowed to trade more, but not without bound).
            "entry_cooldown_minutes":  10,   # min 10 min between entries on same symbol
            "max_daily_trades":        30,   # hard daily cap across all symbols
            # Weekly ceiling: 5 days × -1.99% (just below daily halt) = -9.9%.
            # Risky allows a worse week than safe, but caps runaway loss streaks.
            "max_weekly_drawdown_pct": 4.00,
        },
        # ── Signal quality gates ───────────────────────────────────────────────
        # These are lower than balanced (more trades) but must still guarantee
        # positive net EV *after* typical spread cost.
        # EV floor check:  p×R − (1−p) ≥ min_ev
        # At p=0.48, R=1.8 (net): 0.48×1.8 − 0.52 = 0.864 − 0.52 = 0.344  ✓
        # At p=0.48, R=1.5 (net): 0.48×1.5 − 0.52 = 0.720 − 0.52 = 0.200  ✓
        "alignment": {
            "long_min_score":    0.18,   # D1 directional bar (lower than balanced 0.25)
            "mid_min_score":     0.14,   # 1H directional bar
            "short_min_score":   0.10,   # 15m directional bar
            "min_win_prob":      0.48,   # floor above random noise (was 0.42 — inert)
            "min_ev":            0.08,   # real edge requirement (was 0.05 ≈ zero-edge)
            "min_confidence":    0.42,   # agent confidence gate (new — was missing)
            "pullback_tolerance": 0.55,
            # Live-trading note: at 0.18 × 1.20 = 0.22 the Asian session was
            # virtually unrestricted.  1.40 matches balanced and prevents low-quality
            # night entries on low-liquidity crosses.
            "dead_zone_factor":  1.40,   # was 1.20 — too loose for 22:00-06:00 UTC
        },
        # ── Pyramiding ────────────────────────────────────────────────────────
        # scale_in at 50% (down from 75%) keeps per-symbol risk sane:
        #   3 entries on EURUSD: 0.20% + 0.10% + 0.10% = 0.40%  (was 0.50%)
        "scale_in": {
            "enabled":                  True,
            "max_positions_per_symbol": 3,     # 3 entries max per symbol
            "require_profit":           True,  # only scale into winners
            "risk_fraction":            0.50,  # 50% of base risk (was 75%)
        },
        # ── Proactive exit rules ──────────────────────────────────────────────
        # Loosened vs balanced to let winners run, but not so loose that trend
        # reversals go unpunished.
        "exit_rules": {
            "conviction_fade_threshold":      0.10,   # was 0.07 — too much rope
            "tighten_fade_threshold":         0.18,
            "mid_short_opposition_threshold": 0.38,   # was 0.45 — caught reversals too late
            # Time-stop: risky uses D1 signals — if a trade hasn't moved after
            # 12h the D1 thesis has stalled and we're paying swap for nothing.
            # 0 = disabled.
            "max_trade_duration_hours":       12,
        },
        # Risky cycles every 5 min on 21 symbols; tolerate up to 3 min staleness.
        # D1-based signals are valid for much longer than their freshness window.
        "execution": {
            "signal_max_age_seconds": 180,
            "news_blackout_minutes":  20,   # block entries during high-impact events
        },
        # Weekend gap protection on for risky — wider SLs don’t fully protect
        # against a Monday gap spike when holding over the weekend.
        "realtime": {
            "weekend_close_enabled":  True,
            "weekend_close_utc_hour": 20,
        },
        # Partial TP: risky locks in LESS early (40%) to let more profit run.
        "trailing": {
            "partial_tp_enabled":   True,
            "partial_tp_fraction":  0.40,   # close 40% at +1R; trail remaining 60%
            "partial_tp2_enabled":  True,
            "partial_tp2_r_mult":   2.5,    # let risky run further before 2nd partial
            "partial_tp2_fraction": 0.40,   # close 40% of remainder at +2.5R
            "windfall_exit_enabled": True,
            "windfall_r_mult":      4.0,    # let risky ride big moves
        },
        # ── Macro fear guard ─────────────────────────────────────────────────
        # Start reducing earlier (0.40 vs 0.55) and floor at 40% (vs 60%).
        # During a VIX spike, keeping 60% of full risk is dangerous at 0.20% per trade.
        "vix_risk_scaling": {
            "enabled":   True,
            "threshold": 0.40,   # start reducing from here  (was 0.55 — too late)
            "floor":     0.40,   # min 40% of base risk      (was 0.60 — too permissive)
        },
        # ── Regime weighting ─────────────────────────────────────────────────
        # Neutral zone 0.40–0.55 (15-point gap) prevents hairline trendiness
        # jitter from rapidly flipping agent weights.  The original 0.45/0.50
        # (5-point gap) made the regime weighting noisy and nearly always active.
        "regime_weighting": {
            "trend_threshold": 0.55,   # was 0.50
            "range_threshold": 0.40,   # was 0.45
        },
        # ── Per-symbol risk caps ──────────────────────────────────────────────
        # If gold/silver are added back to the risky symbol list, limit their
        # risk to 30% of base (0.20% × 0.30 = 0.06% per trade).  XAUUSD on
        # risky's D1 signals + scale_in (3 positions) = 3× that = 0.18% max
        # correlated gold exposure, still within the 0.40% corr cap.
        "symbol_risk_multipliers": {
            "XAUUSD": 0.30,   # gold: 30% of base on risky
            "XAGUSD": 0.30,   # silver: same
        },
    },

    # ── scalp ──────────────────────────────────────────────────────────────
    # Optimised for fast in-and-out on index CFDs (US30, US500, USTEC, DAX).
    # Cycle: every 60 seconds.  SL = 1.5×ATR, TP = 2×ATR (fast structural).
    # Only ScalpingAgent + MomentumAgent matter; all others are heavily suppressed.
    # Risk per trade is kept tiny (0.10%) to survive a string of losses.
    #
    # magic_number is set to a distinct value per profile so that running two
    # instances simultaneously (e.g. balanced + scalp) keeps their trades
    # fully isolated: each bot only sees, manages, and counts its own orders.
    #   balanced / safe / risky: use magic from config.demo.yaml (434242)
    #   scalp: 434243
    "scalp": {
        "interval_seconds": 60,       # re-run full agent cycle every 1 minute
        "symbols": [
            # Forex majors — tightest spreads, deepest liquidity, fast 1m moves
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
            # Commodities
            "XAUUSD",   # Gold — high ATR, excellent for scalp
            # High-beta JPY crosses (risk-on, wide range)
            "EURJPY", "GBPJPY",
            # Tradable indices
            "DAX",      # Germany 40 — liquid 1m action
            "UK100",    # FTSE 100
            # US indices kept for when they reopen
            "US30", "US500", "USTEC",
        ],
        "mt5": {
            "magic_number": 434243,   # DIFFERENT from balanced (434242) — isolation
        },
        "timeframes": {
            "long":  ["1H"],           # D1/1W replaced by 1H context for scalp
            "mid":   ["15m"],
            "short": ["1m"],           # <-- the money timeframe
        },
        "risk": {
            "base_risk_pct":           0.10,   # tiny risk per scalp — 0.10% equity
            "max_daily_drawdown_pct":  0.75,   # halt if down -0.75% intraday (was 1.5%)
            "max_concurrent_trades":   4,      # max 4 simultaneous scalp positions (was 8)
            "per_symbol_leverage_cap": 3.0,
            "portfolio_leverage_cap":  10.0,
            "max_correlated_positions": 2,     # cap USD exposure (was 3)
            "pivot_buffer_atr":        0.20,   # very narrow pivot buffer for scalp
            # ── Overtrading guards ──────────────────────────────────────────
            # entry_cooldown_minutes: minimum minutes between successive entries on
            # the SAME symbol.  Prevents the 60-second cycle from re-entering a
            # just-closed (or just-opened) position continuously.
            "entry_cooldown_minutes":  5,
            # max_daily_trades: hard cap on the total number of orders placed in a
            # UTC calendar day.  Prevents compounding losses during a bad session.
            "max_daily_trades":        20,
            # Weekly ceiling: scalp daily limit is -0.75%; 5 days × -0.74% = -3.7%.
            # Cap at -2.5% to stop a sustained losing week early.
            "max_weekly_drawdown_pct": 2.50,
        },
        "alignment": {
            # Thresholds tightened after live-trading analysis showed that the
            # original loose values (min_ev=0.03, min_win_prob=0.42) produced
            # near-random entries with average win_prob=0.52 — not enough edge
            # to cover spread + slippage on fast scalp moves.
            "long_min_score":    -1.0,   # disable LONG tier gate (no LONG agents in scalp)
            "mid_min_score":     -1.0,   # disable MID  tier gate (no MID  agents in scalp)
            "short_min_score":    0.25,  # need a real 1m signal (was 0.05)
            "min_win_prob":       0.52,  # must cover spread cost (was 0.42)
            "min_ev":             0.20,  # true net EV after spread (was 0.03)
            "min_confidence":     0.50,  # short-tier agent confidence gate (NEW)
            "pullback_tolerance": 0.50,  # tighter pullback acceptance (was 0.75)
            "dead_zone_factor":   1.00,  # no extra tightening at night
            "min_agent_consensus": 3,    # at least 3 of non-breadth agents agree
        },
        "agents": {
            # Scalp-specific agents get high weight; structural/macro agents suppressed.
            # ScalpingAgent  : 1m RSI/MACD/BB momentum snap  (dominant)
            # VwapScalpAgent : VWAP reversion/breakout bias  (dominant)
            # SqueezeBreakout: BB/Keltner squeeze fire        (dominant)
            # OrderFlowAgent : body+vol+VWAP commitment proxy (strong)
            # MomentumAgent  : confirms impulse               (supporting)
            # PatternAgent   : HH/HL structure on 15m         (supporting)
            # All others are run at low weight as veto guards only.
            "agent_weights": {
                "ScalpingAgent":        3.0,
                "VwapScalpAgent":       2.5,
                "SqueezeBreakoutAgent": 2.5,
                "OrderFlowAgent":       2.0,
                "MomentumAgent":        1.5,
                "PatternAgent":         0.5,
                "TrendAgent":           0.3,
                "RegimeAgent":          0.3,
                "VolatilityAgent":      0.3,
                "BreadthAgent":         0.2,
                "MeanReversionAgent":   0.1,
                "IntermarketAgent":     0.2,
                "SessionBreakoutAgent": 0.2,
                "DivergenceAgent":      0.2,
            },
        },
        "scale_in": {
            "enabled": False,        # never pyramid scalp trades
        },
        "exit_rules": {
            # ── D1 / conviction-based exits are DISABLED for scalp ──────────
            # In scalp mode, dir_long comes from 1H agents running at 0.1–0.3×
            # weight; it is almost always near 0.  The conviction_fade check
            # (|dir_long| < threshold) would fire on every surveillance tick
            # (every 20s), silently closing every trade seconds after entry.
            # Same for tighten_on_fade which halves the TP on the same signal.
            "conviction_fade_enabled":        False,
            "tighten_on_fade_enabled":        False,
            # ── Instead: exit on SHORT-tier signal flip (1m reversal) ───────
            # When the dominant 1m momentum reverses past this threshold, the
            # scalp thesis is invalidated and we close immediately.
            "use_short_tier_exits":           True,
            "short_flip_threshold":           0.35,   # |dir_short| vs trade direction
            # ── Mid+Short opposition (kept but looser) ───────────────────────
            "mid_short_opposition_threshold": 0.60,
            "conviction_fade_threshold":      0.05,   # kept for reference (unused)
            "tighten_fade_threshold":         0.10,   # kept for reference (unused)
            # ── Structural ratchet disabled for scalp ───────────────────────
            # update_structural_sl_tp() recalculates SL/TP from weekly pivots every
            # 20s and would overwrite the smart 1m-ATR targets set at entry.
            "skip_structural_sl_tp_ratchet":  True,
            # Time-stop for scalp: if a scalp hasn't hit TP in 2h it's stuck.
            "max_trade_duration_hours":       2,
        },
        # Scalp processes 15 symbols sequentially; with a 1-min cycle and 20s
        # surveillance, bar data can be 40-90s old by order time.  60s is
        # tight enough to catch runaway cycles but loose enough for normal ops.
        "execution": {
            "signal_max_age_seconds": 60,
            # Scalp is most sensitive to news spikes — wider blackout window.
            "news_blackout_minutes":  45,
        },
        "trailing": {
            "atr_multiplier":    1.0,   # very tight trailing (1×ATR)
            "tp_extend_enabled": False, # never extend TP on a scalp
        },
        # ── Per-symbol risk caps ──────────────────────────────────────────────
        # Gold on 1m is manageable (tight ATR SL), but at current prices
        # (~$3,000/oz, 1m ATR = $0.25-0.50) even a 2×ATR SL = $0.50-$1 can be
        # breached by a single news spike without filling at the stated stop.
        # Reduce gold risk to 50% of the scalp base to limit hard-stop slippage.
        "symbol_risk_multipliers": {
            "XAUUSD": 0.50,   # gold: 50% of scalp base risk (0.10% × 0.5 = 0.05%)
            "XAGUSD": 0.50,   # silver: same — high ATR, gap-risk on news
        },
        # Weekend close: scalp trades are intraday by design — never hold over
        # the weekend.  Close 1h earlier than swing profiles.
        # NOTE: this overrides the config.demo.yaml realtime block for scalp.
        "realtime": {
            "surveillance_interval_seconds": 20,
            "weekend_close_enabled":         True,
            "weekend_close_utc_hour":         19,   # 19:00 UTC — 1h earlier for scalp
        },
        # ── Tight ATR-based SL/TP override for scalp mode ──────────────────
        # Weekly pivots / D1 swing levels are 100-200+ pts away on indices —
        # far too distant for a 1m scalp.  This block replaces those structural
        # targets with ATR-derived distances computed from the SHORT (1m) TF.
        # MT5 then auto-closes the moment TP is reached, capturing rapid gains
        # before they reverse.
        #
        # Rule of thumb: SL must be > 1×ATR_1m to survive the full range of the
        # bar currently forming.  0.8×ATR was too tight — normal wicks hit it
        # before the bar even closed.  2.0×ATR gives ~two full bar ranges of room,
        # absorbing intra-bar wicks AND the first few ticks of the next bar.
        #
        #   EURUSD 1m ATR_14 ≈ 0.8-1.5 pips  →  SL ≈ 1.6-3 pips, TP ≈ 2.4-4.5 pips
        #   XAUUSD 1m ATR_14 ≈ $0.25-0.50    →  SL ≈ $0.50-$1,   TP ≈ $0.75-$1.50
        #   R:R  = tp_atr_mult / sl_atr_mult  = 3.0 / 2.0 = 1.5 (unchanged)
        "tight_sl_tp": {
            "enabled":     True,
            "sl_atr_mult": 2.0,   # SL = 2× ATR_1m — survives full bar range + first wick
            "tp_atr_mult": 3.0,   # TP = 3× ATR_1m — R:R 1.5, realistic within 2-5 bars
        },
        "vix_risk_scaling": {
            "enabled":   True,
            "threshold": 0.60,
            "floor":     0.50,
        },
        # NOTE: realtime block above (line ~418) already sets surveillance_interval_seconds=20,
        # weekend_close_enabled=True, weekend_close_utc_hour=19.  Do NOT add a second
        # "realtime" key here — Python dict duplicate keys silently drop the first.
    },
}


def _deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge *patch* into *base*, returning a new dict."""
    result = copy.deepcopy(base)
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_config_from_yaml(
    path: Union[str, Path],
    profile: str = "balanced",
) -> TradingConfig:
    """
    Load configuration from a YAML file and return a TradingConfig instance.

    *profile* can be ``"safe"``, ``"balanced"`` (default), or ``"risky"``.
    The profile preset is deep-merged on top of the user YAML before
    TradingConfig is constructed, so individual YAML settings still apply
    except where the profile explicitly overrides them.

    The YAML may contain a ``profile:`` key as well — the CLI argument
    (if provided) takes precedence.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    # Resolve profile: CLI argument > YAML key > default "balanced"
    resolved_profile = profile or data.get("profile", "balanced")
    if resolved_profile not in _PROFILES:
        raise ValueError(
            f"Unknown profile '{resolved_profile}'. "
            f"Valid options: {list(_PROFILES.keys())}"  # includes 'scalp'
        )
    data["profile"] = resolved_profile   # store resolved value in config

    # Apply profile preset (patch on top of user data)
    preset = _PROFILES[resolved_profile]
    if preset:
        data = _deep_merge(data, preset)

    # Keep the raw `alignment` block intact so TradingGraph can read it directly.
    # Also mirror the human-friendly keys into the structured sub-models so that
    # any code that reads alignment_thresholds / probability still works.
    alignment = data.get("alignment", {})
    if alignment:
        prob = data.setdefault("probability", {})
        at   = data.setdefault("alignment_thresholds", {})
        _map = {
            "long_min_score":  ("alignment_thresholds", "long"),
            "mid_min_score":   ("alignment_thresholds", "mid"),
            "short_min_score": ("alignment_thresholds", "short"),
            "min_win_prob":    ("probability", "min_win_prob"),
            "min_ev":          ("probability", "min_expectancy_r"),
        }
        for yaml_key, (section, field) in _map.items():
            if yaml_key in alignment:
                data[section][field] = alignment[yaml_key]
        # alignment dict is left in `data` so TradingConfig.alignment gets it.

    return TradingConfig(**{k: v for k, v in data.items()
                            if not k.startswith("_")})
