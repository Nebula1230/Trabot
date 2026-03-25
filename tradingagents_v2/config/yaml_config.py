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
        },
        "alignment": {
            "long_min_score":   0.35,   # require stronger D1 conviction
            "mid_min_score":    0.28,
            "short_min_score":  0.20,
            "min_win_prob":     0.52,   # higher win-prob bar
            "min_ev":           0.12,
            "pullback_tolerance": 0.25, # tighter pullback acceptance
            "dead_zone_factor": 1.60,   # extra strict during low-liquidity hours
        },
        "scale_in": {
            "enabled":                 False,  # no pyramiding in safe mode
        },
        "exit_rules": {
            "conviction_fade_threshold":      0.15,   # close sooner when D1 weakens
            "tighten_fade_threshold":         0.25,
            "mid_short_opposition_threshold": 0.30,
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
    },

    "balanced": {
        # No overrides — balanced is the default: all settings come from user YAML.
    },

    "risky": {
        "mt5": {
            "magic_number": 434244,   # distinct from balanced (434242) for isolation
        },
        "risk": {
            "base_risk_pct":           0.20,   # 0.20% per trade (4× safe)
            "max_daily_drawdown_pct":  2.50,   # halt at -2.5% intraday
            "max_concurrent_trades":   8,
            "per_symbol_leverage_cap": 4.0,
            "portfolio_leverage_cap":  8.0,
            "max_correlated_positions": 3,
        },
        "alignment": {
            "long_min_score":   0.18,   # accept weaker D1 signals
            "mid_min_score":    0.14,
            "short_min_score":  0.10,
            "min_win_prob":     0.42,
            "min_ev":           0.05,
            "pullback_tolerance": 0.55,
            "dead_zone_factor": 1.20,   # only slight tightening at night
        },
        "scale_in": {
            "enabled":                  True,
            "max_positions_per_symbol": 3,     # allow up to 3 entries per symbol
            "require_profit":           True,  # still only scale into winners
            "risk_fraction":            0.75,  # scale-in at 75% of base risk
        },
        "exit_rules": {
            "conviction_fade_threshold":      0.07,   # hold longer before giving up
            "tighten_fade_threshold":         0.15,
            "mid_short_opposition_threshold": 0.45,   # need stronger opposition to close
        },
        "vix_risk_scaling": {
            "enabled":   True,
            "threshold": 0.55,   # tolerate more fear before trimming
            "floor":     0.60,   # keep 60% of base risk even at max fear
        },
        "regime_weighting": {
            "trend_threshold": 0.50,
            "range_threshold": 0.45,
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
            "max_daily_drawdown_pct":  1.50,   # halt if down -1.5% intraday
            "max_concurrent_trades":   8,      # up to 8 simultaneous scalp positions
            "per_symbol_leverage_cap": 3.0,
            "portfolio_leverage_cap":  10.0,
            "max_correlated_positions": 3,     # cap USD exposure (many majors are USD-correlated)
            "pivot_buffer_atr":        0.20,   # very narrow pivot buffer for scalp
        },
        "alignment": {
            # Very loose thresholds — we want the scalp agent to fire quickly.
            # long / mid are set to 0.0 because the scalp registry contains only
            # SHORT-timeframe agents.  Those tiers would otherwise score 0.0 and
            # block every signal via the alignment gate.
            "long_min_score":   -1.0,   # disable LONG tier gate (no LONG agents in scalp; strict > needs -1.0)
            "mid_min_score":    -1.0,   # disable MID  tier gate (no MID  agents in scalp; strict > needs -1.0)
            "short_min_score":  0.05,
            "min_win_prob":     0.42,
            "min_ev":           0.03,
            "pullback_tolerance": 0.75,   # accept entries anywhere in the bar
            "dead_zone_factor": 1.00,     # no extra tightening at night
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
        },
        "trailing": {
            "atr_multiplier":    1.0,   # very tight trailing (1×ATR)
            "tp_extend_enabled": False, # never extend TP on a scalp
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
        "realtime": {
            # Surveillance loop runs every 20s so SL is tightened quickly on scalps
            "surveillance_interval_seconds": 20,
        },
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
