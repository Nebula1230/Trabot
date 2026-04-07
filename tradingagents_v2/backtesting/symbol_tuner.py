"""
Per-symbol parameter tuner — finds optimal sizing and exit thresholds
for each symbol via analytical replay + grid search with IS/OOS validation.

Architecture (fast — runs ONE full backtest, then replays analytically):
  1. Run a single reference backtest at 1H granularity
  2. Collect all trades with their confidence, streak context, entry_type, pnl_r
  3. For each param combo, re-compute risk_amount analytically and rebuild
     the equity curve from the altered dollar PnL → derive Sharpe
  4. CT mid-flip threshold: run a separate reference backtest per CT value
     (only 5 runs, vs 40+ for full grid)

Tunable parameters (3 stages, searched sequentially):
  Stage 1: confidence_sizing.floor × confidence_sizing.ceil
  Stage 2: streak_sizing.loss_cut × streak_sizing.loss_cut_per_streak
  Stage 3: exit_rules.ct_mid_flip_threshold  (needs separate backtests)

Scoring: Sharpe ratio (min-trade-count filter).
Split:   First 70% of date range = In-Sample, last 30% = Out-of-Sample.

Usage:
    tuner = SymbolTuner(config)
    result = tuner.tune("USDJPY", "2024-06-01", "2025-06-01")
    # result.best_params → inject into config["symbol_overrides"]["USDJPY"]

CLI:
    python -m tradingagents_v2.backtesting --tune --symbol USDJPY EURUSD ...
    python -m tradingagents_v2.backtesting --tune-file symbol_params.json ...
"""

import copy
import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .engine import BacktestEngine, BacktestResult, ClosedTrade
from .metrics import compute_metrics
from ..config.settings import TradingConfig
from ..data.loader import _MIN_BARS

logger = logging.getLogger("SymbolTuner")

# ─────────────────────────────────────────────────────────────────────────────
# Parameter grids
# ─────────────────────────────────────────────────────────────────────────────

CONF_FLOOR_GRID = [0.60, 0.70, 0.80, 0.90, 1.0]
CONF_CEIL_GRID  = [1.0, 1.10, 1.20, 1.30, 1.40]

STREAK_LOSS_CUT_GRID = [0.50, 0.60, 0.70, 0.80]
STREAK_LOSS_PER_GRID = [0.05, 0.10, 0.15, 0.20]

CT_MID_FLIP_GRID = [0.40, 0.50, 0.55, 0.60, 0.70]

MIN_TRADES = 15   # trials with fewer trades are penalised


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TuneTrialResult:
    params: Dict[str, Any]
    sharpe: float
    total_return_pct: float
    total_trades: int
    win_rate: float
    max_drawdown_pct: float


@dataclass
class SymbolTuneResult:
    symbol: str
    best_params: Dict[str, Any]
    is_sharpe: float
    oos_sharpe: float
    is_return_pct: float
    oos_return_pct: float
    trials_run: int
    stages: Dict[str, List[TuneTrialResult]] = field(default_factory=dict)


@dataclass
class _TradeRecord:
    """Compact record of one closed trade for analytical replay."""
    pnl_r: float            # R-value (trade quality, independent of sizing)
    confidence: float       # fusion confidence at entry
    is_short: bool          # short direction?
    entry_type: str         # "full-alignment", "counter-trend-scalp", etc.
    base_risk: float        # risk_amount BEFORE confidence/streak modifiers
    trade_idx: int          # sequential index (0-based) for streak calc


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_end_bar(bar_dates: List[datetime], date_str: str) -> int:
    end_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    for i, d in enumerate(bar_dates):
        if d > end_dt:
            return i
    return len(bar_dates)


def _slice_bars(
    all_bars: Dict[str, Dict[str, np.ndarray]], end_idx: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        key: {col: arr[:min(end_idx, len(arr))] for col, arr in arrays.items()}
        for key, arrays in all_bars.items()
    }


def _deep_merge(base: Dict, overrides: Dict) -> None:
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            base[key].update(val)
        else:
            base[key] = val


def _compute_sharpe(equity_curve: List[float]) -> float:
    """Annualised Sharpe from an equity curve (bar-level)."""
    if len(equity_curve) < 2:
        return 0.0
    eq = np.array(equity_curve, dtype=float)
    rets = np.diff(eq) / eq[:-1]
    if rets.std() < 1e-12:
        return 0.0
    # Annualise using a fixed factor (~252 trading days).
    # Previous code used sqrt(len(rets)) which made Sharpe a t-statistic,
    # biasing IS/OOS comparison and CT threshold selection.
    ann_factor = math.sqrt(252)
    return float(rets.mean() / rets.std() * ann_factor)


def _max_drawdown_pct(equity_curve: List[float]) -> float:
    eq = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1.0)
    return float(dd.max() * 100) if len(dd) > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Analytical replay: re-build equity curve from trade list + new sizing params
# ─────────────────────────────────────────────────────────────────────────────

def _replay_sizing(
    trades: List[_TradeRecord],
    initial_equity: float,
    base_risk_pct: float,
    conf_floor: float,
    conf_ceil: float,
    conf_base: float,
    short_penalty: float,
    streak_enabled: bool,
    win_boost: float,
    loss_cut: float,
    loss_cut_per_streak: float,
) -> Tuple[List[float], int]:
    """Replay trade list with new sizing params → (equity_curve, n_wins).

    Returns a simplified equity curve (one point per trade) and win count.
    """
    equity = initial_equity
    curve = [equity]
    wins = 0

    for i, t in enumerate(trades):
        # Base risk from current equity
        risk = equity * base_risk_pct / 100.0

        # Confidence sizing
        frac = min(1.0, max(0.0,
            (t.confidence - conf_base) / max(1.0 - conf_base, 1e-9)))
        conf_mult = conf_floor + frac * (conf_ceil - conf_floor)
        risk *= conf_mult

        # Short penalty
        if t.is_short and short_penalty < 1.0:
            risk *= short_penalty

        # Streak sizing
        if streak_enabled and i >= 2:
            recent = [trades[j].pnl_r for j in range(max(0, i - 10), i)]
            if len(recent) >= 2:
                last = recent[-1]
                streak_len = 1
                for prev_r in reversed(recent[:-1]):
                    if (prev_r > 0) == (last > 0):
                        streak_len += 1
                    else:
                        break
                if streak_len >= 2:
                    if last > 0:
                        s_mult = min(win_boost, 1.0 + 0.05 * streak_len)
                    else:
                        s_mult = max(loss_cut, 1.0 - loss_cut_per_streak * streak_len)
                    risk *= s_mult

        # PnL in dollars
        pnl = t.pnl_r * risk
        equity += pnl
        curve.append(equity)
        if pnl > 0:
            wins += 1

    return curve, wins


# ─────────────────────────────────────────────────────────────────────────────
# Tuner
# ─────────────────────────────────────────────────────────────────────────────

class SymbolTuner:
    """
    Per-symbol parameter optimiser using analytical replay.

    Speed: runs ONE full backtest per CT-threshold value (~5 backtests total),
    then replays the trade list analytically for confidence + streak sizing
    (~36 combos × instant math = <1 second).

    Total time per symbol ≈ 5× one backtest run (e.g. 5 × 8 min = 40 min
    at 1H granularity with 1-year data).
    """

    TUNE_MID_TF = "1H"

    def __init__(
        self,
        config: TradingConfig,
        is_fraction: float = 0.70,
        initial_equity: float = 100_000.0,
        data_dir: Optional[str] = None,
        max_workers: int = 3,
    ):
        self.config = config
        self.is_fraction = is_fraction
        self.initial_equity = initial_equity
        self.data_dir = data_dir
        self.max_workers = max_workers
        # Cached bars from tuning — reusable for the final backtest to ensure
        # data consistency between tuning and validation/production runs.
        self.cached_bars: Dict[str, Tuple[Dict[str, Dict[str, np.ndarray]], str, List[datetime]]] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def tune(self, symbol: str, start_date: str, end_date: str) -> SymbolTuneResult:
        """Run full tuning for a single symbol."""
        logger.info(f"Tuning [{symbol}] {start_date}→{end_date}")

        # Tuning config: coarser mid-TF for speed
        tune_cfg = self.config.model_copy(
            update={"timeframes": self.config.timeframes.model_copy(
                update={"mid": [self.TUNE_MID_TF]})}
        )

        # Pre-load bars ONCE
        base_engine = BacktestEngine(tune_cfg)
        if self.data_dir:
            base_engine.set_data_dir(self.data_dir)
        all_bars, mid_tf, bar_dates = base_engine._preload_bars(
            symbol, start_date, end_date
        )

        if not all_bars or len(bar_dates) < _MIN_BARS + 10:
            logger.error(f"Not enough bars for {symbol}")
            return SymbolTuneResult(
                symbol=symbol, best_params={},
                is_sharpe=0.0, oos_sharpe=0.0,
                is_return_pct=0.0, oos_return_pct=0.0,
                trials_run=0,
            )

        # Cache bars for reuse by the final production backtest
        self.cached_bars[symbol] = (all_bars, mid_tf, bar_dates)

        # IS / OOS date split
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        is_days = int((end_dt - start_dt).days * self.is_fraction)
        is_end = (start_dt + timedelta(days=is_days)).strftime("%Y-%m-%d")

        base_dict = tune_cfg.model_dump()
        base_risk_pct = float(base_dict.get("risk", {}).get("base_risk_pct", 0.20))
        conf_sizing = base_dict.get("confidence_sizing", {})
        streak_cfg  = base_dict.get("streak_sizing", {})
        conf_base   = float(conf_sizing.get("base_conf", 0.45))
        short_penalty = float(conf_sizing.get("short_sizing_penalty", 1.0))
        streak_enabled = bool(streak_cfg.get("enabled", True))
        win_boost   = float(streak_cfg.get("win_boost", 1.15))

        total_trials = 0
        stages: Dict[str, List[TuneTrialResult]] = {}

        # ══════════════════════════════════════════════════════════════════
        # Stage 3 first: CT mid-flip threshold (requires separate backtests)
        # Run one backtest per CT value to get different trade outcomes
        # ══════════════════════════════════════════════════════════════════
        n_workers = min(len(CT_MID_FLIP_GRID), self.max_workers)
        _par_label = f"parallel ×{n_workers}" if n_workers > 1 else "sequential"
        logger.info(
            f"  Stage A: CT mid-flip — running {len(CT_MID_FLIP_GRID)} backtests ({_par_label})"
        )

        def _run_ct_trial(ct_val: float) -> Tuple[float, BacktestResult]:
            trial_dict = copy.deepcopy(base_dict)
            _deep_merge(trial_dict, {"exit_rules": {"ct_mid_flip_threshold": ct_val}})
            trial_config = TradingConfig(**trial_dict)
            eng = BacktestEngine(trial_config)
            if self.data_dir:
                eng.set_data_dir(self.data_dir)
            result = eng.run_with_bars(
                symbol, start_date, end_date,
                all_bars, mid_tf, bar_dates, self.initial_equity,
            )
            return ct_val, result

        ct_backtests: Dict[float, BacktestResult] = {}
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_run_ct_trial, ct): ct for ct in CT_MID_FLIP_GRID}
                for fut in as_completed(futures):
                    ct_val, result = fut.result()
                    ct_backtests[ct_val] = result
                    m = compute_metrics(result)
                    logger.info(
                        f"    ct={ct_val}  trades={m['total_trades']}  "
                        f"Sharpe={m['sharpe']:.3f}  return={m['total_return_pct']:+.2f}%"
                    )
        else:
            for ct_val in CT_MID_FLIP_GRID:
                ct_val, result = _run_ct_trial(ct_val)
                ct_backtests[ct_val] = result
                m = compute_metrics(result)
                logger.info(
                    f"    ct={ct_val}  trades={m['total_trades']}  "
                    f"Sharpe={m['sharpe']:.3f}  return={m['total_return_pct']:+.2f}%"
                )

        # For each CT backtest, split trades into IS / OOS and evaluate
        ct_trials: List[TuneTrialResult] = []
        best_ct_val = CT_MID_FLIP_GRID[0]
        best_ct_sharpe = -999.0
        for ct_val, result in ct_backtests.items():
            is_trades = [t for t in result.trades
                         if t.open_dt is None or
                         t.open_dt < datetime.strptime(is_end, "%Y-%m-%d").replace(tzinfo=timezone.utc)]
            if not is_trades:
                is_trades = result.trades[:int(len(result.trades) * self.is_fraction)]
            is_records = self._trades_to_records(is_trades)
            curve, wins = _replay_sizing(
                is_records, self.initial_equity, base_risk_pct,
                float(conf_sizing.get("floor", 0.70)),
                float(conf_sizing.get("ceil", 1.30)),
                conf_base, short_penalty,
                streak_enabled, win_boost,
                float(streak_cfg.get("loss_cut", 0.80)),
                float(streak_cfg.get("loss_cut_per_streak", 0.10)),
            )
            sharpe = _compute_sharpe(curve)
            ret_pct = (curve[-1] / self.initial_equity - 1) * 100 if curve else 0.0
            trial = TuneTrialResult(
                params={"exit_rules": {"ct_mid_flip_threshold": ct_val}},
                sharpe=sharpe,
                total_return_pct=ret_pct,
                total_trades=len(is_records),
                win_rate=wins / max(len(is_records), 1),
                max_drawdown_pct=_max_drawdown_pct(curve),
            )
            ct_trials.append(trial)
            total_trials += 1
            if sharpe > best_ct_sharpe and len(is_records) >= MIN_TRADES:
                best_ct_sharpe = sharpe
                best_ct_val = ct_val

        stages["ct_mid_flip"] = ct_trials
        logger.info(f"    → best ct_mid_flip={best_ct_val}  IS Sharpe={best_ct_sharpe:.3f}")

        # Use the best-CT backtest's trade list for analytical sizing replay
        ref_result = ct_backtests[best_ct_val]
        all_trades = ref_result.trades

        # Split trades into IS / OOS
        is_end_dt = datetime.strptime(is_end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        is_trades = [t for t in all_trades
                     if t.open_dt is None or t.open_dt < is_end_dt]
        oos_trades = [t for t in all_trades
                      if t.open_dt is not None and t.open_dt >= is_end_dt]
        if not is_trades:
            split_idx = int(len(all_trades) * self.is_fraction)
            is_trades = all_trades[:split_idx]
            oos_trades = all_trades[split_idx:]

        is_records = self._trades_to_records(is_trades)
        oos_records = self._trades_to_records(oos_trades)

        logger.info(
            f"  IS: {len(is_records)} trades  OOS: {len(oos_records)} trades  "
            f"(split at {is_end})"
        )

        # ══════════════════════════════════════════════════════════════════
        # Stage 1: Confidence sizing (analytical replay — instant)
        # ══════════════════════════════════════════════════════════════════
        n_conf = sum(1 for f in CONF_FLOOR_GRID for c in CONF_CEIL_GRID if f <= c)
        logger.info(f"  Stage B: Confidence sizing ({n_conf} combos, analytical)")
        conf_trials: List[TuneTrialResult] = []
        for floor_val in CONF_FLOOR_GRID:
            for ceil_val in CONF_CEIL_GRID:
                if floor_val > ceil_val:
                    continue
                curve, wins = _replay_sizing(
                    is_records, self.initial_equity, base_risk_pct,
                    floor_val, ceil_val, conf_base, short_penalty,
                    streak_enabled, win_boost,
                    float(streak_cfg.get("loss_cut", 0.80)),
                    float(streak_cfg.get("loss_cut_per_streak", 0.10)),
                )
                sharpe = _compute_sharpe(curve)
                ret_pct = (curve[-1] / self.initial_equity - 1) * 100 if curve else 0.0
                trial = TuneTrialResult(
                    params={"confidence_sizing": {"floor": floor_val, "ceil": ceil_val}},
                    sharpe=sharpe,
                    total_return_pct=ret_pct,
                    total_trades=len(is_records),
                    win_rate=wins / max(len(is_records), 1),
                    max_drawdown_pct=_max_drawdown_pct(curve),
                )
                conf_trials.append(trial)
                total_trials += 1

        best_conf = self._pick_best(conf_trials)
        stages["confidence"] = conf_trials
        best_floor = best_conf.params["confidence_sizing"]["floor"]
        best_ceil = best_conf.params["confidence_sizing"]["ceil"]
        logger.info(
            f"    → floor={best_floor}, ceil={best_ceil}  "
            f"Sharpe={best_conf.sharpe:.3f}"
        )

        # ══════════════════════════════════════════════════════════════════
        # Stage 2: Streak sizing (analytical replay — instant)
        # ══════════════════════════════════════════════════════════════════
        n_streak = len(STREAK_LOSS_CUT_GRID) * len(STREAK_LOSS_PER_GRID)
        logger.info(f"  Stage C: Streak sizing ({n_streak} combos, analytical)")
        streak_trials: List[TuneTrialResult] = []
        for lc in STREAK_LOSS_CUT_GRID:
            for lp in STREAK_LOSS_PER_GRID:
                curve, wins = _replay_sizing(
                    is_records, self.initial_equity, base_risk_pct,
                    best_floor, best_ceil, conf_base, short_penalty,
                    streak_enabled, win_boost, lc, lp,
                )
                sharpe = _compute_sharpe(curve)
                ret_pct = (curve[-1] / self.initial_equity - 1) * 100 if curve else 0.0
                trial = TuneTrialResult(
                    params={"streak_sizing": {"loss_cut": lc, "loss_cut_per_streak": lp}},
                    sharpe=sharpe,
                    total_return_pct=ret_pct,
                    total_trades=len(is_records),
                    win_rate=wins / max(len(is_records), 1),
                    max_drawdown_pct=_max_drawdown_pct(curve),
                )
                streak_trials.append(trial)
                total_trials += 1

        best_streak = self._pick_best(streak_trials)
        stages["streak"] = streak_trials
        best_lc = best_streak.params["streak_sizing"]["loss_cut"]
        best_lp = best_streak.params["streak_sizing"]["loss_cut_per_streak"]
        logger.info(
            f"    → loss_cut={best_lc}, per_streak={best_lp}  "
            f"Sharpe={best_streak.sharpe:.3f}"
        )

        # ── Combine best params ──────────────────────────────────────────
        best_params: Dict[str, Any] = {
            "confidence_sizing": {"floor": best_floor, "ceil": best_ceil},
            "streak_sizing": {"loss_cut": best_lc, "loss_cut_per_streak": best_lp},
            "exit_rules": {"ct_mid_flip_threshold": best_ct_val},
        }

        # ── Final validation: REAL engine backtest with best params ────────
        # Analytical replay is used for the search phase only; the final IS/OOS
        # metrics come from an actual engine run to ensure accuracy.
        logger.info(f"  Validation backtest with best params (ct={best_ct_val}) …")
        val_dict = copy.deepcopy(base_dict)
        _deep_merge(val_dict, {"exit_rules": {"ct_mid_flip_threshold": best_ct_val}})
        val_dict["symbol_overrides"] = {
            symbol: {
                "confidence_sizing": {"floor": best_floor, "ceil": best_ceil},
                "streak_sizing": {"loss_cut": best_lc, "loss_cut_per_streak": best_lp},
            }
        }
        val_config = TradingConfig(**val_dict)
        val_engine = BacktestEngine(val_config)
        if self.data_dir:
            val_engine.set_data_dir(self.data_dir)
        val_result = val_engine.run_with_bars(
            symbol, start_date, end_date,
            all_bars, mid_tf, bar_dates, self.initial_equity,
        )
        val_metrics = compute_metrics(val_result)

        # Split equity curve into IS / OOS for Sharpe calculation
        is_end_bar = _find_end_bar(val_result.bar_dates, is_end)
        is_curve_real = val_result.equity_curve[:max(is_end_bar, 2)]
        oos_curve_real = ([val_result.equity_curve[max(is_end_bar - 1, 0)]]
                         + val_result.equity_curve[is_end_bar:])

        is_sharpe = _compute_sharpe(is_curve_real)
        oos_sharpe = _compute_sharpe(oos_curve_real)
        is_ret = (is_curve_real[-1] / is_curve_real[0] - 1) * 100 if len(is_curve_real) > 1 else 0.0
        oos_ret = (oos_curve_real[-1] / oos_curve_real[0] - 1) * 100 if len(oos_curve_real) > 1 else 0.0
        full_sharpe = val_metrics.get("sharpe", 0.0)
        full_ret = val_metrics.get("total_return_pct", 0.0)

        efficiency = oos_sharpe / is_sharpe if is_sharpe > 0.01 else 0.0
        logger.info(
            f"  ── {symbol} tuning complete ──\n"
            f"    Full:  Sharpe={full_sharpe:.3f}  return={full_ret:+.2f}%  "
            f"trades={val_metrics['total_trades']}\n"
            f"    IS:    Sharpe={is_sharpe:.3f}  return={is_ret:+.2f}%\n"
            f"    OOS:   Sharpe={oos_sharpe:.3f}  return={oos_ret:+.2f}%\n"
            f"    efficiency={efficiency:.2f}  trials={total_trials}\n"
            f"    params={json.dumps(best_params, indent=None)}"
        )

        return SymbolTuneResult(
            symbol=symbol,
            best_params=best_params,
            is_sharpe=is_sharpe,
            oos_sharpe=oos_sharpe,
            is_return_pct=is_ret,
            oos_return_pct=oos_ret,
            trials_run=total_trials,
            stages=stages,
        )

    def tune_symbols(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        save_path: Optional[str] = None,
    ) -> Dict[str, SymbolTuneResult]:
        """Tune multiple symbols. Optionally save results to JSON."""
        results: Dict[str, SymbolTuneResult] = {}
        for i, sym in enumerate(symbols, 1):
            logger.info(f"═══ Symbol {i}/{len(symbols)}: {sym} ═══")
            results[sym] = self.tune(sym, start_date, end_date)

        if save_path:
            self.save_results(results, save_path)
        return results

    @staticmethod
    def save_results(results: Dict[str, "SymbolTuneResult"], path: str) -> None:
        """Save tuning results to a JSON that can be reloaded as symbol_overrides."""
        data: Dict[str, Any] = {
            "tuned_at": datetime.now(timezone.utc).isoformat(),
            "symbol_overrides": {},
            "details": {},
        }
        for sym, r in results.items():
            data["symbol_overrides"][sym] = r.best_params
            data["details"][sym] = {
                "is_sharpe": r.is_sharpe,
                "oos_sharpe": r.oos_sharpe,
                "is_return_pct": r.is_return_pct,
                "oos_return_pct": r.oos_return_pct,
                "trials_run": r.trials_run,
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Tuning results saved to {path}")

    @staticmethod
    def load_overrides(path: str) -> Dict[str, Dict[str, Any]]:
        """Load symbol_overrides dict from a previously saved tuning JSON."""
        with open(path) as f:
            data = json.load(f)
        return data.get("symbol_overrides", {})

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _trades_to_records(trades: List[ClosedTrade]) -> List[_TradeRecord]:
        """Convert ClosedTrade list to compact _TradeRecord for replay."""
        return [
            _TradeRecord(
                pnl_r=t.pnl_r,
                confidence=t.confidence,
                is_short=(t.direction == "short"),
                entry_type=getattr(t, "entry_type", "full-alignment"),
                base_risk=t.risk_amount,
                trade_idx=i,
            )
            for i, t in enumerate(trades)
        ]

    @staticmethod
    def _pick_best(trials: List[TuneTrialResult]) -> TuneTrialResult:
        """Pick trial with highest Sharpe (min-trades filter)."""
        valid = [t for t in trials if t.total_trades >= MIN_TRADES]
        if not valid:
            valid = sorted(trials, key=lambda t: t.total_trades, reverse=True)[:1]
        if not valid:
            valid = trials
        return max(valid, key=lambda t: (t.sharpe, t.total_return_pct))
