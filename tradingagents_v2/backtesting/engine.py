"""
Backtesting engine — replays historical bars through the full agent pipeline.

Uses MT5 copy_rates_range() to pre-load all required timeframes, then walks
bar-by-bar on the mid-tier clock to simulate the live trading loop.

No look-ahead guarantee: _BacktestDataLoader._load_bars() always caps the
returned array to bars[:current_step], so feature computation at step t
never sees bar t+1.

Fill model (realistic):
  Entry  — fills at open of next bar + half dynamic spread.
  Stop   — hit when bar's low <= SL (long) or high >= SL (short).
            Fills at SL price.
  TP     — hit when bar's high >= TP (long) or low <= TP (short).
            Fills at TP price.
  Same-bar SL/TP — intra-bar path inferred from OHLC shape:
            bullish (C>=O) → dip-first (O→L→H→C), bearish → spike-first.
            Whichever level is reached first on the inferred path wins.
  Spread — dynamic: base spread × volatility multiplier (bar_range/ATR14,
            capped at 5×).  Models real broker widening during fast moves.
  Slippage — random ±0.3 pip on every fill (mirrors real market micro-noise).
  Spread guard — rejects entry when spread > 20% of stop distance.
  Lot rounding — rounds volume to 0.01 step, clamps to broker min/max.
  Commission — round-trip per-lot cost deducted from every closed trade.
  Swap   — direction-aware overnight financing per symbol; Wed 3× for weekend
            roll.  Weekend charges folded via 7/5 calendar correction.
"""

import asyncio
import bisect
import logging
import random
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple

import sys as _sys_
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None  # progress bar is optional

from ..core.types import TechnicalFeatures, PortfolioState, RiskLimits
from .debug_tracer import get_tracer as _get_tracer
from ..core.graph import TradingGraph
from ..data.loader import DataLoader, _MIN_BARS, _BARS_PER_YEAR
from ..config.settings import TradingConfig

logger = logging.getLogger("Backtest")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimPosition:
    symbol: str
    direction: str          # "long" or "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float         # lots
    risk_amount: float
    open_bar: int
    ticket: int
    confidence: float = 0.0
    win_probability: float = 0.0
    agent_votes: Dict[str, float] = field(default_factory=dict)
    # Pending = True means signal fired on bar N; fill happens at bar N+1 open.
    # This prevents look-ahead entries (can't fill at the bar that triggers signal).
    pending: bool = True
    # Original SL at entry — used as 1R unit for trailing/partial calculations.
    entry_sl: float = 0.0
    # Entry type: "full-alignment", "pullback-entry", "counter-trend-scalp".
    # Counter-trend entries skip D1-flip exits (D1 is expected to be opposed).
    entry_type: str = "full-alignment"
    # Partial TP / windfall fire guards (prevent double-closes).
    partial_tp1_fired: bool = False
    partial_tp2_fired: bool = False
    windfall_fired:    bool = False
    # ATR at entry time — used for ATR-adaptive stale_sl scaling.
    entry_atr: float = 0.0
    # Peak profit seen (in R) — used for stale winner protection.
    peak_profit_r: float = 0.0
    # Global step index (portfolio mode only) — maps to unified_dates for open_dt.
    open_global_step: int = -1


@dataclass
class ClosedTrade:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    risk_amount: float
    pnl: float              # in account currency
    pnl_r: float            # P&L in R units (pnl / risk_amount)
    open_bar: int
    close_bar: int
    exit_reason: str        # "tp", "sl", "eod", "signal"
    confidence: float = 0.0
    win_probability: float = 0.0
    agent_votes: Dict[str, float] = field(default_factory=dict)
    open_dt:  Optional[datetime] = None   # UTC datetime of entry bar
    close_dt: Optional[datetime] = None   # UTC datetime of exit bar


@dataclass
class BacktestResult:
    profile: str
    symbol: str
    start_date: str
    end_date: str
    trades: List[ClosedTrade]
    equity_curve: List[float]
    bar_dates: List[datetime]
    initial_equity: float
    config: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Mock executor — provides the interface TradingGraph expects
# ─────────────────────────────────────────────────────────────────────────────

class _BacktestExecutor:
    """Minimal mock executor for the backtesting loop."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.simulation_mode = True
        self.magic_number = 424242
        self.initialized = True
        self.logger = logging.getLogger("BacktestExecutor")

        self._current_prices: Dict[str, Tuple[float, float]] = {}
        self._open_positions: List[SimPosition] = []
        self._ticket_counter = 1
        self.placed_orders: List[Dict] = []   # cleared by engine each step

    def set_price(self, symbol: str, bid: float, ask: float) -> None:
        self._current_prices[symbol] = (bid, ask)

    def get_current_price(self, symbol: str) -> Optional[Tuple[float, float]]:
        return self._current_prices.get(symbol)

    def get_account_info(self) -> Dict:
        return {
            "balance": 0, "equity": 0, "margin": 0,
            "free_margin": 0, "leverage": 30, "profit": 0,
        }

    def get_symbol_info(self, symbol: str):
        # Return symbol-appropriate tick specs consistent with engine._pip_size()
        # and engine._TICK_VALUE() so that graph.py lot sizing and engine._pnl_at()
        # use identical assumptions — preventing pnl_r > 1.0 at clean SL hits.
        sym = (symbol or "").upper()

        if sym.endswith("JPY"):
            _tick_size  = 0.01       # 1 pip = 0.01 for JPY pairs
            _digits     = 3
            _point      = 0.001
            _tick_value = BacktestEngine._TICK_VALUE.get(sym.rstrip("Mm"), 9.1)
        elif "XAU" in sym or "GOLD" in sym:
            _tick_size  = 1.0
            _digits     = 2
            _point      = 0.01
            _tick_value = BacktestEngine._TICK_VALUE.get(sym.rstrip("Mm"), 1.0)
        elif any(x in sym for x in ("DAX", "UK100", "US30", "US500", "USTEC")):
            _tick_size  = 1.0
            _digits     = 2
            _point      = 1.0
            _tick_value = BacktestEngine._TICK_VALUE.get(sym.rstrip("Mm"), 1.0)
        else:
            # Default: USD-quote pair (EURUSD, GBPUSD, AUDUSD, etc.)
            _tick_size  = 0.0001
            _digits     = 5
            _point      = 0.0001
            _tick_value = BacktestEngine._TICK_VALUE.get(sym.rstrip("Mm"), 10.0)

        class _SI:
            spread               = 3
            point                = _point
            digits               = _digits
            trade_tick_size      = _tick_size
            trade_tick_value     = _tick_value
            volume_min           = 0.01
            volume_max           = 1000.0
            volume_step          = 0.01
            trade_contract_size  = 100_000
            trade_stops_level    = 0
            trade_mode           = 4
            filling_mode         = 1
        return _SI()

    def place_bracket_order(self, trade_plan, entry_price=None,
                            stop_loss=None, take_profit=None) -> Optional[Dict]:
        # Spread guard — mirrors MT5Executor.place_bracket_order
        _ep     = entry_price or trade_plan.entry_price
        _sl     = stop_loss   or trade_plan.stop_loss
        _tp     = take_profit or trade_plan.take_profit
        _max_sf = float(self.config.get("mt5", {}).get("max_spread_fraction", 0.20))
        if _max_sf > 0 and _sl and _ep:
            _stop_dist = abs(_ep - _sl)
            if _stop_dist > 1e-9:
                _cur = self._current_prices.get(trade_plan.symbol if hasattr(trade_plan, "symbol") else "")
                if _cur:
                    _spread_price = abs(_cur[1] - _cur[0])  # ask - bid
                    if _spread_price / _stop_dist > _max_sf:
                        return None  # spread too wide — reject
        ticket = self._ticket_counter
        self._ticket_counter += 1
        order = {
            "order_ticket": ticket,
            "trade_plan": trade_plan,
            "entry_price": _ep,
            "stop_loss":   _sl,
            "take_profit": _tp,
        }
        self.placed_orders.append(order)
        return order

    def close_position(self, ticket: int) -> bool:
        return True

    def modify_stop_loss(self, ticket: int, sl: float) -> bool:
        for pos in self._open_positions:
            if pos.ticket == ticket:
                pos.stop_loss = sl
                return True
        return False

    def modify_sl_tp(self, ticket: int, sl: float, tp: float) -> bool:
        for pos in self._open_positions:
            if pos.ticket == ticket:
                pos.stop_loss = sl
                pos.take_profit = tp
                return True
        return False

    def get_open_positions(self) -> List[Dict]:
        return [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.direction == "long" else "SELL",
                "profit": 0.0,
                "price_open": p.entry_price,
                "price_current": self._current_prices.get(
                    p.symbol, (p.entry_price, p.entry_price)
                )[0],
                "sl": p.stop_loss,
                "tp": p.take_profit,
                "magic": self.magic_number,
                "open_time": 0,
                "volume": p.quantity,
            }
            for p in self._open_positions
        ]

    def get_closed_trades(self, days: int = 30) -> List[Dict]:
        return []

    def copy_rates(self, symbol: str, timeframe: str, n: int) -> Optional[List]:
        return None  # BacktestDataLoader overrides _load_bars; this is never called

    def get_tf_map(self) -> Dict:
        return {}

    def shutdown(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Bar-capped DataLoader  (the no-look-ahead guarantee)
# ─────────────────────────────────────────────────────────────────────────────

class _BacktestDataLoader(DataLoader):
    """
    DataLoader that returns only pre-loaded bars up to _current_step.

    _all_bars format: {"{symbol}_{tf}": {"close": ndarray, "high": ndarray, ...}}
    _current_step is the exclusive upper bound — bar at index _current_step is
    the bar that just closed and triggered the agent run.  Features are computed
    from bars[:_current_step], giving no look-ahead.
    """

    def __init__(
        self,
        all_bars: Dict[str, Dict[str, np.ndarray]],
        tier_tf_map: Dict[str, str],
        macro_history: Optional[Dict] = None,
    ):
        # Pass simulation=True to skip MT5 connection in the base class,
        # then immediately override it to False so the feature pipeline
        # uses our pre-loaded real bars instead of the synthetic fallback.
        # Suppress the misleading "running in simulation mode" WARNING that
        # fires inside DataLoader.__init__ before we can override the flag.
        import logging as _logging
        _dl_log = _logging.getLogger("DataLoader")
        _prev = _dl_log.level
        _dl_log.setLevel(_logging.ERROR)
        super().__init__(simulation=True)
        _dl_log.setLevel(_prev)

        self._all_bars = all_bars
        self._tier_tf_map = tier_tf_map
        self._current_step = _MIN_BARS
        self._current_bar_ts: Optional[float] = None  # unix timestamp of current bar
        # Override: feature pipeline must NOT use synthetic bars
        self.simulation = False
        # Pre-loaded historical macro signals, keyed by datetime.date.
        # When populated, _macro_features() returns the bar-date-aligned value
        # instead of neutral zeros.  Sorted list for O(log n) bisect lookup.
        self._macro_history: Dict = macro_history or {}
        self._macro_dates: List = sorted(self._macro_history.keys()) if self._macro_history else []

    def _load_bars(self, symbol: str, tf: str, n: int) -> Optional[Dict]:
        key = f"{symbol}_{tf}"
        bars = self._all_bars.get(key)
        if bars is None:
            return None
        # Use the bar timestamp to derive the correct end-index for each
        # timeframe independently.  Without this, _current_step (a mid_tf
        # index) is used raw on all arrays — at step 500 of 4H bars, the
        # 15m array (8,000+ bars) gets sliced to [:500] instead of [:8,000],
        # so short-tier agents see bars from the start of the dataset, not
        # the most-recent window.  Longer TFs (D1) happen to work because
        # len(D1_bars) < _current_step keeps them fully visible.
        if self._current_bar_ts is not None and "time" in bars:
            # bisect_right: first index whose timestamp is STRICTLY after current bar
            end = bisect.bisect_right(bars["time"], self._current_bar_ts)
        else:
            end = min(self._current_step, len(bars["close"]))
        start = max(0, end - n)
        # Use a per-timeframe minimum-bars check.
        # _MIN_BARS=250 is designed for 1m/15m where indicators like EMA(200)
        # need hundreds of bars.  For 4H and 1D, fewer bars are available by
        # construction (30-day warmup = ~120 4H bars, ~22 D1 bars) but the
        # indicators are still meaningful once we have at least 50 bars.
        _tf_min_bars = {
            "1D": 20, "1W": 10, "4H": 40, "1H": _MIN_BARS,
        }.get(tf, _MIN_BARS)
        if end - start < _tf_min_bars:
            return None
        return {k: v[start:end] for k, v in bars.items()}

    def get_multi_features(self, symbol: str) -> Dict[str, Optional[TechnicalFeatures]]:
        result = {}
        for tier, tf in self._tier_tf_map.items():
            result[tier] = self.get_features(symbol, tf, n_bars=300)
        # Always expose 1H features under the raw '1H' key so that plan creation
        # can use 1H ATR for SL/TP sizing regardless of the --mid-tf override.
        # Without this, --mid-tf 1m would use 1m ATR (~3-5 pips) for a D1 signal
        # trade, producing SLs too tight for normal intraday noise.
        if "1H" not in self._tier_tf_map.values():
            h1 = self.get_features(symbol, "1H", n_bars=300)
            if h1 is not None:
                result["1H"] = h1
        # Log feature summary per tier
        for _tier, _feat in result.items():
            if _feat is not None:
                self.logger.debug(
                    f"[FEATURES] {symbol} {_tier}({self._tier_tf_map.get(_tier, '?')}) "
                    f"close={getattr(_feat, 'ema20', 0):.5f} "
                    f"rsi14={getattr(_feat, 'rsi_14', 0):.2f} "
                    f"atr14={getattr(_feat, 'atr_14', 0):.6f} "
                    f"adx={getattr(_feat, 'adx_14', 0):.1f} "
                    f"session_break={getattr(_feat, 'session_break_score', 0):.3f}"
                )
        return result

    # Macro features in backtest: use pre-loaded historical values when available.
    # Falls back to neutral zeros if macro_history was not pre-loaded (e.g. yfinance
    # unavailable).  No look-ahead: each bar date D only sees macro data from D or earlier.
    def _macro_features(self) -> Dict[str, float]:
        _null = {"dxy_trend": 0.0, "vix_signal": 0.0, "crude_trend": 0.0, "yield_trend": 0.0}
        if not self._macro_history or self._current_bar_ts is None:
            return _null
        from datetime import timezone as _tz
        bar_date = datetime.fromtimestamp(int(self._current_bar_ts), tz=_tz.utc).date()
        # Find the most recent macro date <= bar_date (handles weekends / market holidays)
        idx = bisect.bisect_right(self._macro_dates, bar_date) - 1
        if idx < 0:
            return _null
        return self._macro_history.get(self._macro_dates[idx], _null)


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Single-symbol walk-through backtester that replays the full agent pipeline.

    Usage::

        from tradingagents_v2.backtesting import BacktestEngine
        from tradingagents_v2.config.yaml_config import load_config_from_yaml

        cfg = load_config_from_yaml("config.yaml", profile="balanced")
        engine = BacktestEngine(cfg)
        result = engine.run("EURUSD", "2025-01-01", "2026-01-01")
    """

    # ── Spread model ──────────────────────────────────────────────────────
    # Base spread: typical "good conditions" spread for each symbol (price units).
    # Dynamic widening is applied per-bar based on volatility relative to the
    # 14-bar ATR median.  See _dynamic_spread().
    _BASE_SPREAD: Dict[str, float] = {
        "EURUSD": 0.00004, "GBPUSD": 0.00005, "USDJPY": 0.003,
        "USDCHF": 0.00005, "AUDUSD": 0.00005, "USDCAD": 0.00005,
        "NZDUSD": 0.00007, "XAUUSD": 0.25,   "EURJPY": 0.006,
        "GBPJPY": 0.008,   "DAX": 1.5,        "UK100": 1.0,
        "US30": 3.0,       "US500": 0.50,     "USTEC": 2.0,
    }
    # Maximum spread multiplier during volatile bars (base × cap)
    _SPREAD_WIDEN_CAP: float = 5.0

    # ── Commission: USD per standard lot per round-trip ────────────────────
    # Typical ECN FX broker charges ~$7/lot.  Indices/metals vary.
    _COMMISSION_PER_LOT: Dict[str, float] = {
        "EURUSD": 7.0,  "GBPUSD": 7.0,  "USDJPY": 7.0,
        "USDCHF": 7.0,  "AUDUSD": 7.0,  "USDCAD": 7.0,
        "NZDUSD": 7.0,  "EURJPY": 7.0,  "GBPJPY": 7.0,
        "AUDJPY": 7.0,  "EURGBP": 7.0,  "XAUUSD": 5.0,
        "DAX": 3.0,     "UK100": 3.0,   "US30":  3.0,
        "US500": 3.0,   "USTEC": 3.0,
    }

    # USD account P&L per pip per standard lot (10 = $10/pip for most majors)
    _TICK_VALUE: Dict[str, float] = {
        "EURUSD": 10.0, "GBPUSD": 10.0, "AUDUSD": 10.0, "NZDUSD": 10.0,
        "USDCHF": 10.0, "USDCAD": 10.0, "USDJPY":  9.1, "EURJPY":  9.1,
        "GBPJPY":  9.1, "AUDJPY":  9.1, "XAUUSD":  1.0, "XAGUSD":  5.0,
        "DAX": 1.0,     "UK100": 1.0,   "US30":    1.0, "US500":   1.0,
        "USTEC": 1.0,
    }

    # ── Swap rates: (long, short) in USD per lot per night ─────────────────
    # Negative = trader pays, positive = trader receives.
    # Based on typical ECN broker rates.  Unknown symbols default to (-3.0, -1.0).
    _SWAP_RATE: Dict[str, tuple] = {
        "EURUSD": (-5.0, +0.8),  "GBPUSD": (-4.5, +0.5),
        "USDJPY": (+2.0, -5.5),  "USDCHF": (+1.5, -4.8),
        "AUDUSD": (-3.2, -0.5),  "USDCAD": (+1.0, -4.2),
        "NZDUSD": (-2.8, -0.8),  "EURJPY": (-1.5, -3.0),
        "GBPJPY": (-0.5, -3.5),  "AUDJPY": (+1.0, -4.0),
        "EURGBP": (-3.0, +0.5),  "XAUUSD": (-8.0, +2.0),
        "DAX":    (-1.5, -0.5),  "UK100":  (-1.2, -0.3),
        "US30":   (-2.0, -0.5),  "US500":  (-1.5, -0.3),
        "USTEC":  (-2.0, -0.5),
    }

    # ── Volume constraints (mirrors MT5 symbol_info) ──────────────────────
    _VOL_MIN:  float = 0.01
    _VOL_MAX:  float = 100.0
    _VOL_STEP: float = 0.01

    # ── Execution realism ─────────────────────────────────────────────────
    _SLIPPAGE_PIPS: float = 0.3     # random ±slippage on every fill
    _MAX_SPREAD_FRAC: float = 0.20  # reject entry when spread > 20% of stop dist

    # ── yfinance backfill constants ────────────────────────────────────────
    # Broker symbol → yfinance ticker translation
    _YF_SYMBOL: Dict[str, str] = {
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
        "USDCHF": "USDCHF=X", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X",
        "NZDUSD": "NZDUSD=X", "EURJPY": "EURJPY=X", "GBPJPY": "GBPJPY=X",
        "AUDJPY": "AUDJPY=X", "CADJPY": "CADJPY=X", "CHFJPY": "CHFJPY=X",
        "EURGBP": "EURGBP=X", "EURCHF": "EURCHF=X", "EURCAD": "EURCAD=X",
        "XAUUSD": "GC=F",     "XAGUSD": "SI=F",
        "DAX":    "^GDAXI",   "UK100":  "^FTSE",    "US30":   "^DJI",
        "US500":  "^GSPC",    "USTEC":  "^IXIC",
    }
    # TradingAgents TF string → yfinance interval string
    # Note: 4H is not a native yfinance interval; 1h bars are downloaded and
    # resampled to 4H inside _preload_bars_yfinance.
    _YF_INTERVAL: Dict[str, str] = {
        "1m":  "1m",  "5m":  "5m",  "15m": "15m", "30m": "30m",
        "1H":  "1h",  "4H":  "1h",  "1D":  "1d",  "1W":  "1wk",
    }
    # Maximum calendar days of history yfinance reliably returns per interval.
    # Conservative estimates (actual limits are 7 / 60 / 730 days).
    _YF_MAX_DAYS: Dict[str, int] = {
        "1m": 6, "5m": 58, "15m": 58, "30m": 58,
        "1h": 725, "1d": 99999, "1wk": 99999,
    }

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger("BacktestEngine")
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._n_portfolio_symbols: int = 1
        self._data_dir: Optional[str] = None  # set via set_data_dir() for CSV/Parquet loading

    def set_data_dir(self, path: str) -> None:
        """Set the directory containing CSV/Parquet bar files for offline loading."""
        self._data_dir = path

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_equity: float = 100_000.0,
    ) -> BacktestResult:
        """Run a single-symbol backtest and return a BacktestResult."""
        self.logger.info(
            f"Backtest [{symbol}] {start_date}→{end_date} "
            f"(profile={self.config.profile})"
        )

        all_bars, mid_tf, bar_dates = self._preload_bars(symbol, start_date, end_date)
        if not all_bars or len(bar_dates) < _MIN_BARS + 10:
            self.logger.error(f"Not enough historical bars for {symbol}")
            return BacktestResult(
                profile=self.config.profile, symbol=symbol,
                start_date=start_date, end_date=end_date,
                trades=[], equity_curve=[initial_equity],
                bar_dates=bar_dates or [], initial_equity=initial_equity,
                config=self.config.model_dump(),
            )
        return self._run_from_bars(
            symbol, start_date, end_date, all_bars, mid_tf, bar_dates, initial_equity
        )

    def run_with_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        all_bars: Dict[str, Dict[str, np.ndarray]],
        mid_tf: str,
        bar_dates: "List[datetime]",
        initial_equity: float = 100_000.0,
    ) -> "BacktestResult":
        """Run backtest with externally pre-loaded bars (skips MT5 connection).

        Used by WalkForwardValidator to avoid repeated MT5 connect/disconnect
        cycles and to guarantee warm-up bars are available for every window.
        """
        if not all_bars or len(bar_dates) < _MIN_BARS + 10:
            self.logger.error(
                f"Not enough historical bars for {symbol} "
                f"(got {len(bar_dates)}, need {_MIN_BARS + 10})"
            )
            return BacktestResult(
                profile=self.config.profile, symbol=symbol,
                start_date=start_date, end_date=end_date,
                trades=[], equity_curve=[initial_equity],
                bar_dates=bar_dates, initial_equity=initial_equity,
                config=self.config.model_dump(),
            )
        return self._run_from_bars(
            symbol, start_date, end_date, all_bars, mid_tf, bar_dates, initial_equity
        )

    def _run_from_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        all_bars: Dict[str, Dict[str, np.ndarray]],
        mid_tf: str,
        bar_dates: "List[datetime]",
        initial_equity: float,
    ) -> "BacktestResult":
        """Build graph and run the async backtest core. Shared by run() and run_with_bars()."""
        # Use the mid_tf parameter (which reflects yfinance degradation) rather
        # than the original config value.  When 30m→1H degradation occurs, the
        # bar data is stored under the degraded key (EURUSD_1H); if tier_tf_map
        # still points to "30m" the data loader finds nothing → 0 trades.
        tier_tf_map = {
            "long":  self.config.timeframes.long[0]  if self.config.timeframes.long  else "1D",
            "mid":   mid_tf,
            "short": self.config.timeframes.short[0] if self.config.timeframes.short else "15m",
        }

        executor = _BacktestExecutor(self.config.model_dump())
        macro_history = self._load_macro_history(start_date, end_date)
        loader   = _BacktestDataLoader(all_bars, tier_tf_map, macro_history=macro_history)

        from ..agents import (
            RegimeAgent, TrendAgent, MomentumAgent, MeanReversionAgent,
            VolatilityAgent, BreadthAgent, PatternAgent, IntermarketAgent,
            SessionBreakoutAgent, DivergenceAgent, ScalpingAgent,
            VwapScalpAgent, SqueezeBreakoutAgent, OrderFlowAgent,
            CorrelationAgent,
        )
        from ..runner import _build_registry
        registry = _build_registry(self.config)
        graph = TradingGraph(
            registry, self.config.model_dump(),
            executor=executor, data_loader=loader,
        )

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self._run_async(
                    symbol, all_bars, mid_tf, bar_dates,
                    initial_equity, executor, loader, graph, tier_tf_map,
                    start_date, end_date,
                )
            )
        finally:
            loop.close()

        return result

    def run_multi(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_equity: float = 100_000.0,
    ) -> "List[BacktestResult]":
        """Portfolio-level backtest: all symbols trade simultaneously.

        Shared across symbols:
          - Equity curve (trades in GBPUSD affect EURUSD sizing)
          - Daily/weekly DD circuit breakers
          - max_concurrent_trades and max_correlated_positions
          - Weekend close-all

        Per-symbol:
          - Agent pipeline, features, data loader
          - Entry cooldowns, streak guards
        """
        if len(symbols) == 1:
            self._n_portfolio_symbols = 1
            return [self.run(symbols[0], start_date, end_date, initial_equity)]

        self._n_portfolio_symbols = len(symbols)
        self.logger.info(
            f"Portfolio backtest: {len(symbols)} symbols simultaneously"
        )

        # ── Pre-load bars for all symbols ─────────────────────────────────
        all_bars: Dict[str, Dict[str, np.ndarray]] = {}
        bar_dates_by_sym: Dict[str, List[datetime]] = {}
        mid_tf: Optional[str] = None
        valid_symbols: List[str] = []

        for sym in symbols:
            sym_bars, sym_mid_tf, sym_dates = self._preload_bars(sym, start_date, end_date)
            if not sym_bars or len(sym_dates) < _MIN_BARS + 10:
                self.logger.warning(f"Not enough bars for {sym} — skipping")
                continue
            all_bars.update(sym_bars)
            bar_dates_by_sym[sym] = sym_dates
            valid_symbols.append(sym)
            if mid_tf is None:
                mid_tf = sym_mid_tf

        if not valid_symbols or mid_tf is None:
            self.logger.error("No valid symbols for portfolio backtest")
            self._n_portfolio_symbols = 1
            return []

        # ── Build unified timeline (union of all bar timestamps) ──────────
        all_ts_set: set = set()
        for dates in bar_dates_by_sym.values():
            all_ts_set.update(dates)
        unified_dates = sorted(all_ts_set)

        # Build per-symbol timestamp→index map for O(1) bar lookup
        sym_ts_to_idx: Dict[str, Dict[float, int]] = {}
        for sym in valid_symbols:
            mid_key = f"{sym}_{mid_tf}"
            bars = all_bars.get(mid_key)
            if bars is not None and "time" in bars:
                sym_ts_to_idx[sym] = {
                    float(bars["time"][i]): i
                    for i in range(len(bars["time"]))
                }
            else:
                sym_ts_to_idx[sym] = {}

        # ── Build per-symbol graph + loader + executor ────────────────────
        tf_cfg = self.config.timeframes
        tier_tf_map = {
            "long":  tf_cfg.long[0]  if tf_cfg.long  else "1D",
            "mid":   mid_tf,
            "short": tf_cfg.short[0] if tf_cfg.short else "15m",
        }
        macro_history = self._load_macro_history(start_date, end_date)

        from ..agents import (
            RegimeAgent, TrendAgent, MomentumAgent, MeanReversionAgent,
            VolatilityAgent, BreadthAgent, PatternAgent, IntermarketAgent,
            SessionBreakoutAgent, DivergenceAgent, ScalpingAgent,
            VwapScalpAgent, SqueezeBreakoutAgent, OrderFlowAgent,
            CorrelationAgent,
        )
        from ..runner import _build_registry

        sym_ctx: Dict[str, Dict] = {}
        for sym in valid_symbols:
            loader = _BacktestDataLoader(all_bars, tier_tf_map, macro_history=macro_history)
            executor = _BacktestExecutor(self.config.model_dump())
            registry = _build_registry(self.config)
            graph = TradingGraph(
                registry, self.config.model_dump(),
                executor=executor, data_loader=loader,
            )
            sym_ctx[sym] = {
                "loader": loader,
                "executor": executor,
                "graph": graph,
                "mid_bars": all_bars[f"{sym}_{mid_tf}"],
                "bar_dates": bar_dates_by_sym[sym],
                "tick_val": self._tick_value(sym),
                "pip_sz": self._pip_size(sym),
                "base_spread": self._spread(sym),
                "last_agent_state": None,
                "last_surv_state": None,
                "last_entry": {},
                "last_order_attempt": {},
                "streak_history": [],
                "streak_block_dir": None,
                "streak_block_until": 0,
                "streak_block_count": 0,
            }

        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(
                self._run_async_portfolio(
                    valid_symbols, all_bars, mid_tf, unified_dates,
                    initial_equity, sym_ctx, sym_ts_to_idx,
                    tier_tf_map, start_date, end_date,
                )
            )
        finally:
            loop.close()

        self._n_portfolio_symbols = 1
        return results

    def _run_multi_independent(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_equity: float = 100_000.0,
    ) -> "List[BacktestResult]":
        """Legacy independent-symbol backtest (each symbol gets its own equity)."""
        self._n_portfolio_symbols = len(symbols)
        results = [self.run(s, start_date, end_date, initial_equity) for s in symbols]
        self._n_portfolio_symbols = 1
        return results

    # ── Portfolio-level async core ────────────────────────────────────────────

    async def _run_async_portfolio(
        self,
        symbols: List[str],
        all_bars: Dict[str, Dict[str, np.ndarray]],
        mid_tf: str,
        unified_dates: List[datetime],
        initial_equity: float,
        sym_ctx: Dict[str, Dict],
        sym_ts_to_idx: Dict[str, Dict[float, int]],
        tier_tf_map: Dict[str, str],
        start_date: str,
        end_date: str,
    ) -> "List[BacktestResult]":
        """Interleaved multi-symbol backtest with shared equity and DD tracking.

        Walks a unified timeline (union of all symbols' bar timestamps).
        At each timestamp, for each symbol that has a bar:
          1. Fill pending entries
          2. Check SL/TP exits
          3. Run surveillance (trailing SL, partial TP, time-stop, etc.)
          4. Run signal pipeline (agents → entry decision)

        Shared state: equity, DD circuit breakers, position count, weekend close.
        Per-symbol state: agent pipeline, features, cooldowns, streak guards.
        """
        n_bars_total = len(unified_dates)
        if n_bars_total == 0:
            return []

        # ── Deterministic slippage: per-symbol RNG for reproducible backtests
        # Using per-symbol Random instances ensures that each symbol gets the
        # exact same slippage sequence whether running solo or in a portfolio.
        # A shared global RNG would interleave calls across symbols, making
        # fills non-reproducible when the symbol set changes.
        _sym_rng: Dict[str, random.Random] = {}
        for _i, _s in enumerate(symbols):
            _sym_rng[_s] = random.Random(42 + _i)

        # ── Pre-compute config dicts once ────────────────────────────────
        _cfg_raw       = self.config.model_dump()
        _trailing_cfg  = _cfg_raw.get("trailing", {})
        _exit_cfg_raw  = _cfg_raw.get("exit_rules", {})
        _al_cfg        = _cfg_raw.get("alignment", {})
        _rt_cfg        = _cfg_raw.get("realtime", {})

        _partial1_on   = bool(_trailing_cfg.get("partial_tp_enabled", False))
        _partial1_frac = float(_trailing_cfg.get("partial_tp_fraction", 0.50))
        _partial2_on   = bool(_trailing_cfg.get("partial_tp2_enabled", False))
        _partial2_r    = float(_trailing_cfg.get("partial_tp2_r_mult", 2.0))
        _partial2_frac = float(_trailing_cfg.get("partial_tp2_fraction", 0.50))
        _windfall_on   = bool(_trailing_cfg.get("windfall_exit_enabled", False))
        _windfall_r    = float(_trailing_cfg.get("windfall_r_mult", 3.0))
        _tp_extend_on  = bool(_trailing_cfg.get("tp_extend_enabled", True))
        _tp_extend_mult = float(_trailing_cfg.get("tp_extend_atr_mult", 2.0))
        _early_be_r    = float(_trailing_cfg.get("early_be_r", 0.0))
        _be_buffer_r   = float(_trailing_cfg.get("be_buffer_r", 0.0))
        _stale_sl_hours = float(_trailing_cfg.get("stale_sl_hours", 0.0))
        _stale_sl_r    = float(_trailing_cfg.get("stale_sl_r_mult", 0.75))
        _atr_mult      = float(_trailing_cfg.get("atr_multiplier", 1.5))
        _max_hours_ts  = float(_exit_cfg_raw.get("max_trade_duration_hours", 0))
        _fade_enabled  = bool(_exit_cfg_raw.get("conviction_fade_enabled", False))
        _fade_thresh   = float(_exit_cfg_raw.get("conviction_fade_threshold", 0.10))
        _opp_enabled   = bool(_exit_cfg_raw.get("mid_short_opposition_enabled", True))
        _opp_thresh    = float(_exit_cfg_raw.get("mid_short_opposition_threshold", 0.35))
        _tighten_on    = bool(_exit_cfg_raw.get("tighten_on_fade_enabled", True))
        _tighten_thresh = float(_exit_cfg_raw.get("tighten_fade_threshold", 0.20))
        _tighten_tp_m  = float(_exit_cfg_raw.get("tighten_tp_atr_mult", 0.5))
        _flip_thresh   = float(_exit_cfg_raw.get("d1_flip_threshold",
                                                  _al_cfg.get("long_min_score", 0.25)))
        _scalp_exits   = bool(_exit_cfg_raw.get("use_short_tier_exits", False))
        _scalp_flip    = float(_exit_cfg_raw.get("short_flip_threshold", 0.35))
        _ct_max_hours  = float(_exit_cfg_raw.get("ct_max_hours", 2.0))
        _skip_struct_ratchet = bool(_exit_cfg_raw.get("skip_structural_sl_tp_ratchet", False))

        _tf_hours_map = {"1m": 1/60, "5m": 5/60, "15m": 0.25, "30m": 0.5,
                         "1H": 1.0, "4H": 4.0, "1D": 24.0}
        _hrs_per_bar  = _tf_hours_map.get(mid_tf, 1.0)
        self._hrs_per_bar = _hrs_per_bar

        _wkend_enabled = bool(_rt_cfg.get("weekend_close_enabled", False))
        _wkend_hour    = int(_rt_cfg.get("weekend_close_utc_hour", 20))
        _dd_pct        = self.config.risk.max_daily_drawdown_pct
        _wk_dd_pct     = float(_cfg_raw.get("risk", {}).get("max_weekly_drawdown_pct", 0))
        _leverage      = float(_cfg_raw.get("risk", {}).get("leverage", 30))

        _bar_seconds_map = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
                            "1H": 3600, "4H": 14400, "1D": 86400}
        _bar_seconds   = _bar_seconds_map.get(mid_tf, 3600)
        _sig_interval_s = int(_cfg_raw.get("interval_seconds", None)
                              or getattr(self.config, "interval_seconds", 300) or 300)
        _agent_interval = max(1, round(_sig_interval_s / _bar_seconds))

        _surv_explicit = _cfg_raw.get("realtime", {}).get("surveillance_interval_seconds")
        if _surv_explicit is not None:
            _surv_seconds = int(_surv_explicit)
        else:
            _surv_by_tf = {"1m": 10, "5m": 15, "15m": 30, "30m": 45,
                           "1H": 60, "4H": 90, "1D": 120}
            _surv_seconds = _surv_by_tf.get(mid_tf, 60)
        _surv_interval = max(1, round(_surv_seconds / _bar_seconds))

        _cooldown_base = float(getattr(self.config.risk, "entry_cooldown_minutes", 0.0))
        _cooldown_scale = {"1m": 0.25, "5m": 0.50, "15m": 0.75, "30m": 0.85}.get(mid_tf, 1.0)
        _cooldown_min  = max(5.0, _cooldown_base * _cooldown_scale)
        _max_daily     = int(getattr(self.config.risk, "max_daily_trades", 0))

        _streak_base_bars = max(1, int(4 * 3600 / _bar_seconds))
        _streak_max_bars  = max(1, int(24 * 3600 / _bar_seconds))

        _news_atr_mult = float(_cfg_raw.get("execution", {}).get("news_spike_atr_mult", 3.0))
        _news_blackout_enabled = bool(_cfg_raw.get("execution", {}).get("news_blackout_minutes", 0) > 0)

        # ── Shared state ──────────────────────────────────────────────────
        equity = initial_equity
        equity_curve: List[float] = []
        equity_dates: List[datetime] = []
        open_positions: List[SimPosition] = []   # ALL symbols
        closed_trades: List[ClosedTrade] = []    # ALL symbols
        self._prev_closed_count_pf = 0           # reset for each portfolio run
        _day_open_equity = equity
        _day_halted = False
        _current_day: Optional[int] = None
        _wk_open_equity = equity
        _wk_halted = False
        _current_week: Optional[int] = None
        _day_trade_count = 0
        _weekend_blocked = False
        # Per-symbol equity curves: track each symbol's contribution separately
        # so that per-symbol metrics (return%, maxDD, Sharpe) reflect that
        # symbol's trades, not the whole portfolio.  Each curve = initial_equity
        # + cumulative realized P&L for that symbol + unrealized P&L for that
        # symbol's open positions.
        _sym_equity_curve: Dict[str, List[float]] = {s: [] for s in symbols}
        _sym_realized: Dict[str, float] = {s: 0.0 for s in symbols}
        risk_limits = RiskLimits(
            base_risk_pct=self.config.risk.base_risk_pct,
            max_daily_drawdown_pct=self.config.risk.max_daily_drawdown_pct,
            max_concurrent_trades=self.config.risk.max_concurrent_trades,
        )

        # Per-symbol step counters (for agent_interval / surv_interval)
        _sym_step: Dict[str, int] = {s: 0 for s in symbols}

        # Signal start: skip bars before start_date
        _start_dt = None
        if start_date:
            try:
                _start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                pass

        # Wait until each symbol has enough warmup bars
        _sym_warmup_done: Dict[str, bool] = {s: False for s in symbols}

        # Suppress graph logs during backtest
        _graph_logger = logging.getLogger("TradingGraph")
        _graph_orig_level = _graph_logger.level
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
            _graph_logger.setLevel(logging.WARNING)

        self.logger.info(
            f"[portfolio] {len(symbols)} symbols, {n_bars_total} unified bars, "
            f"mid_tf={mid_tf}, agent_interval={_agent_interval}, "
            f"surv_interval={_surv_interval}"
        )

        _is_tty = getattr(_sys_.stderr, "isatty", lambda: False)()
        _bar_range = range(n_bars_total)
        if _tqdm is not None and _is_tty:
            _progress = _tqdm(_bar_range, desc="Portfolio backtest",
                              unit="bar", leave=True, dynamic_ncols=True)
        else:
            _progress = _bar_range
        _report_every = max(100, n_bars_total // 20)

        for global_step in _progress:
            bar_date = unified_dates[global_step]
            bar_ts = bar_date.timestamp()

            # Record equity at top of bar
            equity_curve.append(equity)
            equity_dates.append(bar_date)

            # ── Daily / weekly guard reset ────────────────────────────────
            if _current_day != bar_date.day:
                _current_day = bar_date.day
                _day_open_equity = equity
                _day_halted = False
                _day_trade_count = 0
            if not _day_halted and _dd_pct > 0:
                day_dd = (equity - _day_open_equity) / max(_day_open_equity, 1e-9) * 100
                if day_dd <= -_dd_pct:
                    _day_halted = True
                    for _ddpos in list(open_positions):
                        if not _ddpos.pending:
                            _tv = sym_ctx[_ddpos.symbol]["tick_val"]
                            _ps = sym_ctx[_ddpos.symbol]["pip_sz"]
                            _mb = sym_ctx[_ddpos.symbol]["mid_bars"]
                            _si = sym_ts_to_idx[_ddpos.symbol].get(bar_ts)
                            _cl = float(_mb["close"][_si]) if _si is not None else _ddpos.entry_price
                            _ddpnl = self._pnl_at(_ddpos, _cl, _tv, _ps, close_step=_si or 0)
                            closed_trades.append(ClosedTrade(
                                symbol=_ddpos.symbol, direction=_ddpos.direction,
                                entry_price=_ddpos.entry_price, exit_price=_cl,
                                stop_loss=_ddpos.stop_loss, take_profit=_ddpos.take_profit,
                                quantity=_ddpos.quantity, risk_amount=_ddpos.risk_amount,
                                pnl=_ddpnl, pnl_r=_ddpnl / max(_ddpos.risk_amount, 1e-9),
                                open_bar=_ddpos.open_bar, close_bar=global_step,
                                exit_reason="dd_close_all", confidence=_ddpos.confidence,
                                win_probability=_ddpos.win_probability,
                                agent_votes=_ddpos.agent_votes,
                            ))
                    open_positions.clear()

            _iso_week = bar_date.isocalendar()[1]
            if _current_week != _iso_week:
                _current_week = _iso_week
                _wk_open_equity = equity
                _wk_halted = False
            if not _wk_halted and _wk_dd_pct > 0:
                wk_dd = (equity - _wk_open_equity) / max(_wk_open_equity, 1e-9) * 100
                if wk_dd <= -_wk_dd_pct:
                    _wk_halted = True
                    for _wkpos in list(open_positions):
                        if not _wkpos.pending:
                            _tv = sym_ctx[_wkpos.symbol]["tick_val"]
                            _ps = sym_ctx[_wkpos.symbol]["pip_sz"]
                            _mb = sym_ctx[_wkpos.symbol]["mid_bars"]
                            _si = sym_ts_to_idx[_wkpos.symbol].get(bar_ts)
                            _cl = float(_mb["close"][_si]) if _si is not None else _wkpos.entry_price
                            _wkpnl = self._pnl_at(_wkpos, _cl, _tv, _ps, close_step=_si or 0)
                            closed_trades.append(ClosedTrade(
                                symbol=_wkpos.symbol, direction=_wkpos.direction,
                                entry_price=_wkpos.entry_price, exit_price=_cl,
                                stop_loss=_wkpos.stop_loss, take_profit=_wkpos.take_profit,
                                quantity=_wkpos.quantity, risk_amount=_wkpos.risk_amount,
                                pnl=_wkpnl, pnl_r=_wkpnl / max(_wkpos.risk_amount, 1e-9),
                                open_bar=_wkpos.open_bar, close_bar=global_step,
                                exit_reason="weekly_dd_close_all", confidence=_wkpos.confidence,
                                win_probability=_wkpos.win_probability,
                                agent_votes=_wkpos.agent_votes,
                            ))
                    open_positions.clear()

            # ── Weekend close (shared across all symbols) ─────────────────
            _weekday = bar_date.weekday()
            if _wkend_enabled:
                if _weekday == 4 and bar_date.hour >= _wkend_hour:
                    if not _weekend_blocked:
                        _weekend_blocked = True
                        for _wpos in list(open_positions):
                            if not _wpos.pending:
                                _tv = sym_ctx[_wpos.symbol]["tick_val"]
                                _ps = sym_ctx[_wpos.symbol]["pip_sz"]
                                _mb = sym_ctx[_wpos.symbol]["mid_bars"]
                                _si = sym_ts_to_idx[_wpos.symbol].get(bar_ts)
                                _cl = float(_mb["close"][_si]) if _si is not None else _wpos.entry_price
                                _wpnl = self._pnl_at(_wpos, _cl, _tv, _ps, close_step=_si or 0)
                                closed_trades.append(ClosedTrade(
                                    symbol=_wpos.symbol, direction=_wpos.direction,
                                    entry_price=_wpos.entry_price, exit_price=_cl,
                                    stop_loss=_wpos.stop_loss, take_profit=_wpos.take_profit,
                                    quantity=_wpos.quantity, risk_amount=_wpos.risk_amount,
                                    pnl=_wpnl, pnl_r=_wpnl / max(_wpos.risk_amount, 1e-9),
                                    open_bar=_wpos.open_bar, close_bar=global_step,
                                    exit_reason="weekend", confidence=_wpos.confidence,
                                    win_probability=_wpos.win_probability,
                                    agent_votes=_wpos.agent_votes,
                                ))
                        open_positions.clear()
                elif _weekday < 4:
                    _weekend_blocked = False

            # ── Process each symbol that has a bar at this timestamp ──────
            for sym in symbols:
                idx = sym_ts_to_idx[sym].get(bar_ts)
                if idx is None:
                    continue

                ctx = sym_ctx[sym]
                mid_bars = ctx["mid_bars"]
                tick_val = ctx["tick_val"]
                pip_sz   = ctx["pip_sz"]
                loader   = ctx["loader"]
                executor = ctx["executor"]
                graph    = ctx["graph"]

                # Check warmup
                if idx < _MIN_BARS:
                    continue
                if not _sym_warmup_done[sym]:
                    if _start_dt and bar_date < _start_dt:
                        continue
                    _sym_warmup_done[sym] = True

                _sym_step[sym] += 1
                step = _sym_step[sym]

                loader._current_step = idx
                loader._current_bar_ts = float(mid_bars["time"][idx])

                hi = float(mid_bars["high"][idx])
                lo = float(mid_bars["low"][idx])
                cl = float(mid_bars["close"][idx])
                bar_open = float(mid_bars["open"][idx])

                # Dynamic spread
                _atr_win_sp = [mid_bars["high"][i] - mid_bars["low"][i]
                               for i in range(max(0, idx - 14), idx)]
                _atr_sp = float(np.mean(_atr_win_sp)) if _atr_win_sp else (hi - lo)
                spread = self._dynamic_spread(sym, hi - lo, _atr_sp)
                bid = cl - spread / 2
                ask = cl + spread / 2
                executor.set_price(sym, bid, ask)
                executor._open_positions = [p for p in open_positions if p.symbol == sym]

                # ── Fill pending entries at bar open ──────────────────────
                _fill_rejects = []
                for pos in open_positions:
                    if pos.pending and pos.symbol == sym:
                        _stop_dist = abs(pos.entry_price - pos.stop_loss)
                        if _stop_dist > 1e-9 and spread / _stop_dist > self._MAX_SPREAD_FRAC:
                            _fill_rejects.append(pos)
                            continue
                        pos.quantity = round(round(pos.quantity / self._VOL_STEP) * self._VOL_STEP, 8)
                        pos.quantity = max(self._VOL_MIN, min(self._VOL_MAX, pos.quantity))
                        _slip = _sym_rng[sym].uniform(-self._SLIPPAGE_PIPS, self._SLIPPAGE_PIPS) * pip_sz
                        if pos.direction == "long":
                            pos.entry_price = bar_open + spread / 2 + _slip
                        else:
                            pos.entry_price = bar_open - spread / 2 + _slip
                        pos.pending = False
                        pos.entry_atr = _atr_sp
                for _rj in _fill_rejects:
                    open_positions.remove(_rj)

                # ── SL/TP exit check for this symbol's positions ─────────
                sym_positions = [p for p in open_positions if p.symbol == sym and not p.pending]
                still_open: List[SimPosition] = []
                for pos in sym_positions:
                    exited = self._check_exit(
                        pos, hi, lo, idx, tick_val, pip_sz, closed_trades,
                        bar_open=bar_open, bar_close=cl,
                    )
                    if exited:
                        # Update bar index to global_step for consistent equity tracking
                        closed_trades[-1] = ClosedTrade(
                            **{**closed_trades[-1].__dict__, "close_bar": global_step}
                        )
                        open_positions.remove(pos)
                    else:
                        still_open.append(pos)

                # ── Surveillance: run agents in exit_check_only mode ──────
                if still_open and (step % _surv_interval == 0 or ctx["last_surv_state"] is None):
                    try:
                        _surv_feats_by_tf = loader.get_multi_features(sym)
                        _surv_feat = (
                            _surv_feats_by_tf.get("mid")
                            or _surv_feats_by_tf.get("long")
                            or _surv_feats_by_tf.get("short")
                        )
                        if _surv_feat is not None:
                            _s = await graph.run(
                                symbol=sym,
                                features=_surv_feat,
                                portfolio_state=None,
                                risk_limits=risk_limits,
                                features_by_tf=_surv_feats_by_tf,
                                exit_check_only=True,
                            )
                            ctx["last_surv_state"] = _s
                    except Exception:
                        pass

                # ── Position management: trailing SL, partial TP, etc. ───
                _fusion = (ctx["last_surv_state"].get("timeframe_fusion")
                           if ctx["last_surv_state"] else None)
                _atr_wins = [mid_bars["high"][i] - mid_bars["low"][i]
                             for i in range(max(0, idx - 14), idx)]
                _atr = float(np.mean(_atr_wins)) if _atr_wins else 0.0

                for pos in still_open:
                    _one_r = abs(pos.entry_price - pos.entry_sl) if pos.entry_sl else 0.0
                    if _one_r < 1e-9:
                        continue
                    _is_long = pos.direction == "long"
                    _profit_dist = (cl - pos.entry_price) if _is_long else (pos.entry_price - cl)
                    _profit_r = _profit_dist / _one_r
                    if _profit_r > pos.peak_profit_r:
                        pos.peak_profit_r = _profit_r

                    _should_close = False
                    _close_reason = ""

                    # Time-stop
                    if _max_hours_ts > 0:
                        _hours_held = (idx - pos.open_bar) * _hrs_per_bar
                        _is_ct = getattr(pos, "entry_type", "") == "counter-trend-scalp"
                        _eff_ts = _ct_max_hours if _is_ct else _max_hours_ts
                        if _hours_held >= _eff_ts:
                            _should_close = True
                            _close_reason = "time-stop"

                    # Conviction fade / D1 flip / opposition
                    if not _should_close and _fusion is not None:
                        _d_long = _fusion.dir_long
                        _d_mid = _fusion.dir_mid
                        _d_short = _fusion.dir_short
                        if _scalp_exits:
                            if _is_long and _d_short < -_scalp_flip:
                                _should_close = True; _close_reason = "scalp: short flip bearish"
                            elif not _is_long and _d_short > _scalp_flip:
                                _should_close = True; _close_reason = "scalp: short flip bullish"
                            if not _should_close and _opp_enabled:
                                if _is_long and _d_mid < -_opp_thresh and _d_short < -_opp_thresh:
                                    _should_close = True; _close_reason = "scalp: mid+short bearish"
                                elif not _is_long and _d_mid > _opp_thresh and _d_short > _opp_thresh:
                                    _should_close = True; _close_reason = "scalp: mid+short bullish"
                        else:
                            _is_ct = getattr(pos, "entry_type", "") == "counter-trend-scalp"
                            if not _is_ct:
                                if _is_long and _d_long < -_flip_thresh:
                                    _should_close = True; _close_reason = "D1 flipped bearish"
                                elif not _is_long and _d_long > _flip_thresh:
                                    _should_close = True; _close_reason = "D1 flipped bullish"
                            if not _should_close and _fade_enabled and abs(_d_long) < _fade_thresh:
                                if not _is_ct:
                                    _should_close = True; _close_reason = "conviction faded"
                            if not _should_close and _opp_enabled:
                                if _is_long and _d_mid < -_opp_thresh and _d_short < -_opp_thresh:
                                    _should_close = True; _close_reason = "mid+short opposition"
                                elif not _is_long and _d_mid > _opp_thresh and _d_short > _opp_thresh:
                                    _should_close = True; _close_reason = "mid+short opposition"
                            if not _should_close and _is_ct:
                                _trade_sign = 1.0 if _is_long else -1.0
                                if _d_mid * _trade_sign < -_opp_thresh:
                                    _should_close = True; _close_reason = "CT mid flipped against"

                    if _should_close:
                        _pnl_c = self._pnl_at(pos, cl, tick_val, pip_sz, close_step=idx)
                        closed_trades.append(ClosedTrade(
                            symbol=pos.symbol, direction=pos.direction,
                            entry_price=pos.entry_price, exit_price=cl,
                            stop_loss=pos.stop_loss, take_profit=pos.take_profit,
                            quantity=pos.quantity, risk_amount=pos.risk_amount,
                            pnl=_pnl_c, pnl_r=_pnl_c / max(pos.risk_amount, 1e-9),
                            open_bar=pos.open_bar, close_bar=global_step,
                            exit_reason=_close_reason, confidence=pos.confidence,
                            win_probability=pos.win_probability,
                            agent_votes=pos.agent_votes,
                        ))
                        open_positions.remove(pos)
                        continue

                    # Tighten-on-fade: conviction weakening but not close-worthy.
                    # Lock SL at break-even and pull TP close when dir_long is
                    # between fade_thresh and tighten_thresh and trade is green.
                    if (_tighten_on and _fusion is not None
                            and not _scalp_exits and _atr > 0
                            and _profit_dist > 0
                            and _fade_thresh < abs(_fusion.dir_long) < _tighten_thresh):
                        if _is_long:
                            _t_sl = max(pos.stop_loss, pos.entry_price)
                            _t_tp = cl + _tighten_tp_m * _atr
                            if _t_sl > pos.stop_loss:
                                pos.stop_loss = _t_sl
                            if pos.take_profit <= 0 or _t_tp < pos.take_profit - 0.2 * _atr:
                                pos.take_profit = _t_tp
                        else:
                            _t_sl = min(pos.stop_loss, pos.entry_price) if pos.stop_loss > 0 else pos.entry_price
                            _t_tp = cl - _tighten_tp_m * _atr
                            if pos.stop_loss <= 0 or _t_sl < pos.stop_loss:
                                pos.stop_loss = _t_sl
                            if pos.take_profit <= 0 or _t_tp > pos.take_profit + 0.2 * _atr:
                                pos.take_profit = _t_tp

                    # Windfall exit
                    if _windfall_on and not pos.windfall_fired and _profit_dist >= _windfall_r * _one_r:
                        pos.windfall_fired = True
                        _pnl_wf = self._pnl_at(pos, cl, tick_val, pip_sz, close_step=idx)
                        closed_trades.append(ClosedTrade(
                            symbol=pos.symbol, direction=pos.direction,
                            entry_price=pos.entry_price, exit_price=cl,
                            stop_loss=pos.stop_loss, take_profit=pos.take_profit,
                            quantity=pos.quantity, risk_amount=pos.risk_amount,
                            pnl=_pnl_wf, pnl_r=_pnl_wf / max(pos.risk_amount, 1e-9),
                            open_bar=pos.open_bar, close_bar=global_step,
                            exit_reason="windfall", confidence=pos.confidence,
                            win_probability=pos.win_probability,
                            agent_votes=pos.agent_votes,
                        ))
                        open_positions.remove(pos)
                        continue

                    # Partial TP1
                    if (_partial1_on and not pos.partial_tp1_fired
                            and 1.0 * _one_r <= _profit_dist < 1.5 * _one_r):
                        _qty1 = pos.quantity * _partial1_frac
                        _p1 = self._pnl_at_qty(pos, cl, _qty1, tick_val, pip_sz)
                        _orig_risk = pos.risk_amount
                        _part_risk = _orig_risk * _partial1_frac
                        closed_trades.append(ClosedTrade(
                            symbol=pos.symbol, direction=pos.direction,
                            entry_price=pos.entry_price, exit_price=cl,
                            stop_loss=pos.stop_loss, take_profit=pos.take_profit,
                            quantity=_qty1, risk_amount=_part_risk,
                            pnl=_p1, pnl_r=_p1 / max(_part_risk, 1e-9),
                            open_bar=pos.open_bar, close_bar=global_step,
                            exit_reason="partial_tp1", confidence=pos.confidence,
                            win_probability=pos.win_probability, agent_votes=pos.agent_votes,
                        ))
                        pos.quantity -= _qty1
                        pos.risk_amount = _orig_risk * (1.0 - _partial1_frac)
                        pos.partial_tp1_fired = True
                        if _is_long:
                            pos.stop_loss = pos.entry_price - _be_buffer_r * _one_r
                        else:
                            pos.stop_loss = pos.entry_price + _be_buffer_r * _one_r

                    # Partial TP2
                    if _partial2_on and not pos.partial_tp2_fired and _profit_dist >= _partial2_r * _one_r:
                        _qty2 = pos.quantity * _partial2_frac
                        _p2 = self._pnl_at_qty(pos, cl, _qty2, tick_val, pip_sz)
                        _orig_risk2 = pos.risk_amount
                        _part_risk2 = _orig_risk2 * _partial2_frac
                        closed_trades.append(ClosedTrade(
                            symbol=pos.symbol, direction=pos.direction,
                            entry_price=pos.entry_price, exit_price=cl,
                            stop_loss=pos.stop_loss, take_profit=pos.take_profit,
                            quantity=_qty2, risk_amount=_part_risk2,
                            pnl=_p2, pnl_r=_p2 / max(_part_risk2, 1e-9),
                            open_bar=pos.open_bar, close_bar=global_step,
                            exit_reason="partial_tp2", confidence=pos.confidence,
                            win_probability=pos.win_probability, agent_votes=pos.agent_votes,
                        ))
                        pos.quantity -= _qty2
                        pos.risk_amount = _orig_risk2 * (1.0 - _partial2_frac)
                        pos.partial_tp2_fired = True
                        _bump = 0.7 * _one_r
                        if _is_long:
                            pos.stop_loss = max(pos.stop_loss, pos.entry_price + _bump)
                        else:
                            pos.stop_loss = min(pos.stop_loss, pos.entry_price - _bump)

                    # Progressive stale-SL tightening
                    if _stale_sl_hours > 0 and _profit_dist < 0:
                        _hours_held_sl = (idx - pos.open_bar) * _hrs_per_bar
                        if _hours_held_sl >= _stale_sl_hours:
                            _overtime_frac = min(1.0, (_hours_held_sl - _stale_sl_hours)
                                                      / (2.0 * _stale_sl_hours))
                            _atr_ratio = min(1.0, _atr / pos.entry_atr) if pos.entry_atr > 1e-9 else 1.0
                            _prog_r = _stale_sl_r * _atr_ratio * (1.0 - 0.40 * _overtime_frac)
                            if _is_long:
                                _tight_sl = pos.entry_price - _prog_r * _one_r
                                if pos.stop_loss < _tight_sl - 1e-9:
                                    pos.stop_loss = _tight_sl
                            else:
                                _tight_sl = pos.entry_price + _prog_r * _one_r
                                if pos.stop_loss > _tight_sl + 1e-9:
                                    pos.stop_loss = _tight_sl

                    # Stale winner protection
                    if (_stale_sl_hours > 0 and _profit_r > 0 and _profit_r < 1.0
                            and pos.peak_profit_r - _profit_r >= 0.3):
                        _hours_held_sw = (idx - pos.open_bar) * _hrs_per_bar
                        if _hours_held_sw >= _stale_sl_hours:
                            if _is_long:
                                _be_floor = pos.entry_price - _be_buffer_r * _one_r
                                if pos.stop_loss < _be_floor - 1e-9:
                                    pos.stop_loss = _be_floor
                            else:
                                _be_floor = pos.entry_price + _be_buffer_r * _one_r
                                if pos.stop_loss > _be_floor + 1e-9:
                                    pos.stop_loss = _be_floor

                    # Trailing SL stages
                    if _atr > 0:
                        _be_sl_long  = pos.entry_price - _be_buffer_r * _one_r
                        _be_sl_short = pos.entry_price + _be_buffer_r * _one_r
                        if _is_long:
                            if _profit_dist >= 2.0 * _one_r:
                                _new_sl = max(pos.stop_loss, cl - _atr * _atr_mult, pos.entry_price + 0.5 * _one_r)
                                if _tp_extend_on and pos.take_profit > 0 and (pos.take_profit - cl) < _atr:
                                    pos.take_profit = cl + _tp_extend_mult * _atr
                            elif _profit_dist >= 1.5 * _one_r:
                                _new_sl = max(pos.stop_loss, pos.entry_price + 0.5 * _one_r)
                            elif _profit_dist >= 1.0 * _one_r:
                                _new_sl = max(pos.stop_loss, _be_sl_long)
                            elif _early_be_r > 0 and _profit_dist >= _early_be_r * _one_r:
                                _new_sl = max(pos.stop_loss, _be_sl_long)
                            else:
                                _new_sl = pos.stop_loss
                            if _new_sl > pos.stop_loss:
                                pos.stop_loss = _new_sl
                        else:
                            if _profit_dist >= 2.0 * _one_r:
                                _new_sl = min(pos.stop_loss, cl + _atr * _atr_mult, pos.entry_price - 0.5 * _one_r)
                                if _tp_extend_on and pos.take_profit > 0 and (cl - pos.take_profit) < _atr:
                                    pos.take_profit = cl - _tp_extend_mult * _atr
                            elif _profit_dist >= 1.5 * _one_r:
                                _new_sl = min(pos.stop_loss, pos.entry_price - 0.5 * _one_r)
                            elif _profit_dist >= 1.0 * _one_r:
                                _new_sl = min(pos.stop_loss, _be_sl_short)
                            elif _early_be_r > 0 and _profit_dist >= _early_be_r * _one_r:
                                _new_sl = min(pos.stop_loss, _be_sl_short)
                            else:
                                _new_sl = pos.stop_loss
                            if _new_sl < pos.stop_loss:
                                pos.stop_loss = _new_sl

                    # Structural pivot ratchet (mirrors trailing_stop._update_structural_one)
                    if not _skip_struct_ratchet and _atr > 0 and _profit_dist > 0:
                        _struct_buf  = 0.25 * _atr
                        _min_sl_dist = 1.5 * _atr
                        _live_p = ask if _is_long else bid
                        _pivots = loader.get_structural_levels(sym, _live_p, _atr)
                        if _is_long:
                            _struct_sl = _pivots["nearest_support"] - _struct_buf
                            _struct_sl = min(_struct_sl, _live_p - _min_sl_dist)
                            if _struct_sl > pos.stop_loss:
                                pos.stop_loss = _struct_sl
                            _resist_cands = sorted(
                                l for l in (_pivots["r1"], _pivots["r2"])
                                if l > pos.entry_price + _min_sl_dist
                            )
                            if _resist_cands and pos.take_profit > 0:
                                _struct_tp = _resist_cands[0]
                                if _struct_tp > pos.take_profit:
                                    pos.take_profit = _struct_tp
                        else:
                            _struct_sl = _pivots["nearest_resist"] + _struct_buf
                            _struct_sl = max(_struct_sl, _live_p + _min_sl_dist)
                            if _struct_sl < pos.stop_loss:
                                pos.stop_loss = _struct_sl
                            _supp_cands = sorted(
                                (l for l in (_pivots["s1"], _pivots["s2"])
                                 if l < pos.entry_price - _min_sl_dist),
                                reverse=True,
                            )
                            if _supp_cands and pos.take_profit > 0:
                                _struct_tp = _supp_cands[0]
                                if _struct_tp < pos.take_profit:
                                    pos.take_profit = _struct_tp

                    # Late _should_close check (stale_market_close sets it after
                    # the early conviction check at section above).
                    if _should_close:
                        _pnl_lc = self._pnl_at(pos, cl, tick_val, pip_sz, close_step=idx)
                        closed_trades.append(ClosedTrade(
                            symbol=pos.symbol, direction=pos.direction,
                            entry_price=pos.entry_price, exit_price=cl,
                            stop_loss=pos.stop_loss, take_profit=pos.take_profit,
                            quantity=pos.quantity, risk_amount=pos.risk_amount,
                            pnl=_pnl_lc, pnl_r=_pnl_lc / max(pos.risk_amount, 1e-9),
                            open_bar=pos.open_bar, close_bar=global_step,
                            exit_reason=_close_reason, confidence=pos.confidence,
                            win_probability=pos.win_probability,
                            agent_votes=pos.agent_votes,
                        ))
                        open_positions.remove(pos)
                        continue

                # ── Signal pipeline: run agents and enter ─────────────────
                executor.placed_orders.clear()
                _run_agents = (step % _agent_interval == 0) or (ctx["last_agent_state"] is None)
                if not _run_agents or _day_halted or _wk_halted or _weekend_blocked:
                    continue

                # Pre-graph entry guards (shared position count)
                _skip_entry = False
                if _cooldown_min > 0:
                    _pre_last_t = ctx["last_order_attempt"].get(sym) or ctx["last_entry"].get(sym)
                    if _pre_last_t is not None:
                        _pre_elapsed = (bar_date - _pre_last_t).total_seconds() / 60.0
                        if _pre_elapsed < _cooldown_min:
                            _skip_entry = True
                if not _skip_entry and _max_daily > 0 and _day_trade_count >= _max_daily:
                    _skip_entry = True
                # Shared max_concurrent across ALL symbols
                if not _skip_entry:
                    _pre_mc = risk_limits.max_concurrent_trades
                    if _pre_mc > 0 and len([p for p in open_positions if not p.pending]) >= _pre_mc:
                        _skip_entry = True
                if not _skip_entry and _news_blackout_enabled and _atr_sp > 0:
                    _bar_range_v = hi - lo
                    if _bar_range_v > _news_atr_mult * _atr_sp:
                        _skip_entry = True
                if not _skip_entry and _leverage > 0:
                    _margin_used = sum(
                        p.quantity * p.entry_price * 100_000 / _leverage
                        for p in open_positions if not p.pending
                    )
                    _free_margin = equity - _margin_used
                    if _free_margin < 0.20 * equity:
                        _skip_entry = True
                if _skip_entry:
                    continue

                try:
                    features_by_tf = loader.get_multi_features(sym)
                    features = (
                        features_by_tf.get("mid")
                        or features_by_tf.get("long")
                        or features_by_tf.get("short")
                    )
                    if features is None:
                        continue

                    # Build portfolio state with shared equity and ALL positions
                    portfolio = self._make_portfolio_multi(
                        equity, _day_open_equity, open_positions,
                        sym_ctx, sym_ts_to_idx, bar_ts,
                    )
                    _sym_hist = ctx.get("streak_history", [])
                    _recent_pnl_r = [1.0 if t[1] > 0 else -1.0 for t in _sym_hist[-6:]]

                    state = await graph.run(
                        symbol=sym,
                        features=features,
                        portfolio_state=portfolio,
                        risk_limits=risk_limits,
                        features_by_tf=features_by_tf,
                        recent_pnl_r=_recent_pnl_r,
                    )
                    ctx["last_agent_state"] = state

                    for order in executor.placed_orders:
                        # Shared max_concurrent guard
                        _max_conc = risk_limits.max_concurrent_trades
                        if _max_conc > 0 and len([p for p in open_positions if not p.pending]) >= _max_conc:
                            break

                        if _cooldown_min > 0:
                            last_t = ctx["last_entry"].get(sym)
                            if last_t is not None:
                                elapsed = (bar_date - last_t).total_seconds() / 60.0
                                if elapsed < _cooldown_min:
                                    continue

                        if _max_daily > 0 and _day_trade_count >= _max_daily:
                            break

                        tp = order["trade_plan"]
                        _dir = getattr(tp.recipe.direction, "value", str(tp.recipe.direction))
                        if "." in _dir:
                            _dir = _dir.split(".")[-1].lower()

                        # Streak guard
                        _blk_dir = ctx.get("streak_block_dir")
                        _blk_until = ctx.get("streak_block_until", 0)
                        if _blk_dir == _dir and step < _blk_until:
                            continue
                        elif _blk_dir and (step >= _blk_until or _dir != _blk_dir):
                            if _dir != _blk_dir:
                                ctx["streak_block_count"] = 0
                            ctx["streak_block_dir"] = None
                            ctx["streak_block_until"] = 0

                        _sl_dist = abs((order["entry_price"] or 0) - (order["stop_loss"] or 0))
                        if _sl_dist < 1e-7:
                            continue

                        pos = SimPosition(
                            symbol=sym,
                            direction=_dir,
                            entry_price=order["entry_price"],
                            stop_loss=order["stop_loss"],
                            take_profit=order["take_profit"],
                            quantity=tp.quantity,
                            risk_amount=tp.risk_amount,
                            open_bar=idx,  # symbol-local bar index (for hours_held calc)
                            ticket=order["order_ticket"],
                            open_global_step=global_step,
                            confidence=tp.confidence,
                            win_probability=tp.recipe.win_probability,
                            agent_votes={
                                name: float(out.dir_score)
                                for name, out in state.get("agent_outputs", {}).items()
                                if hasattr(out, "dir_score")
                            },
                            pending=True,
                            entry_sl=order["stop_loss"],
                            entry_type=state.get("metadata", {}).get("entry_type", "full-alignment"),
                        )
                        open_positions.append(pos)
                        ctx["last_entry"][sym] = bar_date
                        _day_trade_count += 1

                    if executor.placed_orders:
                        ctx["last_order_attempt"][sym] = bar_date

                except Exception as exc:
                    self.logger.debug(f"Portfolio step {global_step} {sym}: {exc}")

            # ── Update shared equity from ALL positions ───────────────────
            unrealized = 0.0
            _sym_unrealized: Dict[str, float] = {s: 0.0 for s in symbols}
            for p in open_positions:
                if p.pending:
                    continue
                _pctx = sym_ctx[p.symbol]
                _p_idx = sym_ts_to_idx[p.symbol].get(bar_ts)
                if _p_idx is not None:
                    _p_bid = float(_pctx["mid_bars"]["close"][_p_idx]) - self._spread(p.symbol) / 2
                    _p_ask = float(_pctx["mid_bars"]["close"][_p_idx]) + self._spread(p.symbol) / 2
                    _p_price = _p_bid if p.direction == "long" else _p_ask
                    _p_pnl = self._pnl_at(p, _p_price, _pctx["tick_val"], _pctx["pip_sz"],
                                               close_step=_p_idx)
                    unrealized += _p_pnl
                    _sym_unrealized[p.symbol] = _sym_unrealized.get(p.symbol, 0.0) + _p_pnl
            realized = sum(t.pnl for t in closed_trades)
            equity = initial_equity + realized + unrealized
            equity_curve[-1] = equity

            # Update per-symbol equity curves
            for _ct in closed_trades[getattr(self, "_prev_closed_count_pf", 0):]:
                _sym_realized[_ct.symbol] = _sym_realized.get(_ct.symbol, 0.0) + _ct.pnl
            for _s in symbols:
                _s_eq = initial_equity + _sym_realized[_s] + _sym_unrealized.get(_s, 0.0)
                if len(_sym_equity_curve[_s]) < len(equity_curve):
                    _sym_equity_curve[_s].append(_s_eq)
                else:
                    _sym_equity_curve[_s][-1] = _s_eq

            # Record streak history for closed trades
            for _ct in closed_trades[getattr(self, "_prev_closed_count_pf", 0):]:
                if _ct.exit_reason not in ("partial_tp1", "partial_tp2"):
                    sh = sym_ctx[_ct.symbol].get("streak_history", [])
                    sh.append((_ct.direction, _ct.pnl, global_step))
                    sym_ctx[_ct.symbol]["streak_history"] = sh
                    if len(sh) >= 2 and sh[-1][1] < 0 and sh[-2][1] < 0 and sh[-1][0] == sh[-2][0]:
                        _prev_count = sym_ctx[_ct.symbol].get("streak_block_count", 0)
                        sym_ctx[_ct.symbol]["streak_block_count"] = _prev_count + 1
                        _cooldown_bars = min(_streak_base_bars * (2 ** _prev_count), _streak_max_bars)
                        sym_ctx[_ct.symbol]["streak_block_dir"] = sh[-1][0]
                        sym_ctx[_ct.symbol]["streak_block_until"] = _sym_step[_ct.symbol] + _cooldown_bars
            self._prev_closed_count_pf = len(closed_trades)

            # Non-TTY progress
            if not _is_tty:
                _done = global_step + 1
                if _done % _report_every == 0 or _done == n_bars_total:
                    _pct = 100.0 * _done / n_bars_total
                    print(f"  [portfolio] {_done}/{n_bars_total} bars ({_pct:.0f}%)",
                          file=_sys_.stderr, flush=True)

        # ── Close remaining positions at last bar ─────────────────────────
        _graph_logger.setLevel(_graph_orig_level)
        for pos in open_positions:
            ctx = sym_ctx[pos.symbol]
            _mb = ctx["mid_bars"]
            _last_cl = float(_mb["close"][-1])
            pnl = self._pnl_at(pos, _last_cl, ctx["tick_val"], ctx["pip_sz"],
                                close_step=len(_mb["close"]) - 1)
            pnl_r = pnl / max(pos.risk_amount, 1e-9)
            closed_trades.append(ClosedTrade(
                symbol=pos.symbol, direction=pos.direction,
                entry_price=pos.entry_price, exit_price=_last_cl,
                stop_loss=pos.stop_loss, take_profit=pos.take_profit,
                quantity=pos.quantity, risk_amount=pos.risk_amount,
                pnl=pnl, pnl_r=pnl_r,
                open_bar=pos.open_bar, close_bar=n_bars_total - 1,
                exit_reason="eod", confidence=pos.confidence,
                win_probability=pos.win_probability, agent_votes=pos.agent_votes,
            ))
            _sym_realized[pos.symbol] = _sym_realized.get(pos.symbol, 0.0) + pnl

        final_equity = initial_equity + sum(t.pnl for t in closed_trades)
        if equity_curve:
            equity_curve[-1] = final_equity
        # Finalize per-symbol equity curves
        for _s in symbols:
            _s_final = initial_equity + _sym_realized[_s]
            if _sym_equity_curve[_s]:
                _sym_equity_curve[_s][-1] = _s_final

        # Stamp open_dt / close_dt on trades
        for t in closed_trades:
            # open_bar is symbol-local → look up from per-symbol bar_dates
            _sym_dates = sym_ctx[t.symbol]["bar_dates"]
            if 0 <= t.open_bar < len(_sym_dates):
                t.open_dt = _sym_dates[t.open_bar]
            # close_bar is global_step → look up from unified_dates
            if 0 <= t.close_bar < n_bars_total:
                t.close_dt = unified_dates[min(t.close_bar, n_bars_total - 1)]

        self.logger.info(
            f"Portfolio backtest done: {len(closed_trades)} trades across "
            f"{len(symbols)} symbols  equity {initial_equity:.0f} → {final_equity:.0f} "
            f"({(final_equity / initial_equity - 1) * 100:+.2f}%)"
        )

        # ── Build per-symbol BacktestResults ──────────────────────────────
        results: List[BacktestResult] = []
        for sym in symbols:
            sym_trades = [t for t in closed_trades if t.symbol == sym]
            # Use per-symbol equity curve so that compute_metrics reports
            # return%, maxDD, and Sharpe for THIS symbol's contribution,
            # not the whole portfolio aggregate.
            _s_curve = _sym_equity_curve.get(sym, equity_curve)
            results.append(BacktestResult(
                profile=self.config.profile,
                symbol=sym,
                start_date=start_date,
                end_date=end_date,
                trades=sym_trades,
                equity_curve=_s_curve,
                bar_dates=equity_dates,
                initial_equity=initial_equity,
                config=self.config.model_dump(),
            ))

        return results
    #   - Staged SL trail: break-even @ 1R, +0.5R @ 1.5R, ATR-trail @ 2R
    #   - TP extension: push TP forward when price closes within 1 ATR (stage 3)
    #   - Windfall exit at N×R (windfall_r_mult)
    #   - Partial TP1 at 1R (stage 1 only), Partial TP2 at N×R
    #   - D1 flip / conviction fade / mid+short opposition exits
    #   - Scalp short-tier exits
    #   - Time-stop (max_trade_duration_hours)
    #   - Daily + weekly drawdown circuit breakers
    #   - Weekend gap close (Friday ≥ utc cutoff)
    #   - Entry cooldown, daily trade cap
    #   - Signal loop vs surveillance loop throttle (agent_interval / surv_interval)
    #   - Macro features (DXY/VIX/crude/yields) via pre-loaded yfinance history
    #   - Bar-level timestamp for dead zone (uses _current_bar_ts, not wall clock)
    #
    #  NOT SIMULATED ⚠️:
    #   - News calendar blackout: simulated via volatility-spike heuristic
    #     (range > 3×ATR blocks entry); no ForexFactory feed in backtest.
    #   - Cross-symbol capital sharing: handled by _run_async_portfolio()
    #     (run_multi). This single-symbol _run_async is used only for
    #     individual symbol backtests or walk-forward windows.

    async def _run_async(
        self,
        symbol: str,
        all_bars: Dict[str, Dict[str, np.ndarray]],
        mid_tf: str,
        bar_dates: List[datetime],
        initial_equity: float,
        executor: _BacktestExecutor,
        loader: _BacktestDataLoader,
        graph: TradingGraph,
        tier_tf_map: Dict[str, str],
        start_date: str = "",
        end_date: str = "",
    ) -> BacktestResult:
        mid_key = f"{symbol}_{mid_tf}"
        mid_bars = all_bars[mid_key]
        n_bars = len(bar_dates)
        _base_spread = self._spread(symbol)
        spread = _base_spread                    # overwritten per bar below
        tick_val = self._tick_value(symbol)
        pip_sz = self._pip_size(symbol)

        # ── Deterministic slippage: seed RNG for reproducible backtests ──
        random.seed(42)

        open_positions: List[SimPosition] = []
        closed_trades: List[ClosedTrade] = []
        executor._open_positions = open_positions

        # ── Debug dump file for SL lifecycle analysis ─────────────────────
        # Writes CSV to /tmp/bt_sl_debug_<symbol>.csv with every fill, SL change,
        # and exit. Enable by checking for the file after a run.
        import os as _os
        _dump_path = f"/tmp/bt_sl_debug_{symbol}.csv"
        try:
            self._sl_dump = open(_dump_path, "w")
            self._sl_dump.write(
                "event,bar,time,symbol,direction,ticket,"
                "entry,stop_loss,take_profit,entry_sl,"
                "one_r_pip,sl_dist_pip,qty,risk_amt,"
                "profit_r,reason\n"
            )
        except Exception:
            self._sl_dump = None

        equity = initial_equity
        # equity_curve and equity_dates are populated INSIDE the bar loop (starting
        # at _sig_bar) so the chart only covers the requested date range, not the
        # warm-up period that precedes it.
        equity_curve: List[float] = []
        equity_dates: List[datetime] = []

        portfolio = self._make_portfolio(equity, initial_equity, open_positions, mid_bars, 0, tick_val, pip_sz)
        risk_limits = RiskLimits(
            base_risk_pct=self.config.risk.base_risk_pct,
            max_daily_drawdown_pct=self.config.risk.max_daily_drawdown_pct,
            max_concurrent_trades=self.config.risk.max_concurrent_trades,
        )

        # ── Pre-compute config dicts once (avoid model_dump() per bar) ──────
        _cfg_raw       = self.config.model_dump()
        _trailing_cfg  = _cfg_raw.get("trailing", {})
        _exit_cfg_raw  = _cfg_raw.get("exit_rules", {})
        _al_cfg        = _cfg_raw.get("alignment", {})
        _rt_cfg        = _cfg_raw.get("realtime", {})
        # Trailing / partial TP / windfall settings
        _atr_mult      = float(_trailing_cfg.get("atr_multiplier", 1.5))
        _partial1_on   = bool(_trailing_cfg.get("partial_tp_enabled", False))
        _partial1_frac = float(_trailing_cfg.get("partial_tp_fraction", 0.50))
        _partial2_on   = bool(_trailing_cfg.get("partial_tp2_enabled", False))
        _partial2_r    = float(_trailing_cfg.get("partial_tp2_r_mult", 2.0))
        _partial2_frac = float(_trailing_cfg.get("partial_tp2_fraction", 0.50))
        _windfall_on   = bool(_trailing_cfg.get("windfall_exit_enabled", False))
        _windfall_r    = float(_trailing_cfg.get("windfall_r_mult", 3.0))
        # TP extension: push TP forward when price closes in (stage 3 running trade).
        # Mirrors trailing_stop.py._manage_one stage 3 tp_extend logic exactly.
        _tp_extend_on   = bool(_trailing_cfg.get("tp_extend_enabled", True))
        _tp_extend_mult = float(_trailing_cfg.get("tp_extend_atr_mult", 2.0))
        # Early break-even: move SL to entry at +early_be_r (e.g. +0.5R).
        # Fires BEFORE stage 1 (+1R). 0 = disabled.
        _early_be_r     = float(_trailing_cfg.get("early_be_r", 0.0))
        _be_buffer_r    = float(_trailing_cfg.get("be_buffer_r", 0.0))
        # Stale-SL tightening: after N hours if still losing, cap at -stale_r×1R.
        _stale_sl_hours = float(_trailing_cfg.get("stale_sl_hours", 0.0))
        _stale_sl_r     = float(_trailing_cfg.get("stale_sl_r_mult", 0.75))
        # Exit rule settings
        _max_hours_ts  = float(_exit_cfg_raw.get("max_trade_duration_hours", 0))
        # Default False: conviction_fade is opt-in (must be explicitly enabled in config).
        # Defaulting True was silently force-closing positions on every bar where |dir_long|
        # dipped below 0.10, killing trades that were entered correctly.
        _fade_enabled  = bool(_exit_cfg_raw.get("conviction_fade_enabled", False))
        _fade_thresh   = float(_exit_cfg_raw.get("conviction_fade_threshold", 0.10))
        _opp_enabled   = bool(_exit_cfg_raw.get("mid_short_opposition_enabled", True))
        _opp_thresh    = float(_exit_cfg_raw.get("mid_short_opposition_threshold", 0.35))
        # Tighten-on-fade: when conviction is weakening but not yet at close threshold,
        # lock SL at break-even and tighten TP.  Mirrors runner._check_and_close_positions.
        _tighten_on     = bool(_exit_cfg_raw.get("tighten_on_fade_enabled", True))
        _tighten_thresh = float(_exit_cfg_raw.get("tighten_fade_threshold", 0.20))
        _tighten_tp_m   = float(_exit_cfg_raw.get("tighten_tp_atr_mult", 0.5))
        # d1_flip_threshold: separate config key lets profiles set a higher bar for
        # D1 reversal exits than the entry long_min_score.  Using long_min_score as
        # exit threshold was causing premature exits (score crosses 0.28 on noise).
        _flip_thresh   = float(_exit_cfg_raw.get("d1_flip_threshold",
                                                  _al_cfg.get("long_min_score", 0.25)))
        _scalp_exits   = bool(_exit_cfg_raw.get("use_short_tier_exits", False))
        _scalp_flip    = float(_exit_cfg_raw.get("short_flip_threshold", 0.35))
        # Counter-trend scalp time-stop: shorter than regular trades (default 2h)
        _ct_max_hours  = float(_exit_cfg_raw.get("ct_max_hours", 2.0))
        # Structural pivot ratchet: disabled for profiles where mid_tf noise
        # produces micro-structure that mangles staged trailing.
        _skip_struct_ratchet = bool(_exit_cfg_raw.get("skip_structural_sl_tp_ratchet", False))
        # Time-stop: hours per bar by mid-tier timeframe
        _tf_hours_map  = {"1m": 1/60, "5m": 5/60, "15m": 0.25, "30m": 0.5,
                          "1H": 1.0,  "4H": 4.0,  "1D": 24.0}
        _hrs_per_bar   = _tf_hours_map.get(mid_tf, 1.0)
        self._hrs_per_bar = _hrs_per_bar   # expose to _pnl_at for correct swap calc
        # Weekend close settings
        _wkend_enabled = bool(_rt_cfg.get("weekend_close_enabled", False))
        _wkend_hour    = int(_rt_cfg.get("weekend_close_utc_hour", 20))
        # Weekly circuit breaker
        _wk_dd_pct     = float(_cfg_raw.get("risk", {}).get("max_weekly_drawdown_pct", 0))
        # News blackout: volatility-spike heuristic (range > N×ATR blocks entry)
        # Mirrors runner._NewsCalendar intent without needing ForexFactory data.
        _news_atr_mult = float(_cfg_raw.get("execution", {}).get("news_spike_atr_mult", 3.0))
        _news_blackout_enabled = bool(_cfg_raw.get("execution", {}).get("news_blackout_minutes", 0) > 0)
        # Margin tracking: approximate free margin to reject oversized entries
        _leverage      = float(_cfg_raw.get("risk", {}).get("leverage", 30))
        _margin_used   = 0.0   # running total of margin consumed by open positions

        # Agent-run throttle: only invoke the full signal pipeline every N bars,
        # mirroring the live runner's signal loop vs surveillance loop separation.
        # Surveillance (exit checks, trailing SL) runs EVERY bar regardless.
        # Signal (new entries) runs every interval_seconds / bar_seconds bars.
        #   Live risky: interval_seconds=300, bar=60s → every 5 bars
        #   Live scalp: interval_seconds=60,  bar=60s → every 1 bar
        #   H1 mid-tf:  interval_seconds=300, bar=3600s → every 1 bar (rounds up to 1)
        _bar_seconds_map = {
            "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
            "1H": 3600, "4H": 14400, "1D": 86400,
        }
        _bar_seconds   = _bar_seconds_map.get(mid_tf, 3600)
        # Signal loop: full agent pipeline + entries (mirrors runner._signal_loop)
        _sig_interval  = int(_cfg_raw.get("interval_seconds", None) or getattr(self.config, "interval_seconds", 300) or 300)
        _agent_interval = max(1, round(_sig_interval / _bar_seconds))
        _last_agent_state: Optional[dict] = None
        # Surveillance loop: exit_check_only (mirrors runner._surveillance_loop)
        # Auto-derive from mid TF when not explicitly set in config.
        _surv_explicit = _cfg_raw.get("realtime", {}).get("surveillance_interval_seconds")
        if _surv_explicit is not None:
            _surv_seconds = int(_surv_explicit)
        else:
            _surv_by_tf = {
                "1m": 10, "5m": 15, "15m": 30, "30m": 45,
                "1H": 60, "4H": 90, "1D": 120,
            }
            _surv_seconds = _surv_by_tf.get(mid_tf, 60)
        _surv_interval = max(1, round(_surv_seconds / _bar_seconds))
        _last_surv_state: Optional[dict] = None
        self.logger.info(
            f"[timing] mid_tf={mid_tf}  bar={_bar_seconds}s  "
            f"signal={_sig_interval}s → agent_interval={_agent_interval} bars  "
            f"surv={_surv_seconds}s → surv_interval={_surv_interval} bars"
        )

        # Signal start: skip warm-up bars AND any bars before the requested start_date.
        # This ensures WF sub-windows only trade within their assigned window period.
        _sig_bar = _MIN_BARS
        if start_date:
            try:
                _start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                _idx = next((i for i, d in enumerate(bar_dates) if d >= _start_dt), _MIN_BARS)
                _sig_bar = max(_idx, _MIN_BARS)
            except Exception:
                pass

        # ── Live-parity guards ────────────────────────────────────────────────
        # Daily drawdown circuit breaker
        _dd_pct          = self.config.risk.max_daily_drawdown_pct
        _day_open_equity = equity
        _day_halted      = False
        _current_day: Optional[int] = None
        # Weekly drawdown circuit breaker
        _wk_open_equity  = equity
        _wk_halted       = False
        _current_week: Optional[int] = None   # ISO week number
        # Entry cooldown per symbol
        # Scale cooldown by mid-tf granularity: the configured value assumes 1H bars.
        # On faster timeframes (1m, 5m, 15m) a shorter cooldown lets the bot adapt
        # to intraday signal changes rather than sitting idle for an hour.
        _cooldown_base   = float(getattr(self.config.risk, "entry_cooldown_minutes", 0.0))
        _cooldown_scale  = {"1m": 0.25, "5m": 0.50, "15m": 0.75, "30m": 0.85}.get(mid_tf, 1.0)
        _cooldown_min    = max(5.0, _cooldown_base * _cooldown_scale)  # floor at 5 min
        if _cooldown_scale < 1.0:
            self.logger.info(
                f"[cooldown] {_cooldown_base:.0f}min × {_cooldown_scale:.2f} "
                f"({mid_tf} scaling) → {_cooldown_min:.0f}min"
            )
        _last_entry: Dict[str, datetime] = {}
        _last_order_attempt: Dict[str, datetime] = {}  # last graph.py order (accepted OR rejected)
        # Daily trade cap
        _max_daily       = int(getattr(self.config.risk, "max_daily_trades", 0))
        _day_trade_count = 0
        # Weekend block
        _weekend_blocked = False

        # ── Same-direction losing streak guard ──────────────────────────────
        # Prevents the bot from blindly repeating the same losing direction.
        # If the last 2 completed trades on this symbol were both losers AND
        # both the same direction, block that direction for a cooling period.
        # Escalating cooldown: 4h → 8h → 16h → 24h (doubles each re-trigger,
        # caps at 24h) to prevent the "block → expire → lose → block" loop.
        _streak_history: Dict[str, List[tuple]] = {}   # symbol → [(dir, pnl, bar_idx), ...]
        _streak_block_until: Dict[str, int] = {}       # symbol → bar index when block expires
        _streak_block_dir: Dict[str, str] = {}         # symbol → blocked direction
        _streak_block_count: Dict[str, int] = {}       # symbol → consecutive streak count
        _streak_base_bars = max(1, int(4 * 3600 / _bar_seconds))  # 4h in bars
        _streak_max_bars  = max(1, int(24 * 3600 / _bar_seconds)) # 24h cap
        self._prev_closed_count = 0

        # ── Intra-bar simulation (IBS) ────────────────────────────────────────
        # When intra_bar_max_trades > 0 and mid_tf == "1m", the bar loop replays
        # the intra-bar OHLC price path through 4 waypoints.  After a TP hit the
        # position is immediately re-entered in the same direction at the TP price
        # (momentum continuation) — up to intra_bar_max_trades times per bar.
        # SL hits do NOT trigger a re-entry (signal failed — wait for next bar).
        # Price waypoint order:
        #   Bullish bar (C > O): O → L → H → C  (dip first, then rally)
        #   Bearish/doji       : O → H → L → C  (spike first, then fall)
        _ibs_max  = int(_cfg_raw.get("execution", {}).get("intra_bar_max_trades", 0))
        _ibs_mode = _ibs_max > 0 and mid_tf == "1m"
        _last_ibs_order: Optional[Dict] = None   # cached order for IBS re-entry

        # ── Suppress chatty graph logs during backtest ─────────────────────
        # In non-debug mode, graph.py's per-pipeline INFO lines ("Running
        # agents", "Fusing", "Order placed", etc.) flood the output with
        # proposals that the engine may reject.  Silence them; the engine
        # will log confirmed ENTRY/EXIT events instead.
        _graph_logger = logging.getLogger("TradingGraph")
        _graph_orig_level = _graph_logger.level
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
            _graph_logger.setLevel(logging.WARNING)

        _bar_range = range(_sig_bar, n_bars)
        _is_tty = getattr(_sys_.stderr, "isatty", lambda: False)()
        # tqdm for interactive terminals; plain-text fallback when piped
        # (leave=True so the final bar stays visible after the loop ends).
        if _tqdm is not None and _is_tty:
            _progress = _tqdm(_bar_range, desc=f"Backtest {symbol}",
                              unit="bar", leave=True, dynamic_ncols=True)
        else:
            _progress = _bar_range
        # How often to emit a plain-text progress line in non-TTY mode
        # (every ~5% of bars, but at least every 100 bars).
        _n_bars_range = max(len(_bar_range), 1)
        _report_every = max(100, _n_bars_range // 20)

        for step in _progress:
            loader._current_step = step
            loader._current_bar_ts = float(mid_bars["time"][step])   # timestamp-based TF indexing
            bar_date = bar_dates[step]
            hi = float(mid_bars["high"][step])
            lo = float(mid_bars["low"][step])
            cl = float(mid_bars["close"][step])
            # ── Dynamic spread: widen during volatile bars ─────────────────
            _bar_range = hi - lo
            _atr_win_sp = [mid_bars["high"][i] - mid_bars["low"][i]
                           for i in range(max(0, step - 14), step)]
            _atr_sp = float(np.mean(_atr_win_sp)) if _atr_win_sp else _bar_range
            spread = self._dynamic_spread(symbol, _bar_range, _atr_sp)
            # Plain-text progress for non-TTY mode (e.g. `command | tail -20`)
            if not _is_tty:
                _done = step - _sig_bar + 1
                if _done % _report_every == 0 or _done == _n_bars_range:
                    _pct = 100.0 * _done / _n_bars_range
                    print(f"  [{symbol}] {_done}/{_n_bars_range} bars ({_pct:.0f}%)",
                          file=_sys_.stderr, flush=True)
            # Record equity at the TOP of the bar (before any trades) so that the
            # equity_dates array stays in 1-to-1 sync with equity_curve.
            equity_curve.append(equity)
            equity_dates.append(bar_date)

            # ── Daily / weekly guard reset and circuit breaker check ────────
            if _current_day != bar_date.day:
                _current_day = bar_date.day
                _day_open_equity = equity
                _day_halted = False
                _day_trade_count = 0
            if not _day_halted and _dd_pct > 0:
                day_dd = (equity - _day_open_equity) / max(_day_open_equity, 1e-9) * 100
                if day_dd <= -_dd_pct:
                    _day_halted = True
                    # ── DD close-all: mirror live runner behaviour ─────────
                    # The live runner force-closes every open position when
                    # daily DD is breached (surveillance loop).  Without this
                    # the backtest just blocks new entries while existing
                    # losers keep bleeding — overstating performance vs live.
                    for _ddpos in list(open_positions):
                        if not _ddpos.pending:
                            _ddpnl = self._pnl_at(_ddpos, cl, tick_val, pip_sz, close_step=step)
                            closed_trades.append(ClosedTrade(
                                symbol=_ddpos.symbol, direction=_ddpos.direction,
                                entry_price=_ddpos.entry_price, exit_price=cl,
                                stop_loss=_ddpos.stop_loss, take_profit=_ddpos.take_profit,
                                quantity=_ddpos.quantity, risk_amount=_ddpos.risk_amount,
                                pnl=_ddpnl, pnl_r=_ddpnl / max(_ddpos.risk_amount, 1e-9),
                                open_bar=_ddpos.open_bar, close_bar=step,
                                exit_reason="dd_close_all", confidence=_ddpos.confidence,
                                win_probability=_ddpos.win_probability,
                                agent_votes=_ddpos.agent_votes,
                            ))
                    open_positions.clear()
                    logger.info(
                        f"[DD CLOSE-ALL] bar={step} {bar_date:%H:%M} "
                        f"daily_dd={day_dd:.2f}% — closed all positions"
                    )
            _iso_week = bar_date.isocalendar()[1]
            if _current_week != _iso_week:
                _current_week = _iso_week
                _wk_open_equity = equity
                _wk_halted = False
            if not _wk_halted and _wk_dd_pct > 0:
                wk_dd = (equity - _wk_open_equity) / max(_wk_open_equity, 1e-9) * 100
                if wk_dd <= -_wk_dd_pct:
                    _wk_halted = True
                    # ── Weekly DD close-all (same logic as daily) ──────────
                    for _wkpos in list(open_positions):
                        if not _wkpos.pending:
                            _wkpnl = self._pnl_at(_wkpos, cl, tick_val, pip_sz, close_step=step)
                            closed_trades.append(ClosedTrade(
                                symbol=_wkpos.symbol, direction=_wkpos.direction,
                                entry_price=_wkpos.entry_price, exit_price=cl,
                                stop_loss=_wkpos.stop_loss, take_profit=_wkpos.take_profit,
                                quantity=_wkpos.quantity, risk_amount=_wkpos.risk_amount,
                                pnl=_wkpnl, pnl_r=_wkpnl / max(_wkpos.risk_amount, 1e-9),
                                open_bar=_wkpos.open_bar, close_bar=step,
                                exit_reason="weekly_dd_close_all", confidence=_wkpos.confidence,
                                win_probability=_wkpos.win_probability,
                                agent_votes=_wkpos.agent_votes,
                            ))
                    open_positions.clear()
                    logger.info(
                        f"[WEEKLY DD CLOSE-ALL] bar={step} {bar_date:%H:%M} "
                        f"weekly_dd={wk_dd:.2f}% — closed all positions"
                    )

            # ── Weekend close ────────────────────────────────────────────────
            # Friday ≥ cutoff UTC: close all + block new entries until Monday.
            _weekday = bar_date.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
            if _wkend_enabled:
                if _weekday == 4 and bar_date.hour >= _wkend_hour:
                    if not _weekend_blocked:
                        _weekend_blocked = True
                        for _wpos in list(open_positions):
                            if not _wpos.pending:
                                _wpnl = self._pnl_at(_wpos, cl, tick_val, pip_sz, close_step=step)
                                closed_trades.append(ClosedTrade(
                                    symbol=_wpos.symbol, direction=_wpos.direction,
                                    entry_price=_wpos.entry_price, exit_price=cl,
                                    stop_loss=_wpos.stop_loss, take_profit=_wpos.take_profit,
                                    quantity=_wpos.quantity, risk_amount=_wpos.risk_amount,
                                    pnl=_wpnl, pnl_r=_wpnl / max(_wpos.risk_amount, 1e-9),
                                    open_bar=_wpos.open_bar, close_bar=step,
                                    exit_reason="weekend", confidence=_wpos.confidence,
                                    win_probability=_wpos.win_probability,
                                    agent_votes=_wpos.agent_votes,
                                ))
                        open_positions.clear()
                elif _weekday < 4:
                    _weekend_blocked = False

            bid = cl - spread / 2
            ask = cl + spread / 2
            executor.set_price(symbol, bid, ask)

            # ── Fill pending entries at next bar's open (avoids look-ahead) ──
            # Signals fire at bar N close; we fill at bar N+1 open (realistic).
            bar_open = float(mid_bars["open"][step])
            _fill_rejects: List[SimPosition] = []
            for pos in open_positions:
                if pos.pending:
                    # ── Spread guard: reject if spread > 20% of stop distance ──
                    _stop_dist = abs(pos.entry_price - pos.stop_loss)
                    if _stop_dist > 1e-9 and spread / _stop_dist > self._MAX_SPREAD_FRAC:
                        logger.debug(
                            f"[SPREAD-GUARD] bar={step} {bar_date:%H:%M} {symbol} "
                            f"spread={spread:.5f} > {self._MAX_SPREAD_FRAC:.0%} of "
                            f"stop_dist={_stop_dist:.5f} — rejecting"
                        )
                        _fill_rejects.append(pos)
                        continue
                    # ── Lot rounding (mirrors MT5 vol_step) ──
                    _raw_qty = pos.quantity
                    pos.quantity = round(round(_raw_qty / self._VOL_STEP) * self._VOL_STEP, 8)
                    pos.quantity = max(self._VOL_MIN, min(self._VOL_MAX, pos.quantity))
                    # ── Slippage: random ±0.3 pip micro-noise ──
                    _slip = random.uniform(-self._SLIPPAGE_PIPS, self._SLIPPAGE_PIPS) * pip_sz
                    _planned_entry = pos.entry_price
                    if pos.direction == "long":
                        pos.entry_price = bar_open + spread / 2 + _slip  # fill at ask + slip
                    else:
                        pos.entry_price = bar_open - spread / 2 + _slip  # fill at bid + slip

                    _planned_sl_dist = abs(_planned_entry - pos.stop_loss)
                    _actual_sl_dist  = abs(pos.entry_price - pos.stop_loss)
                    logger.debug(
                        f"[FILL] bar={step} {bar_date:%H:%M} {symbol} "
                        f"{pos.direction.upper()} filled "
                        f"@ {pos.entry_price:.5f} (planned={_planned_entry:.5f} "
                        f"open={bar_open:.5f} slip={_slip:+.5f}) "
                        f"qty={pos.quantity:.2f} SL={pos.stop_loss:.5f} TP={pos.take_profit:.5f} "
                        f"sl_dist: plan={_planned_sl_dist/pip_sz:.1f}pip "
                        f"actual={_actual_sl_dist/pip_sz:.1f}pip "
                        f"entry_sl={pos.entry_sl:.5f} "
                        f"1R={abs(pos.entry_price - pos.entry_sl)/pip_sz:.1f}pip "
                        f"risk=${pos.risk_amount:.2f}"
                    )
                    # Debug: write to dump file for post-analysis
                    _dump_f = getattr(self, '_sl_dump', None)
                    if _dump_f:
                        _dump_f.write(
                            f"FILL,{step},{bar_date:%Y-%m-%d %H:%M},{symbol},"
                            f"{pos.direction},{pos.ticket},"
                            f"{pos.entry_price:.6f},{pos.stop_loss:.6f},"
                            f"{pos.take_profit:.6f},{pos.entry_sl:.6f},"
                            f"{abs(pos.entry_price - pos.entry_sl)/pip_sz:.1f},"
                            f"{_actual_sl_dist/pip_sz:.1f},"
                            f"{pos.quantity:.2f},{pos.risk_amount:.2f}\n"
                        )
                    pos.pending = False
                    pos.entry_atr = _atr_sp  # snapshot ATR at fill for adaptive stale_sl
            # Remove rejected fills
            for _rj in _fill_rejects:
                open_positions.remove(_rj)

            # ── Check exits ──────────────────────────────────────────────────
            if _ibs_mode:
                # IBS: step through OHLC waypoints within the bar.
                # On volatile bars (range > 2×ATR), insert intermediate waypoints
                # at 25%/50%/75% levels for higher-fidelity SL/TP resolution.
                if cl >= bar_open:
                    _ibs_wps = [bar_open, lo, hi, cl]     # bullish: dip first
                else:
                    _ibs_wps = [bar_open, hi, lo, cl]     # bearish: spike first
                # Volatile bar: insert intermediate waypoints for denser path
                if _atr_sp > 0 and (hi - lo) > 2.0 * _atr_sp:
                    _mid1 = (_ibs_wps[0] + _ibs_wps[1]) * 0.5
                    _mid2 = (_ibs_wps[1] + _ibs_wps[2]) * 0.5
                    _mid3 = (_ibs_wps[2] + _ibs_wps[3]) * 0.5
                    _ibs_wps = [_ibs_wps[0], _mid1, _ibs_wps[1], _mid2,
                                _ibs_wps[2], _mid3, _ibs_wps[3]]
                _ibs_this_bar = 0
                still_open_ibs: List[SimPosition] = []
                for pos in open_positions:
                    if pos.pending:
                        still_open_ibs.append(pos)
                        continue
                    _ibs_closed = False
                    for _wp in _ibs_wps[1:]:   # skip waypoint[0] (= bar open = fill price)
                        _sl_hit_ibs = ((pos.direction == "long"  and _wp <= pos.stop_loss) or
                                       (pos.direction == "short" and _wp >= pos.stop_loss))
                        _tp_hit_ibs = ((pos.direction == "long"  and _wp >= pos.take_profit) or
                                       (pos.direction == "short" and _wp <= pos.take_profit))
                        if _sl_hit_ibs and _tp_hit_ibs:
                            _sl_hit_ibs, _tp_hit_ibs = True, False  # SL wins (conservative)
                        if not (_sl_hit_ibs or _tp_hit_ibs):
                            continue
                        _ep_ibs  = pos.stop_loss  if _sl_hit_ibs else pos.take_profit
                        _er_ibs  = "sl"           if _sl_hit_ibs else "tp"
                        _pnl_ibs = self._pnl_at(pos, _ep_ibs, tick_val, pip_sz, close_step=step)
                        _pnl_r_ibs = _pnl_ibs / max(pos.risk_amount, 1e-9)
                        logger.info(
                            f"[EXIT-{_er_ibs.upper()}] {pos.symbol} {pos.direction.upper()} "
                            f"entry={pos.entry_price:.5f} exit={_ep_ibs:.5f} "
                            f"pnl={_pnl_ibs:+.2f} ({_pnl_r_ibs:+.2f}R) "
                            f"bars_held={step - pos.open_bar} (ticket={pos.ticket})"
                        )
                        if _sl_hit_ibs:
                            _one_r_ibs = abs(pos.entry_price - (pos.entry_sl or pos.stop_loss))
                            logger.debug(
                                f"[SL-DIAG] ticket={pos.ticket} "
                                f"entry_sl={pos.entry_sl:.5f} current_sl={pos.stop_loss:.5f} "
                                f"1R_dist={_one_r_ibs/pip_sz:.1f}pip "
                                f"sl_dist={abs(pos.entry_price - pos.stop_loss)/pip_sz:.1f}pip "
                                f"qty={pos.quantity:.2f} risk_amt=${pos.risk_amount:.2f} "
                                f"partial1={'Y' if pos.partial_tp1_fired else 'N'} "
                                f"partial2={'Y' if pos.partial_tp2_fired else 'N'}"
                            )
                        closed_trades.append(ClosedTrade(
                            symbol=pos.symbol, direction=pos.direction,
                            entry_price=pos.entry_price, exit_price=_ep_ibs,
                            stop_loss=pos.stop_loss, take_profit=pos.take_profit,
                            quantity=pos.quantity, risk_amount=pos.risk_amount,
                            pnl=_pnl_ibs, pnl_r=_pnl_r_ibs,
                            open_bar=pos.open_bar, close_bar=step,
                            exit_reason=_er_ibs, confidence=pos.confidence,
                            win_probability=pos.win_probability,
                            agent_votes=pos.agent_votes,
                        ))
                        _ibs_closed = True
                        # ── TP hit → momentum continuation re-entry ──────────
                        # Re-enter in the same direction at the TP price.
                        # SL hits: signal failed — NO re-entry (avoids martingale).
                        _mc = risk_limits.max_concurrent_trades
                        if (_tp_hit_ibs
                                and _ibs_this_bar < _ibs_max
                                and not _day_halted and not _wk_halted
                                and not _weekend_blocked
                                and (_max_daily == 0 or _day_trade_count < _max_daily)
                                and (_mc <= 0 or len(still_open_ibs) < _mc)
                                and _last_ibs_order is not None
                                and _last_agent_state is not None):
                            _lo_ref  = _last_ibs_order
                            _tp_ref  = _lo_ref["trade_plan"]
                            _re_dir  = getattr(_tp_ref.recipe.direction, "value",
                                               str(_tp_ref.recipe.direction))
                            if "." in _re_dir:
                                _re_dir = _re_dir.split(".")[-1].lower()
                            _re_sl_d = abs(_lo_ref["entry_price"] - _lo_ref["stop_loss"])
                            _re_tp_d = abs(_lo_ref["entry_price"] - _lo_ref["take_profit"])
                            if _re_sl_d > 1e-7:
                                _re_fill = (_ep_ibs + spread / 2 if _re_dir == "long"
                                            else _ep_ibs - spread / 2)
                                _re_sl   = (_re_fill - _re_sl_d if _re_dir == "long"
                                            else _re_fill + _re_sl_d)
                                _re_tp   = (_re_fill + _re_tp_d if _re_dir == "long"
                                            else _re_fill - _re_tp_d)
                                _re_tkt  = executor._ticket_counter
                                executor._ticket_counter += 1
                                _re_pos  = SimPosition(
                                    symbol=symbol,
                                    direction=_re_dir,
                                    entry_price=_re_fill,
                                    stop_loss=_re_sl,
                                    take_profit=_re_tp,
                                    quantity=_tp_ref.quantity,
                                    risk_amount=_tp_ref.risk_amount,
                                    open_bar=step,
                                    ticket=_re_tkt,
                                    confidence=_tp_ref.confidence,
                                    win_probability=_tp_ref.recipe.win_probability,
                                    agent_votes={
                                        name: float(out.dir_score)
                                        for name, out in
                                        _last_agent_state.get("agent_outputs", {}).items()
                                        if hasattr(out, "dir_score")
                                    },
                                    pending=False,
                                    entry_sl=_re_sl,
                                )
                                still_open_ibs.append(_re_pos)
                                _ibs_this_bar   += 1
                                _day_trade_count += 1
                                _last_entry[symbol] = bar_date
                        break   # done with waypoints for this position
                    if not _ibs_closed:
                        still_open_ibs.append(pos)
                open_positions.clear()
                open_positions.extend(still_open_ibs)
            else:
                # ── Standard exit check ──────────────────────────────────────
                still_open: List[SimPosition] = []
                for pos in open_positions:
                    if pos.pending:
                        still_open.append(pos)   # pending entries can't exit same bar
                        continue
                    exited = self._check_exit(
                        pos, hi, lo, step, tick_val, pip_sz, closed_trades,
                        bar_open=bar_open, bar_close=cl,
                    )
                    if not exited:
                        still_open.append(pos)
                open_positions.clear()
                open_positions.extend(still_open)

            # ── Surveillance step: refresh fusion scores for exit checks ────
            # Mirrors runner._surveillance_loop: every _surv_interval bars,
            # run agents in exit_check_only mode to get a fresh dir_long/mid/short.
            # Only runs when there are open positions (nothing to check otherwise).
            if open_positions and (step % _surv_interval == 0 or _last_surv_state is None):
                try:
                    _surv_feats_by_tf = loader.get_multi_features(symbol)
                    _surv_feat = (
                        _surv_feats_by_tf.get("mid")
                        or _surv_feats_by_tf.get("long")
                        or _surv_feats_by_tf.get("short")
                    )
                    if _surv_feat is not None:
                        _s = await graph.run(
                            symbol=symbol,
                            features=_surv_feat,
                            portfolio_state=None,
                            risk_limits=risk_limits,
                            features_by_tf=_surv_feats_by_tf,
                            exit_check_only=True,
                        )
                        _last_surv_state = _s
                except Exception:
                    pass   # keep old state on error

            # ── Position management: trailing SL, partial TP, windfall, ────
            # time-stop, conviction fade / D1 flip / opposition exits.
            # Mirrors the surveillance loop in runner.py for live-trading parity.
            if open_positions:
                _fusion = (_last_surv_state.get("timeframe_fusion")
                           if _last_surv_state else None)
                _atr_wins  = [mid_bars["high"][i] - mid_bars["low"][i]
                              for i in range(max(0, step - 14), step)]
                _atr       = float(np.mean(_atr_wins)) if _atr_wins else 0.0
                managed_open: List[SimPosition] = []
                for pos in open_positions:
                    if pos.pending:
                        managed_open.append(pos)
                        continue
                    _one_r = abs(pos.entry_price - pos.entry_sl) if pos.entry_sl else 0.0
                    if _one_r < 1e-9:
                        managed_open.append(pos)
                        continue
                    _is_long     = pos.direction == "long"
                    _profit_dist = (cl - pos.entry_price) if _is_long else (pos.entry_price - cl)
                    _profit_r    = _profit_dist / _one_r
                    # Track peak profit for stale winner protection
                    if _profit_r > pos.peak_profit_r:
                        pos.peak_profit_r = _profit_r

                    # Debug: log position state on first management bar and at key stages
                    _bars_open = step - pos.open_bar
                    if _bars_open <= 1:
                        logger.debug(
                            f"[POS-MGR] bar={step} {symbol} {pos.direction} "
                            f"entry={pos.entry_price:.5f} entry_sl={pos.entry_sl:.5f} "
                            f"SL={pos.stop_loss:.5f} TP={pos.take_profit:.5f} "
                            f"1R={_one_r/pip_sz:.1f}pip SL_dist={abs(pos.entry_price - pos.stop_loss)/pip_sz:.1f}pip "
                            f"profit={_profit_r:+.2f}R "
                            f"(ticket={pos.ticket})"
                        )
                    _should_close = False
                    _close_reason = ""

                    # 1. Time-stop
                    if _max_hours_ts > 0:
                        _hours_held = (step - pos.open_bar) * _hrs_per_bar
                        # Counter-trend scalps use a shorter time-stop (default 2h)
                        # since they're fighting D1 and should exit quickly.
                        _is_ct = getattr(pos, "entry_type", "") == "counter-trend-scalp"
                        _eff_ts = _ct_max_hours if _is_ct else _max_hours_ts
                        if _hours_held >= _eff_ts:
                            _should_close = True
                            _close_reason = "time-stop"

                    # 2. Conviction fade / D1 flip / mid+short opposition
                    if not _should_close and _fusion is not None:
                        _d_long  = _fusion.dir_long
                        _d_mid   = _fusion.dir_mid
                        _d_short = _fusion.dir_short
                        if _scalp_exits:
                            if _is_long and _d_short < -_scalp_flip:
                                _should_close = True; _close_reason = "scalp: 1m flip bearish"
                            elif not _is_long and _d_short > _scalp_flip:
                                _should_close = True; _close_reason = "scalp: 1m flip bullish"
                            if not _should_close and _opp_enabled:
                                if _is_long and _d_mid < -_opp_thresh and _d_short < -_opp_thresh:
                                    _should_close = True; _close_reason = "scalp: mid+short bearish"
                                elif not _is_long and _d_mid > _opp_thresh and _d_short > _opp_thresh:
                                    _should_close = True; _close_reason = "scalp: mid+short bullish"
                        else:
                            _is_ct = getattr(pos, "entry_type", "") == "counter-trend-scalp"
                            # D1-flip exit: skip for counter-trend scalps (D1 is
                            # expected to be opposed — that's how they entered).
                            if not _is_ct:
                                if _is_long and _d_long < -_flip_thresh:
                                    _should_close = True; _close_reason = "D1 flipped bearish"
                                elif not _is_long and _d_long > _flip_thresh:
                                    _should_close = True; _close_reason = "D1 flipped bullish"
                            if not _should_close and _fade_enabled and abs(_d_long) < _fade_thresh:
                                if not _is_ct:
                                    _should_close = True; _close_reason = "conviction faded"
                            # Mid+short opposition: always active (also the primary
                            # exit for CT scalps — intraday momentum fading back).
                            if not _should_close and _opp_enabled:
                                if _is_long and _d_mid < -_opp_thresh and _d_short < -_opp_thresh:
                                    _should_close = True; _close_reason = "mid+short opposition"
                                elif not _is_long and _d_mid > _opp_thresh and _d_short > _opp_thresh:
                                    _should_close = True; _close_reason = "mid+short opposition"
                            # CT-specific exit: if MID flips back to agree with D1,
                            # the counter-trend move is over.
                            if not _should_close and _is_ct:
                                _trade_sign = 1.0 if _is_long else -1.0
                                if _d_mid * _trade_sign < -_opp_thresh:
                                    _should_close = True; _close_reason = "CT mid flipped against"

                    if _should_close:
                        _pnl_c = self._pnl_at(pos, cl, tick_val, pip_sz, close_step=step)
                        logger.info(
                            f"[EXIT-{_close_reason.upper().replace(' ', '_')}] {symbol} {pos.direction.upper()} "
                            f"entry={pos.entry_price:.5f} exit={cl:.5f} "
                            f"pnl={_pnl_c:+.2f} ({_pnl_c / max(pos.risk_amount, 1e-9):+.2f}R) "
                            f"(ticket={pos.ticket})"
                        )
                        closed_trades.append(ClosedTrade(
                            symbol=pos.symbol, direction=pos.direction,
                            entry_price=pos.entry_price, exit_price=cl,
                            stop_loss=pos.stop_loss, take_profit=pos.take_profit,
                            quantity=pos.quantity, risk_amount=pos.risk_amount,
                            pnl=_pnl_c, pnl_r=_pnl_c / max(pos.risk_amount, 1e-9),
                            open_bar=pos.open_bar, close_bar=step,
                            exit_reason=_close_reason, confidence=pos.confidence,
                            win_probability=pos.win_probability,
                            agent_votes=pos.agent_votes,
                        ))
                        continue

                    # 2b. Tighten-on-fade: conviction weakening but not close-worthy.
                    # Mirrors runner._check_and_close_positions tighten logic:
                    # lock SL at break-even and pull TP to nearby when dir_long
                    # is between fade_thresh and tighten_thresh and trade is green.
                    if (_tighten_on and _fusion is not None
                            and not _scalp_exits and _atr > 0
                            and _profit_dist > 0
                            and _fade_thresh < abs(_fusion.dir_long) < _tighten_thresh):
                        if _is_long:
                            _t_sl = max(pos.stop_loss, pos.entry_price)
                            _t_tp = cl + _tighten_tp_m * _atr
                            if _t_sl > pos.stop_loss:
                                pos.stop_loss = _t_sl
                            if pos.take_profit <= 0 or _t_tp < pos.take_profit - 0.2 * _atr:
                                pos.take_profit = _t_tp
                        else:
                            _t_sl = min(pos.stop_loss, pos.entry_price) if pos.stop_loss > 0 else pos.entry_price
                            _t_tp = cl - _tighten_tp_m * _atr
                            if pos.stop_loss <= 0 or _t_sl < pos.stop_loss:
                                pos.stop_loss = _t_sl
                            if pos.take_profit <= 0 or _t_tp > pos.take_profit + 0.2 * _atr:
                                pos.take_profit = _t_tp

                    # 3. Windfall exit
                    if _windfall_on and not pos.windfall_fired and _profit_dist >= _windfall_r * _one_r:
                        pos.windfall_fired = True
                        _pnl_wf = self._pnl_at(pos, cl, tick_val, pip_sz, close_step=step)
                        closed_trades.append(ClosedTrade(
                            symbol=pos.symbol, direction=pos.direction,
                            entry_price=pos.entry_price, exit_price=cl,
                            stop_loss=pos.stop_loss, take_profit=pos.take_profit,
                            quantity=pos.quantity, risk_amount=pos.risk_amount,
                            pnl=_pnl_wf, pnl_r=_pnl_wf / max(pos.risk_amount, 1e-9),
                            open_bar=pos.open_bar, close_bar=step,
                            exit_reason="windfall", confidence=pos.confidence,
                            win_probability=pos.win_probability,
                            agent_votes=pos.agent_votes,
                        ))
                        continue

                    # 4. Partial TP1 (at +1R, only in stage 1: 1R ≤ profit < 1.5R)
                    # Guard matches live trailing_stop.py which requires stage==1 — prevents
                    # firing when price gaps straight from below-1R to stage-2/3 in one bar.
                    if (_partial1_on and not pos.partial_tp1_fired
                            and 1.0 * _one_r <= _profit_dist < 1.5 * _one_r):
                        _qty1 = pos.quantity * _partial1_frac
                        # Commission: proportional share of original round-trip
                        # (mirrors MT5 single-ticket model — no extra commission on partials)
                        _p1   = self._pnl_at_qty(pos, cl, _qty1, tick_val, pip_sz)
                        _orig_risk = pos.risk_amount  # snapshot before reduction
                        _part_risk = _orig_risk * _partial1_frac
                        closed_trades.append(ClosedTrade(
                            symbol=pos.symbol, direction=pos.direction,
                            entry_price=pos.entry_price, exit_price=cl,
                            stop_loss=pos.stop_loss, take_profit=pos.take_profit,
                            quantity=_qty1, risk_amount=_part_risk,
                            pnl=_p1, pnl_r=_p1 / max(_part_risk, 1e-9),
                            open_bar=pos.open_bar, close_bar=step,
                            exit_reason="partial_tp1", confidence=pos.confidence,
                            win_probability=pos.win_probability, agent_votes=pos.agent_votes,
                        ))
                        pos.quantity         -= _qty1
                        # Keep risk_amount proportional to remaining qty (not cascading)
                        pos.risk_amount       = _orig_risk * (1.0 - _partial1_frac)
                        pos.partial_tp1_fired = True
                        # Break-even with buffer: leave be_buffer_r × 1R of room
                        # so the remaining qty isn't stopped by spread/noise.
                        _old_sl_p1 = pos.stop_loss
                        if _is_long:
                            pos.stop_loss = pos.entry_price - _be_buffer_r * _one_r
                        else:
                            pos.stop_loss = pos.entry_price + _be_buffer_r * _one_r
                        logger.info(
                            f"[PARTIAL-TP1] bar={step} {symbol} {pos.direction} "
                            f"closed {_partial1_frac:.0%} @ {cl:.5f} (+1R) "
                            f"SL: {_old_sl_p1:.5f}→{pos.stop_loss:.5f} (BE+{_be_buffer_r:.0%}R) "
                            f"rem_qty={pos.quantity:.2f} rem_risk=${pos.risk_amount:.2f} "
                            f"(ticket={pos.ticket})"
                        )

                    # 5. Partial TP2 (at +N×R)
                    if _partial2_on and not pos.partial_tp2_fired and _profit_dist >= _partial2_r * _one_r:
                        _qty2 = pos.quantity * _partial2_frac
                        _p2   = self._pnl_at_qty(pos, cl, _qty2, tick_val, pip_sz)
                        _orig_risk2 = pos.risk_amount
                        _part_risk2 = _orig_risk2 * _partial2_frac
                        closed_trades.append(ClosedTrade(
                            symbol=pos.symbol, direction=pos.direction,
                            entry_price=pos.entry_price, exit_price=cl,
                            stop_loss=pos.stop_loss, take_profit=pos.take_profit,
                            quantity=_qty2, risk_amount=_part_risk2,
                            pnl=_p2, pnl_r=_p2 / max(_part_risk2, 1e-9),
                            open_bar=pos.open_bar, close_bar=step,
                            exit_reason="partial_tp2", confidence=pos.confidence,
                            win_probability=pos.win_probability, agent_votes=pos.agent_votes,
                        ))
                        pos.quantity          -= _qty2
                        pos.risk_amount        = _orig_risk2 * (1.0 - _partial2_frac)
                        pos.partial_tp2_fired  = True
                        _old_sl_p2 = pos.stop_loss
                        _bump = 0.7 * _one_r
                        if _is_long:
                            pos.stop_loss = max(pos.stop_loss, pos.entry_price + _bump)
                        else:
                            pos.stop_loss = min(pos.stop_loss, pos.entry_price - _bump)
                        if pos.stop_loss != _old_sl_p2:
                            logger.info(
                                f"[PARTIAL-TP2] bar={step} {symbol} {pos.direction} "
                                f"closed {_partial2_frac:.0%} @ {cl:.5f} (+{_partial2_r:.1f}R) "
                                f"SL: {_old_sl_p2:.5f}→{pos.stop_loss:.5f} (+0.7R lock) "
                                f"rem_qty={pos.quantity:.2f} rem_risk=${pos.risk_amount:.2f} "
                                f"(ticket={pos.ticket})"
                            )

                    # 6. Trailing SL stages + TP extension (mirrors trailing_stop.py _manage_one)
                    _old_sl = pos.stop_loss
                    _old_tp = pos.take_profit

                    # 6a. Progressive stale-SL tightening:
                    #     Once trade has been losing for stale_sl_hours, SL begins
                    #     tightening from -stale_r × 1R.  Over the next 2× stale_sl_hours
                    #     the cap squeezes to 0.60 × stale_r × 1R.  This gives early
                    #     losers room to develop while progressively reducing risk on
                    #     trades that remain stale.
                    if _stale_sl_hours > 0 and _profit_dist < 0:
                        _hours_held_sl = (step - pos.open_bar) * _hrs_per_bar
                        if _hours_held_sl >= _stale_sl_hours:
                            _old_sl_stale = pos.stop_loss
                            # Progressive factor: 1.0 at threshold → 0.60 at threshold + 2×stale_sl_hours
                            _overtime_frac = min(1.0, (_hours_held_sl - _stale_sl_hours)
                                                      / (2.0 * _stale_sl_hours))
                            # ATR-adaptive: tighten proportionally when vol compresses
                            _atr_ratio = (min(1.0, _atr / pos.entry_atr)
                                          if pos.entry_atr > 1e-9 else 1.0)
                            _prog_r = _stale_sl_r * _atr_ratio * (1.0 - 0.40 * _overtime_frac)
                            if _is_long:
                                _tight_sl = pos.entry_price - _prog_r * _one_r
                                if pos.stop_loss < _tight_sl - 1e-9:
                                    pos.stop_loss = _tight_sl
                            else:
                                _tight_sl = pos.entry_price + _prog_r * _one_r
                                if pos.stop_loss > _tight_sl + 1e-9:
                                    pos.stop_loss = _tight_sl
                            # Only log when SL moved by ≥0.5 pip (suppress 1m bar spam)
                            if abs(pos.stop_loss - _old_sl_stale) >= 0.5 * pip_sz:
                                logger.info(
                                    f"[STALE-SL] bar={step} {symbol} {pos.direction} "
                                    f"open {_hours_held_sl:.1f}h still losing "
                                    f"({_profit_dist/_one_r:+.2f}R) — "
                                    f"SL tightened {_old_sl_stale:.5f}→{pos.stop_loss:.5f} "
                                    f"(cap at {_prog_r:.2f}R, atr_ratio={_atr_ratio:.2f}, "
                                    f"progress={_overtime_frac:.0%}) "
                                    f"(ticket={pos.ticket})"
                                )
                            # Market-close when squeeze is maxed and still losing
                            if _overtime_frac >= 1.0:
                                _should_close = True
                                _close_reason = "stale_market_close"
                                logger.info(
                                    f"[STALE-CLOSE] bar={step} {symbol} {pos.direction} "
                                    f"stale squeeze maxed ({_hours_held_sl:.1f}h, "
                                    f"{_profit_r:+.2f}R) — market close "
                                    f"(ticket={pos.ticket})"
                                )

                    # 6b. Stale winner protection: profitable trade stalling after
                    #     stale_sl_hours — lock SL at BE if profit dropped from peak.
                    if (_stale_sl_hours > 0
                            and _profit_r > 0 and _profit_r < 1.0
                            and pos.peak_profit_r - _profit_r >= 0.3):
                        _hours_held_sw = (step - pos.open_bar) * _hrs_per_bar
                        if _hours_held_sw >= _stale_sl_hours:
                            _old_sl_sw = pos.stop_loss
                            if _is_long:
                                _be_floor = pos.entry_price - _be_buffer_r * _one_r
                                if pos.stop_loss < _be_floor - 1e-9:
                                    pos.stop_loss = _be_floor
                            else:
                                _be_floor = pos.entry_price + _be_buffer_r * _one_r
                                if pos.stop_loss > _be_floor + 1e-9:
                                    pos.stop_loss = _be_floor
                            if pos.stop_loss != _old_sl_sw:
                                logger.info(
                                    f"[STALE-WIN] bar={step} {symbol} {pos.direction} "
                                    f"open {_hours_held_sw:.1f}h, profit stalling "
                                    f"(now {_profit_r:+.2f}R, peak {pos.peak_profit_r:+.2f}R) — "
                                    f"SL→BE {_old_sl_sw:.5f}→{pos.stop_loss:.5f} "
                                    f"(ticket={pos.ticket})"
                                )

                    if _atr > 0:
                        _be_sl_long  = pos.entry_price - _be_buffer_r * _one_r
                        _be_sl_short = pos.entry_price + _be_buffer_r * _one_r
                        if _is_long:
                            if _profit_dist >= 2.0 * _one_r:
                                _new_sl = max(pos.stop_loss, cl - _atr * _atr_mult, pos.entry_price + 0.5 * _one_r)
                                # TP extension: push TP forward when price closes in on it (mirrors live)
                                if (_tp_extend_on and pos.take_profit > 0
                                        and (pos.take_profit - cl) < _atr):
                                    pos.take_profit = cl + _tp_extend_mult * _atr
                            elif _profit_dist >= 1.5 * _one_r:
                                _new_sl = max(pos.stop_loss, pos.entry_price + 0.5 * _one_r)
                            elif _profit_dist >= 1.0 * _one_r:
                                _new_sl = max(pos.stop_loss, _be_sl_long)
                            elif _early_be_r > 0 and _profit_dist >= _early_be_r * _one_r:
                                # Stage 0: early break-even (with buffer)
                                _new_sl = max(pos.stop_loss, _be_sl_long)
                            else:
                                _new_sl = pos.stop_loss
                            if _new_sl > pos.stop_loss:
                                pos.stop_loss = _new_sl
                        else:  # short
                            if _profit_dist >= 2.0 * _one_r:
                                _new_sl = min(pos.stop_loss, cl + _atr * _atr_mult, pos.entry_price - 0.5 * _one_r)
                                # TP extension: push TP forward when price closes in on it (mirrors live)
                                if (_tp_extend_on and pos.take_profit > 0
                                        and (cl - pos.take_profit) < _atr):
                                    pos.take_profit = cl - _tp_extend_mult * _atr
                            elif _profit_dist >= 1.5 * _one_r:
                                _new_sl = min(pos.stop_loss, pos.entry_price - 0.5 * _one_r)
                            elif _profit_dist >= 1.0 * _one_r:
                                _new_sl = min(pos.stop_loss, _be_sl_short)
                            elif _early_be_r > 0 and _profit_dist >= _early_be_r * _one_r:
                                # Stage 0: early break-even (with buffer)
                                _new_sl = min(pos.stop_loss, _be_sl_short)
                            else:
                                _new_sl = pos.stop_loss
                            if _new_sl < pos.stop_loss:
                                pos.stop_loss = _new_sl

                    # 7. Structural pivot ratchet (mirrors trailing_stop._update_structural_one)
                    # Uses D1 swing nodes + weekly pivots to adjust SL/TP to structure.
                    # Gated by skip_structural_sl_tp_ratchet — on 1m bars the "nearest
                    # support/resist" is micro-structure that changes every bar, collapsing
                    # SL to a few pips from entry and producing all-SL exits.
                    if not _skip_struct_ratchet and _atr > 0 and _profit_dist > 0:
                        _struct_buf      = 0.25 * _atr
                        _min_sl_dist     = 1.5 * _atr
                        _live_p = ask if _is_long else bid
                        _pivots = loader.get_structural_levels(symbol, _live_p, _atr)
                        if _is_long:
                            _struct_sl = _pivots["nearest_support"] - _struct_buf
                            _struct_sl = min(_struct_sl, _live_p - _min_sl_dist)
                            if _struct_sl > pos.stop_loss:
                                pos.stop_loss = _struct_sl
                            # TP ratchet: push to next resistance (only raise)
                            _resist_cands = sorted(
                                l for l in (_pivots["r1"], _pivots["r2"])
                                if l > pos.entry_price + _min_sl_dist
                            )
                            if _resist_cands and pos.take_profit > 0:
                                _struct_tp = _resist_cands[0]
                                if _struct_tp > pos.take_profit:
                                    pos.take_profit = _struct_tp
                        else:
                            _struct_sl = _pivots["nearest_resist"] + _struct_buf
                            _struct_sl = max(_struct_sl, _live_p + _min_sl_dist)
                            if _struct_sl < pos.stop_loss:
                                pos.stop_loss = _struct_sl
                            # TP ratchet: push to next support (only lower)
                            _supp_cands = sorted(
                                (l for l in (_pivots["s1"], _pivots["s2"])
                                 if l < pos.entry_price - _min_sl_dist),
                                reverse=True,
                            )
                            if _supp_cands and pos.take_profit > 0:
                                _struct_tp = _supp_cands[0]
                                if _struct_tp < pos.take_profit:
                                    pos.take_profit = _struct_tp

                    # Log trailing SL/TP changes
                    if pos.stop_loss != _old_sl or pos.take_profit != _old_tp:
                        logger.debug(
                            f"[TRAIL] bar={step} {symbol} {pos.direction} "
                            f"profit={_profit_dist/_one_r:.2f}R "
                            f"SL:{_old_sl:.5f}→{pos.stop_loss:.5f} "
                            f"TP:{_old_tp:.5f}→{pos.take_profit:.5f}"
                        )

                    # Late _should_close check (stale_market_close sets it after
                    # the early check at section 2).
                    if _should_close:
                        _pnl_c = self._pnl_at(pos, cl, tick_val, pip_sz, close_step=step)
                        logger.info(
                            f"[EXIT-{_close_reason.upper().replace(' ', '_')}] {symbol} {pos.direction.upper()} "
                            f"entry={pos.entry_price:.5f} exit={cl:.5f} "
                            f"pnl={_pnl_c:+.2f} ({_pnl_c / max(pos.risk_amount, 1e-9):+.2f}R) "
                            f"(ticket={pos.ticket})"
                        )
                        closed_trades.append(ClosedTrade(
                            symbol=pos.symbol, direction=pos.direction,
                            entry_price=pos.entry_price, exit_price=cl,
                            stop_loss=pos.stop_loss, take_profit=pos.take_profit,
                            quantity=pos.quantity, risk_amount=pos.risk_amount,
                            pnl=_pnl_c, pnl_r=_pnl_c / max(pos.risk_amount, 1e-9),
                            open_bar=pos.open_bar, close_bar=step,
                            exit_reason=_close_reason, confidence=pos.confidence,
                            win_probability=pos.win_probability,
                            agent_votes=pos.agent_votes,
                        ))
                        continue

                    managed_open.append(pos)

                open_positions.clear()
                open_positions.extend(managed_open)

            # ── Update equity ──────────────────────────────────────────────
            # Use bid for longs, ask for shorts (mirrors MT5 floating P&L)
            unrealized = sum(
                self._pnl_at(p, bid if p.direction == "long" else ask,
                             tick_val, pip_sz, close_step=step)
                for p in open_positions if not p.pending
            )
            realized   = sum(t.pnl for t in closed_trades)
            equity = initial_equity + realized + unrealized
            equity_curve[-1] = equity   # update the value appended at bar start

            # ── Record closed trades into streak history ───────────────────
            # Any trades closed since the last check get appended to the per-symbol
            # streak tracker (used by the same-direction losing streak guard).
            _n_closed_now = len(closed_trades)
            if _n_closed_now > getattr(self, "_prev_closed_count", 0):
                for _ct in closed_trades[getattr(self, "_prev_closed_count", 0):]:
                    if _ct.exit_reason not in ("partial_tp1", "partial_tp2"):
                        _streak_history.setdefault(_ct.symbol, []).append(
                            (_ct.direction, _ct.pnl, step)
                        )
                        # Check if we just completed a 2-loss streak
                        _sh = _streak_history[_ct.symbol]
                        if (len(_sh) >= 2
                                and _sh[-1][1] < 0 and _sh[-2][1] < 0
                                and _sh[-1][0] == _sh[-2][0]):
                            # Escalating cooldown: double each re-trigger, cap at 24h
                            _prev_count = _streak_block_count.get(_ct.symbol, 0)
                            _streak_block_count[_ct.symbol] = _prev_count + 1
                            _cooldown_bars = min(
                                _streak_base_bars * (2 ** _prev_count),
                                _streak_max_bars,
                            )
                            _streak_block_dir[_ct.symbol] = _sh[-1][0]
                            _streak_block_until[_ct.symbol] = step + _cooldown_bars
            self._prev_closed_count = _n_closed_now

            # ── Refresh portfolio state (daily_pnl from today's open, not backtest start) ──
            portfolio = self._make_portfolio(equity, _day_open_equity, open_positions, mid_bars, step, tick_val, pip_sz)

            # ── Agent run + entry decision ─────────────────────────────────
            # Only invoke the full agent pipeline every _agent_interval bars to
            # avoid running 10+ agents on every single H1 candle (D1 signals
            # don't change every hour).  Cached state is reused on skipped bars.
            executor.placed_orders.clear()
            # Skip entries when any circuit breaker has fired or weekend block is active
            _run_agents = (step % _agent_interval == 0) or (_last_agent_state is None)
            if not _run_agents or _day_halted or _wk_halted or _weekend_blocked:
                if _day_halted or _wk_halted or _weekend_blocked:
                    logger.debug(
                        f"[SKIP] bar={step} {bar_date:%H:%M} day_halt={_day_halted} "
                        f"wk_halt={_wk_halted} weekend={_weekend_blocked}"
                    )
                continue

            # ── Pre-graph entry guards ─────────────────────────────────────
            # Skip the full agent pipeline when the entry would be rejected
            # anyway.  Prevents misleading "Order placed" logs from graph.py
            # and avoids running 14 agents for nothing.
            _skip_entry = False
            if _cooldown_min > 0:
                # Use the MORE RECENT of last accepted entry or last order
                # attempt — this prevents graph.py from re-running when the
                # engine will reject the order anyway (streak guard, etc.).
                _pre_last_t = _last_order_attempt.get(symbol) or _last_entry.get(symbol)
                if _pre_last_t is not None:
                    _pre_elapsed = (bar_date - _pre_last_t).total_seconds() / 60.0
                    if _pre_elapsed < _cooldown_min:
                        _skip_entry = True
            if not _skip_entry and _max_daily > 0 and _day_trade_count >= _max_daily:
                _skip_entry = True
            if not _skip_entry:
                _pre_mc = risk_limits.max_concurrent_trades
                if _pre_mc > 0 and len(open_positions) >= _pre_mc:
                    _skip_entry = True
            # News blackout heuristic: block entry if current bar's range is
            # an extreme outlier (> N×ATR) — proxy for high-impact news events
            # that the live runner would skip via ForexFactory calendar.
            if not _skip_entry and _news_blackout_enabled and _atr_sp > 0:
                _bar_range_v = hi - lo
                if _bar_range_v > _news_atr_mult * _atr_sp:
                    _skip_entry = True
                    logger.debug(
                        f"[NEWS-SPIKE] bar={step} {bar_date:%H:%M} {symbol} "
                        f"range={_bar_range_v:.5f} > {_news_atr_mult:.1f}×ATR={_atr_sp:.5f}"
                    )
            # Margin guard: reject entry if estimated margin usage > 80% of equity
            if not _skip_entry and _leverage > 0:
                _margin_used = sum(
                    p.quantity * p.entry_price * 100_000 / _leverage
                    for p in open_positions if not p.pending
                )
                _free_margin = equity - _margin_used
                if _free_margin < 0.20 * equity:
                    _skip_entry = True
                    logger.debug(
                        f"[MARGIN] bar={step} {bar_date:%H:%M} {symbol} "
                        f"used={_margin_used:.0f} free={_free_margin:.0f} "
                        f"< 20% equity={equity:.0f}"
                    )
            if _skip_entry:
                continue

            try:
                features_by_tf = loader.get_multi_features(symbol)
                features = (
                    features_by_tf.get("mid")
                    or features_by_tf.get("long")
                    or features_by_tf.get("short")
                )
                if features is None:
                    continue

                # ── Debug tracer: begin bar ──
                _get_tracer().begin_bar(
                    bar_idx=step,
                    bar_time=bar_date.strftime("%Y-%m-%d %H:%M"),
                    symbol=symbol,
                )

                # Build recent P&L R-values for streak sizing (last 6 non-partial trades)
                _sym_hist = _streak_history.get(symbol, [])
                # _streak_history stores (dir, pnl_dollar, bar) — use sign of pnl
                _recent_pnl_r = [1.0 if t[1] > 0 else -1.0 for t in _sym_hist[-6:]]

                state = await graph.run(
                    symbol=symbol,
                    features=features,
                    portfolio_state=portfolio,
                    risk_limits=risk_limits,
                    features_by_tf=features_by_tf,
                    recent_pnl_r=_recent_pnl_r,
                )
                _last_agent_state = state

                for order in executor.placed_orders:
                    # ── Max concurrent positions hard guard ────────────────
                    # Graph.py has its own check via portfolio_state, but here
                    # we enforce it directly so no timing gaps allow overfill.
                    _max_conc = risk_limits.max_concurrent_trades
                    if _max_conc > 0 and len(open_positions) >= _max_conc:
                        logger.debug(
                            f"[MAX-CONC] bar={step} {bar_date:%H:%M} {symbol} "
                            f"{len(open_positions)}/{_max_conc} positions — skipping"
                        )
                        break

                    # ── Entry cooldown guard (mirror runner.py behaviour) ──
                    if _cooldown_min > 0:
                        last_t = _last_entry.get(symbol)
                        if last_t is not None:
                            elapsed = (bar_date - last_t).total_seconds() / 60.0
                            if elapsed < _cooldown_min:
                                logger.debug(
                                    f"[COOLDOWN] bar={step} {bar_date:%H:%M} {symbol} "
                                    f"elapsed={elapsed:.0f}min < cooldown={_cooldown_min:.0f}min"
                                )
                                continue

                    # ── Daily trade cap ────────────────────────────────────
                    if _max_daily > 0 and _day_trade_count >= _max_daily:
                        logger.debug(
                            f"[DAY-CAP] bar={step} {bar_date:%H:%M} {symbol} "
                            f"trades_today={_day_trade_count} >= max={_max_daily}"
                        )
                        break

                    tp = order["trade_plan"]
                    # ── Direction: use .value so we get "long"/"short" (not "Direction.SHORT") ──
                    _dir = getattr(tp.recipe.direction, "value", str(tp.recipe.direction))
                    if "." in _dir:          # fallback: "Direction.SHORT" → "short"
                        _dir = _dir.split(".")[-1].lower()

                    # ── Same-direction losing streak guard ─────────────────
                    # Block re-entry in a direction that lost 2+ times in a row.
                    # Escalating cooldown doubles each re-trigger (4h→8h→16h→24h).
                    _blk_dir   = _streak_block_dir.get(symbol)
                    _blk_until = _streak_block_until.get(symbol, 0)
                    if _blk_dir == _dir and step < _blk_until:
                        logger.debug(
                            f"[STREAK] bar={step} {bar_date:%H:%M} {symbol} "
                            f"last 2 {_dir} trades lost — blocking until bar {_blk_until} "
                            f"(streak #{_streak_block_count.get(symbol, 1)})"
                        )
                        continue
                    elif _blk_dir and (step >= _blk_until or _dir != _blk_dir):
                        # Cooldown expired or direction changed — clear block
                        if _dir != _blk_dir:
                            # Direction flipped — reset escalation counter too
                            _streak_block_count.pop(symbol, None)
                        _streak_block_dir.pop(symbol, None)
                        _streak_block_until.pop(symbol, None)

                    # ── Guard: skip orders with zero-pip stop distance ──────────────────────
                    _sl_dist = abs((order["entry_price"] or 0) - (order["stop_loss"] or 0))
                    if _sl_dist < 1e-7:
                        self.logger.debug(
                            f"[{bar_date:%Y-%m-%d %H:%M}] {symbol} zero-pip SL — skipping"
                        )
                        continue
                    pos = SimPosition(
                        symbol=symbol,
                        direction=_dir,
                        entry_price=order["entry_price"],   # overwritten at next bar open
                        stop_loss=order["stop_loss"],
                        take_profit=order["take_profit"],
                        quantity=tp.quantity,
                        risk_amount=tp.risk_amount,
                        open_bar=step + 1,   # filled at NEXT bar
                        ticket=order["order_ticket"],
                        confidence=tp.confidence,
                        win_probability=tp.recipe.win_probability,
                        agent_votes={
                            name: float(out.dir_score)
                            for name, out in state.get("agent_outputs", {}).items()
                            if hasattr(out, "dir_score")
                        },
                        pending=True,   # fill at next bar open (no look-ahead)
                        entry_sl=order["stop_loss"],  # original SL = 1R reference
                        entry_type=state.get("metadata", {}).get("entry_type", "full-alignment"),
                    )
                    open_positions.append(pos)
                    _last_entry[symbol] = bar_date
                    _day_trade_count += 1
                    _last_ibs_order = order   # cache for intra-bar re-entry (IBS mode)
                    self.logger.info(
                        f"[{bar_date:%Y-%m-%d %H:%M}] ENTRY {symbol} "
                        f"{pos.direction.upper()} @ {pos.entry_price:.5f} "
                        f"SL={pos.stop_loss:.5f} TP={pos.take_profit:.5f} "
                        f"(ticket={pos.ticket})"
                    )

                # Track ANY order attempt for pre-graph cooldown
                if executor.placed_orders:
                    _last_order_attempt[symbol] = bar_date

            except Exception as exc:
                self.logger.debug(f"Step {step} error ({symbol}): {exc}")

        # Close any remaining positions at the last bar
        # Restore graph logger level
        _graph_logger.setLevel(_graph_orig_level)
        last_close = float(mid_bars["close"][n_bars - 1])
        for pos in open_positions:
            pnl = self._pnl_at(pos, last_close, tick_val, pip_sz, close_step=n_bars - 1)
            pnl_r = pnl / max(pos.risk_amount, 1e-9)
            closed_trades.append(ClosedTrade(
                symbol=pos.symbol, direction=pos.direction,
                entry_price=pos.entry_price, exit_price=last_close,
                stop_loss=pos.stop_loss, take_profit=pos.take_profit,
                quantity=pos.quantity, risk_amount=pos.risk_amount,
                pnl=pnl, pnl_r=pnl_r,
                open_bar=pos.open_bar, close_bar=n_bars - 1,
                exit_reason="eod", confidence=pos.confidence,
                win_probability=pos.win_probability, agent_votes=pos.agent_votes,
            ))

        final_equity = initial_equity + sum(t.pnl for t in closed_trades)
        if equity_curve:
            equity_curve[-1] = final_equity

        # Stamp open_dt / close_dt on every trade using the full bar_dates list.
        for t in closed_trades:
            if 0 <= t.open_bar < len(bar_dates):
                t.open_dt  = bar_dates[t.open_bar]
            if 0 <= t.close_bar < len(bar_dates):
                t.close_dt = bar_dates[t.close_bar]

        self.logger.info(
            f"Backtest done: {len(closed_trades)} trades  "
            f"equity {initial_equity:.0f} → {final_equity:.0f} "
            f"({(final_equity / initial_equity - 1) * 100:+.2f}%)"
        )

        # ── Debug tracer: record all exits ──
        _tr = _get_tracer()
        for t in closed_trades:
            _bt = t.bar_time if hasattr(t, "bar_time") else ""
            _tr.record_exit(
                bar_idx=t.close_bar,
                bar_time=t.close_dt.strftime("%Y-%m-%d %H:%M") if t.close_dt else "",
                symbol=t.symbol, direction=t.direction,
                entry=t.entry_price, exit_price=t.exit_price,
                sl=t.stop_loss, tp=t.take_profit,
                reason=t.exit_reason, pnl=t.pnl, pnl_r=t.pnl_r,
                bars_held=max(0, t.close_bar - t.open_bar),
            )

        return BacktestResult(
            profile=self.config.profile, symbol=symbol,
            start_date=start_date, end_date=end_date,
            trades=closed_trades,
            equity_curve=equity_curve,
            bar_dates=equity_dates,
            initial_equity=initial_equity,
            config=self.config.model_dump(),
        )

    # ── Macro history pre-loader ──────────────────────────────────────────────

    def _load_macro_history(self, start_date: str, end_date: str) -> Dict:
        """Pre-fetch DXY/VIX/CL/TNX daily closes for the backtest period via yfinance.

        Computes the same signals as the live DataLoader._macro_features() but
        indexed by calendar date so the backtest can look up the historically
        correct value at each bar without look-ahead.

        Returns a dict {datetime.date → {dxy_trend, vix_signal, crude_trend, yield_trend}}.
        Falls back to empty dict (→ neutral zeros) if yfinance is unavailable.
        """
        try:
            import yfinance as yf
        except ImportError:
            return {}

        try:
            from datetime import timedelta, date as _date
            # Extend 30 days before start so 5-day momentum is valid from bar 1
            fetch_start = (
                datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=30)
            ).strftime("%Y-%m-%d")
            fetch_end = (
                datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=2)
            ).strftime("%Y-%m-%d")

            tickers = ["DX-Y.NYB", "^VIX", "CL=F", "^TNX"]
            data = yf.download(
                tickers, start=fetch_start, end=fetch_end,
                interval="1d", progress=False, auto_adjust=True, threads=True,
            )
            closes = data["Close"] if "Close" in data else data
            closes = closes.dropna(how="all")

            # Build column → list of floats (NaN replaced with None for iteration)
            col_vals: Dict[str, List] = {}
            for col in tickers:
                if col in closes.columns:
                    col_vals[col] = [
                        None if (v != v) else float(v)   # nan check
                        for v in closes[col].values
                    ]
                else:
                    col_vals[col] = []

            raw_dates = [
                d.date() if hasattr(d, "date") else d
                for d in closes.index
            ]

            def _mom(col: str, up_to: int, n: int = 5) -> float:
                """5-day momentum of `col` up to index `up_to`, scaled at ±5% → ±1."""
                pts = [v for v in col_vals.get(col, [])[:up_to + 1] if v is not None]
                if len(pts) < n + 1:
                    return 0.0
                ret = (pts[-1] - pts[-(n + 1)]) / max(abs(pts[-(n + 1)]), 1e-9)
                return float(np.clip(ret / 0.05, -1.0, 1.0))

            history: Dict = {}
            for i, d in enumerate(raw_dates):
                # DXY, crude, yield — 5-day momentum
                signals = {
                    "dxy_trend":   _mom("DX-Y.NYB", i),
                    "crude_trend": _mom("CL=F",      i),
                    "yield_trend": _mom("^TNX",      i),
                    "vix_signal":  0.0,
                }
                # VIX: level (fear above 20) and 5-day direction — matches live loader
                vix_pts = [v for v in col_vals.get("^VIX", [])[:i + 1] if v is not None]
                if len(vix_pts) >= 2:
                    vix_level  = float(np.clip((vix_pts[-1] - 20.0) / 15.0, -1.0, 1.0))
                    lb = min(6, len(vix_pts))
                    vix_change = float(np.clip(
                        (vix_pts[-1] - vix_pts[-lb]) / max(vix_pts[-1], 1e-9) / 0.20,
                        -1.0, 1.0,
                    ))
                    signals["vix_signal"] = float(
                        np.clip(vix_level * 0.6 + vix_change * 0.4, -1.0, 1.0)
                    )
                history[d] = signals

            self.logger.info(
                f"Macro history pre-loaded: {len(history)} days "
                f"({fetch_start} → {fetch_end})"
            )
            return history

        except Exception as exc:
            self.logger.warning(
                f"Macro history pre-load failed ({exc}) — "
                f"IntermarketAgent will use neutral zeros during backtest"
            )
            return {}

    # ── Bar pre-loading ───────────────────────────────────────────────────────

    # ── CSV / Parquet bar preload ─────────────────────────────────────────────

    # Pandas resample rules for generating higher timeframes from 1m base.
    _RESAMPLE_RULE: Dict[str, str] = {
        "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
        "1H": "1h", "4H": "4h", "1D": "1D",
    }

    def _preload_bars_csv(
        self, symbol: str, start_date: str, end_date: str
    ) -> Tuple[Dict, str, List[datetime]]:
        """Load OHLCV bars from CSV or Parquet files in ``self._data_dir``.

        Expected directory layout (any of these patterns)::

            data_dir/
              EURUSD_1m.csv          # single TF per file
              EURUSD_1H.csv
              EURUSD_1D.parquet
              EURUSD.csv             # no TF suffix → auto-detect from timestamps
              GBPUSD_1D.csv          # peer symbol for CorrelationAgent

        Column requirements (case-insensitive, order doesn't matter):
          - ``time`` or ``date`` or ``datetime``: ISO-8601, unix timestamp, or
            ``YYYY.MM.DD HH:MM`` (MT5 export format).
          - ``open``, ``high``, ``low``, ``close``
          - ``volume`` (optional — defaults to 1.0)

        When only one file exists for a symbol (e.g. ``EURUSD_1m.csv``), the
        loader automatically resamples it to all required higher timeframes.
        """
        import pandas as pd
        from pathlib import Path

        data_dir = Path(self._data_dir)
        if not data_dir.is_dir():
            self.logger.error(f"[csv] data_dir not found: {data_dir}")
            return {}, "", []

        tf_cfg   = self.config.timeframes
        long_tf  = tf_cfg.long[0]  if tf_cfg.long  else "1D"
        mid_tf   = tf_cfg.mid[0]   if tf_cfg.mid   else "1H"
        short_tf = tf_cfg.short[0] if tf_cfg.short else "15m"

        raw_start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt       = datetime.strptime(end_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

        _WARM_UP_BY_TF = {
            "1m": 3, "5m": 3, "15m": 7, "30m": 7,
            "1H": 30, "4H": 75, "1D": 120,
        }
        warmup_days    = _WARM_UP_BY_TF.get(mid_tf, 100)
        fetch_start_dt = raw_start_dt - timedelta(days=warmup_days)

        # ── Discover & load files ────────────────────────────────────────
        def _read_ohlcv(path: Path) -> Optional[pd.DataFrame]:
            """Read a single CSV or Parquet file, normalise columns."""
            if path.suffix.lower() in (".parquet", ".pq"):
                df = pd.read_parquet(path)
            else:
                # Try common CSV separators
                for sep in (",", "\t", ";"):
                    try:
                        df = pd.read_csv(path, sep=sep)
                        if len(df.columns) >= 5:
                            break
                    except Exception:
                        continue
                else:
                    return None

            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            # Rename common time column variants
            for alias in ("date", "datetime", "timestamp", "dt"):
                if alias in df.columns and "time" not in df.columns:
                    df = df.rename(columns={alias: "time"})
                    break
            if "time" not in df.columns:
                return None

            # Parse time column
            sample = str(df["time"].iloc[0])
            if sample.replace(".", "").replace("-", "").isdigit() and len(sample) >= 10:
                # Unix timestamp (seconds or milliseconds)
                vals = pd.to_numeric(df["time"], errors="coerce")
                if vals.median() > 1e12:
                    vals = vals / 1000  # ms → s
                df.index = pd.to_datetime(vals, unit="s", utc=True)
            else:
                df.index = pd.to_datetime(df["time"], utc=True)
            df.index.name = None

            # Normalise OHLCV columns
            for alias_map in [
                {"o": "open"}, {"h": "high"}, {"l": "low"}, {"c": "close"},
                {"v": "volume", "tick_volume": "volume", "vol": "volume"},
            ]:
                for alias, target in alias_map.items():
                    if target not in df.columns and alias in df.columns:
                        df = df.rename(columns={alias: target})

            for col in ("open", "high", "low", "close"):
                if col not in df.columns:
                    self.logger.warning(f"[csv] missing column '{col}' in {path.name}")
                    return None
            if "volume" not in df.columns:
                df["volume"] = 1.0

            df = df[["open", "high", "low", "close", "volume"]].copy()
            df = df.sort_index().dropna(subset=["open", "high", "low", "close"])
            return df

        def _detect_tf(df: pd.DataFrame) -> str:
            """Infer the base timeframe from median bar spacing."""
            if len(df) < 2:
                return "1H"
            diffs = df.index.to_series().diff().dropna().dt.total_seconds()
            median_sec = diffs.median()
            for tf, secs in [("1m", 60), ("5m", 300), ("15m", 900),
                             ("30m", 1800), ("1H", 3600), ("4H", 14400), ("1D", 86400)]:
                if abs(median_sec - secs) < secs * 0.4:
                    return tf
            return "1H"

        def _resample(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
            """Resample OHLCV to a higher timeframe."""
            rule = self._RESAMPLE_RULE.get(target_tf)
            if rule is None:
                return df
            return (
                df.resample(rule)
                .agg({"open": "first", "high": "max", "low": "min",
                      "close": "last", "volume": "sum"})
                .dropna(subset=["open", "high", "low", "close"])
            )

        # Search for files matching the symbol
        sym_upper = symbol.upper()
        sym_files: Dict[str, pd.DataFrame] = {}   # tf → DataFrame

        # Pattern: {SYMBOL}_{TF}.csv/.parquet  or  {SYMBOL}.csv/.parquet
        for p in sorted(data_dir.iterdir()):
            if p.is_dir() or p.name.startswith("."):
                continue
            stem = p.stem.upper()
            if not stem.startswith(sym_upper):
                continue
            df = _read_ohlcv(p)
            if df is None:
                continue
            # Extract TF from filename suffix: EURUSD_1m → "1m", EURUSD_1H → "1H"
            suffix = stem[len(sym_upper):]
            if suffix.startswith("_"):
                suffix = suffix[1:]
            # Map common variants
            tf_norm = {
                "1M": "1m", "5M": "5m", "15M": "15m", "30M": "30m",
                "1H": "1H", "4H": "4H", "1D": "1D", "D1": "1D",
                "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
                "H1": "1H", "H4": "4H",
            }.get(suffix, "")
            if not tf_norm:
                tf_norm = _detect_tf(df)
                self.logger.info(f"[csv] {p.name}: auto-detected TF={tf_norm}")
            sym_files[tf_norm] = df
            self.logger.info(f"[csv] loaded {p.name}: {len(df)} bars ({tf_norm})")

        # Also scan for peer symbol files (for CorrelationAgent)
        peer = DataLoader._CORR_PEERS.get(sym_upper.replace("m", ""))
        peer_df_d1: Optional[pd.DataFrame] = None
        if peer:
            for p in sorted(data_dir.iterdir()):
                stem = p.stem.upper()
                if stem.startswith(peer.upper()):
                    df = _read_ohlcv(p)
                    if df is not None:
                        detected = _detect_tf(df)
                        if detected == "1D" or "_1D" in stem or "_D1" in stem:
                            peer_df_d1 = df
                        elif peer_df_d1 is None:
                            peer_df_d1 = _resample(df, "1D")

        if not sym_files:
            self.logger.error(f"[csv] no files found for {symbol} in {data_dir}")
            return {}, "", []

        # ── Resample base data to all needed TFs ─────────────────────────
        needed_tfs = {long_tf, mid_tf, short_tf, "1D", "1H"}
        # Find the finest available TF to use as resampling base
        _TF_ORDER = ["1m", "5m", "15m", "30m", "1H", "4H", "1D"]
        base_tf = None
        base_df = None
        for tf in _TF_ORDER:
            if tf in sym_files:
                base_tf = tf
                base_df = sym_files[tf]
                break
        if base_df is None:
            self.logger.error(f"[csv] could not find usable data for {symbol}")
            return {}, "", []

        # For each needed TF, use the matching file or resample from base
        all_bars: Dict[str, Dict[str, np.ndarray]] = {}
        bar_dates: List[datetime] = []

        for tf_name in needed_tfs:
            if tf_name in sym_files:
                df = sym_files[tf_name]
            elif _TF_ORDER.index(tf_name) > _TF_ORDER.index(base_tf) if tf_name in _TF_ORDER and base_tf in _TF_ORDER else False:
                # Resample up from base
                df = _resample(base_df, tf_name)
            else:
                self.logger.debug(f"[csv] cannot produce {tf_name} from {base_tf} base — skipping")
                continue

            # Trim to date range with warmup
            _tf_warmup = _WARM_UP_BY_TF.get(tf_name, 30)
            _tf_start = raw_start_dt - timedelta(days=max(warmup_days, _tf_warmup))
            df = df[(df.index >= _tf_start) & (df.index <= end_dt + timedelta(days=1))]

            if len(df) < 2:
                continue

            timestamps = (df.index.astype(np.int64) // 10**9).values.astype(float)
            bars_dict = {
                "open":   df["open"].values.astype(float),
                "high":   df["high"].values.astype(float),
                "low":    df["low"].values.astype(float),
                "close":  df["close"].values.astype(float),
                "volume": df["volume"].values.astype(float),
                "time":   timestamps,
            }

            # Strip NaN/inf
            valid_mask = (
                np.isfinite(bars_dict["open"]) & np.isfinite(bars_dict["high"])
                & np.isfinite(bars_dict["low"]) & np.isfinite(bars_dict["close"])
            )
            if valid_mask.sum() < 2:
                continue
            bars_dict = {k: v[valid_mask] for k, v in bars_dict.items()}

            key = f"{symbol}_{tf_name}"
            all_bars[key] = bars_dict

            if tf_name == mid_tf:
                bar_dates = [
                    datetime.fromtimestamp(int(t), tz=timezone.utc)
                    for t in bars_dict["time"]
                ]

        # Add peer D1 bars
        if peer and peer_df_d1 is not None and f"{peer}_1D" not in all_bars:
            peer_df_d1 = peer_df_d1[
                (peer_df_d1.index >= raw_start_dt - timedelta(days=120))
                & (peer_df_d1.index <= end_dt + timedelta(days=1))
            ]
            if len(peer_df_d1) >= 2:
                pts = (peer_df_d1.index.astype(np.int64) // 10**9).values.astype(float)
                all_bars[f"{peer}_1D"] = {
                    "open":   peer_df_d1["open"].values.astype(float),
                    "high":   peer_df_d1["high"].values.astype(float),
                    "low":    peer_df_d1["low"].values.astype(float),
                    "close":  peer_df_d1["close"].values.astype(float),
                    "volume": peer_df_d1["volume"].values.astype(float),
                    "time":   pts,
                }

        if not bar_dates:
            self.logger.error(
                f"[csv] could not produce {mid_tf} bars for {symbol} from {data_dir}"
            )
            return {}, "", []

        self.logger.info(
            f"[csv] {symbol}: {len(bar_dates)} {mid_tf} bars "
            f"({start_date}→{end_date}), TFs loaded: "
            f"{[k.split('_',1)[1] for k in all_bars if k.startswith(symbol)]}"
        )
        return all_bars, mid_tf, bar_dates

    # ── yfinance-based bar preload (fallback when MT5 is unavailable) ────────

    def _preload_bars_yfinance(
        self, symbol: str, start_date: str, end_date: str
    ) -> Tuple[Dict, str, List[datetime]]:
        """Download OHLCV bars from yfinance for all required timeframes.

        Used automatically as a fallback when MT5 is not installed, not running,
        or otherwise unreachable.  Key caveats:
          • Short intraday intervals (1m, 5m, 15m, 30m) have strict history limits
            in the yfinance API.  When the requested mid_tf cannot cover the full
            date range, this method automatically degrades to ``1H`` so the backtest
            still executes with realistic bar granularity.
          • 4H is not a native yfinance interval.  1h bars are downloaded and then
            resampled (OHLCV aggregation) to produce the 4H series.
          • FX volume is tick-count (not real lot volume); all volume-dependent
            features are still valid in relative terms.
        """
        try:
            import yfinance as yf
            import pandas as pd
        except ImportError:
            self.logger.error(
                "yfinance (and/or pandas) not installed — "
                "cannot run backtest without MT5.  Install with: pip install yfinance"
            )
            return {}, "", []

        tf_cfg   = self.config.timeframes
        long_tf  = tf_cfg.long[0]  if tf_cfg.long  else "1D"
        mid_tf   = tf_cfg.mid[0]   if tf_cfg.mid   else "1H"
        short_tf = tf_cfg.short[0] if tf_cfg.short else "15m"

        # Map broker symbol → yfinance ticker
        yf_ticker = self._YF_SYMBOL.get(
            symbol.upper().rstrip("Mm"),
            symbol.upper() + "=X",   # fallback: "NOKUSD" → "NOKUSD=X"
        )

        raw_start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt       = datetime.strptime(end_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
        date_range_days = max(1, (end_dt - raw_start_dt).days)

        # Degrade a requested TF to the finest yfinance can serve for the range
        def _resolve_tf(requested: str) -> str:
            yf_iv    = self._YF_INTERVAL.get(requested, "1d")
            max_days = self._YF_MAX_DAYS.get(yf_iv, 99999)
            # Add 120 day buffer for warmup bars that precede start_date
            if date_range_days + 120 > max_days:
                if requested in ("1m", "5m", "15m", "30m", "4H"):
                    return "1H"
            return requested

        effective_mid_tf = _resolve_tf(mid_tf)
        if effective_mid_tf != mid_tf:
            self.logger.warning(
                f"[yfinance] {mid_tf} history covers only "
                f"{self._YF_MAX_DAYS.get(self._YF_INTERVAL.get(mid_tf,'1d'), 99999)}d "
                f"but range is {date_range_days}d — degrading mid_tf: "
                f"{mid_tf} → {effective_mid_tf}"
            )
            mid_tf = effective_mid_tf

        # Warmup: identical to MT5 path
        _WARM_UP_BY_TF = {
            "1m": 3, "5m": 3, "15m": 7, "30m": 7,
            "1H": 30, "4H": 75, "1D": 120,
        }
        warmup_days      = _WARM_UP_BY_TF.get(mid_tf, 100)
        fetch_start_dt   = raw_start_dt - timedelta(days=warmup_days)
        # yfinance end date is exclusive — add 1 day so the last day is included
        fetch_end_dt     = end_dt + timedelta(days=1)

        all_bars: Dict[str, Dict[str, np.ndarray]] = {}
        bar_dates: List[datetime] = []

        needed_tfs = {long_tf, mid_tf, short_tf, "1D", "1H"}

        for tf_name in needed_tfs:
            effective_tf = _resolve_tf(tf_name)
            yf_iv        = self._YF_INTERVAL.get(effective_tf, "1d")
            max_days_iv  = self._YF_MAX_DAYS.get(yf_iv, 99999)

            # Clamp fetch start so we never ask for data beyond yfinance's window
            _range_needed = (fetch_end_dt - fetch_start_dt).days
            if _range_needed > max_days_iv:
                _actual_start = fetch_end_dt - timedelta(days=max_days_iv - 1)
            else:
                _actual_start = fetch_start_dt

            try:
                df = yf.download(
                    yf_ticker,
                    start=_actual_start.strftime("%Y-%m-%d"),
                    end=fetch_end_dt.strftime("%Y-%m-%d"),
                    interval=yf_iv,
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )
            except Exception as exc:
                self.logger.warning(
                    f"[yfinance] download error for {yf_ticker} / {yf_iv}: {exc}"
                )
                continue

            if df is None or df.empty:
                self.logger.warning(
                    f"[yfinance] no data returned for {yf_ticker} / {yf_iv}"
                )
                continue

            # Flatten MultiIndex columns (e.g. when yfinance auto-nests on ticker name)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Resample 1h → 4H when the original request was "4H"
            if tf_name == "4H" and effective_tf == "1H":
                df = (
                    df.resample("4h")
                    .agg({"Open": "first", "High": "max",
                          "Low": "min",  "Close": "last", "Volume": "sum"})
                    .dropna(subset=["Open", "High", "Low", "Close"])
                )

            # Normalise column names: yfinance → lowercase
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            if "adj_close" in df.columns and "close" not in df.columns:
                df = df.rename(columns={"adj_close": "close"})
            for required_col in ("open", "high", "low", "close"):
                if required_col not in df.columns:
                    self.logger.warning(
                        f"[yfinance] missing column '{required_col}' "
                        f"for {yf_ticker}/{yf_iv} — skipping TF"
                    )
                    break
            else:
                pass   # all required columns present — continue below
            if not {"open", "high", "low", "close"}.issubset(set(df.columns)):
                continue
            if "volume" not in df.columns:
                df["volume"] = 1.0

            # Ensure UTC-aware DatetimeIndex → unix timestamps (seconds)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            timestamps = (df.index.astype(np.int64) // 10 ** 9).values.astype(float)

            bars_dict: Dict[str, np.ndarray] = {
                "open":   df["open"].values.astype(float),
                "high":   df["high"].values.astype(float),
                "low":    df["low"].values.astype(float),
                "close":  df["close"].values.astype(float),
                "volume": df["volume"].values.astype(float),
                "time":   timestamps,
            }

            # Strip rows with any NaN / ±inf in OHLC
            valid_mask = (
                np.isfinite(bars_dict["open"])
                & np.isfinite(bars_dict["high"])
                & np.isfinite(bars_dict["low"])
                & np.isfinite(bars_dict["close"])
            )
            if valid_mask.sum() < 2:
                self.logger.warning(
                    f"[yfinance] too few valid bars for {yf_ticker}/{yf_iv} after NaN filter"
                )
                continue
            bars_dict = {k: v[valid_mask] for k, v in bars_dict.items()}

            key = f"{symbol}_{tf_name}"
            all_bars[key] = bars_dict

            if tf_name == mid_tf:
                bar_dates = [
                    datetime.fromtimestamp(int(t), tz=timezone.utc)
                    for t in bars_dict["time"]
                ]

        if not bar_dates:
            self.logger.error(
                f"[yfinance] could not load mid-tier ({mid_tf}) bars for {symbol} "
                f"(yfinance ticker: {yf_ticker})"
            )
            return {}, mid_tf, []

        # ── Load peer-symbol D1 bars for CorrelationAgent ─────────────────
        peer = DataLoader._CORR_PEERS.get(symbol.upper().replace("m", ""))
        if peer and f"{peer}_1D" not in all_bars:
            peer_ticker = self._YF_SYMBOL.get(peer.upper(), peer.upper() + "=X")
            try:
                peer_df = yf.download(
                    peer_ticker,
                    start=fetch_start_dt.strftime("%Y-%m-%d"),
                    end=fetch_end_dt.strftime("%Y-%m-%d"),
                    interval="1d", progress=False, auto_adjust=True, threads=False,
                )
                if peer_df is not None and not peer_df.empty:
                    if isinstance(peer_df.columns, pd.MultiIndex):
                        peer_df.columns = peer_df.columns.get_level_values(0)
                    peer_df.columns = [c.lower().replace(" ", "_") for c in peer_df.columns]
                    if "adj_close" in peer_df.columns and "close" not in peer_df.columns:
                        peer_df = peer_df.rename(columns={"adj_close": "close"})
                    if {"open", "high", "low", "close"}.issubset(set(peer_df.columns)):
                        if "volume" not in peer_df.columns:
                            peer_df["volume"] = 1.0
                        if peer_df.index.tz is None:
                            peer_df.index = peer_df.index.tz_localize("UTC")
                        else:
                            peer_df.index = peer_df.index.tz_convert("UTC")
                        pts = (peer_df.index.astype(np.int64) // 10 ** 9).values.astype(float)
                        all_bars[f"{peer}_1D"] = {
                            "open":   peer_df["open"].values.astype(float),
                            "high":   peer_df["high"].values.astype(float),
                            "low":    peer_df["low"].values.astype(float),
                            "close":  peer_df["close"].values.astype(float),
                            "volume": peer_df["volume"].values.astype(float),
                            "time":   pts,
                        }
                        self.logger.debug(
                            f"[yfinance] loaded {len(peer_df)} D1 peer bars for {peer}"
                        )
            except Exception as exc:
                self.logger.debug(f"[yfinance] peer {peer} D1 download failed: {exc}")

        self.logger.info(
            f"[yfinance backfill] {symbol} ({yf_ticker}): "
            f"{len(bar_dates)} {mid_tf} bars ({start_date}→{end_date})"
        )
        return all_bars, mid_tf, bar_dates

    def _preload_bars(
        self, symbol: str, start_date: str, end_date: str
    ) -> Tuple[Dict, str, List[datetime]]:
        """Pre-load all required timeframe bars.

        Priority: CSV/Parquet (if _data_dir set) → MT5 → yfinance.
        """
        # ── CSV / Parquet path (highest priority when --data-dir given) ──
        if self._data_dir:
            result = self._preload_bars_csv(symbol, start_date, end_date)
            if result[1]:  # mid_tf non-empty means success
                return result
            self.logger.warning(
                f"[csv] failed for {symbol} — falling back to MT5/yfinance"
            )

        tf_cfg  = self.config.timeframes
        long_tf  = tf_cfg.long[0]  if tf_cfg.long  else "1D"
        mid_tf   = tf_cfg.mid[0]   if tf_cfg.mid   else "1H"
        short_tf = tf_cfg.short[0] if tf_cfg.short else "15m"

        try:
            from mt5linux import MetaTrader5 as mt5lib
            mt5 = mt5lib(
                host=self.config.mt5.model_dump().get("host", "localhost"),
                port=int(self.config.mt5.model_dump().get("port", 18812)),
            )
        except ImportError:
            try:
                import MetaTrader5 as mt5
            except ImportError:
                self.logger.warning(
                    "No MT5 library found — falling back to yfinance backfill"
                )
                return self._preload_bars_yfinance(symbol, start_date, end_date)
        except Exception as _conn_exc:
            # mt5linux constructor raises when the Docker container/server is not reachable
            self.logger.warning(
                f"MT5 connection error ({_conn_exc}) — falling back to yfinance backfill"
            )
            return self._preload_bars_yfinance(symbol, start_date, end_date)

        tf_map = {
            "1m":  mt5.TIMEFRAME_M1,  "5m":  mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15, "30m": mt5.TIMEFRAME_M30,
            "1H":  mt5.TIMEFRAME_H1,  "4H":  mt5.TIMEFRAME_H4,
            "1D":  mt5.TIMEFRAME_D1,
        }

        from datetime import timedelta
        # Extend start date backwards to provide warm-up bars for indicator computation.
        # Warmup is timeframe-aware: short TFs need far fewer calendar days than H4/D1.
        # M1/M5 → 3 days (≈ 4320 / 864 bars);  M15/M30 → 7 days;  H1/H4 → 30 days;  D1 → 100 days.
        _WARM_UP_BY_TF = {
            "1m": 3, "5m": 3, "15m": 7, "30m": 7,
            "1H": 30, "4H": 75, "1D": 120,
        }
        _WARM_UP_DAYS = _WARM_UP_BY_TF.get(mid_tf, 100)
        raw_start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_dt = raw_start_dt - timedelta(days=_WARM_UP_DAYS)
        end_dt   = datetime.strptime(end_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

        cfg = self.config.mt5.model_dump()
        init_kwargs: Dict[str, Any] = {}
        if cfg.get("login"):    init_kwargs["login"]    = int(cfg["login"])
        if cfg.get("password"): init_kwargs["password"] = cfg["password"]
        if cfg.get("server"):   init_kwargs["server"]   = cfg["server"]

        if not mt5.initialize(**init_kwargs):
            self.logger.warning(
                f"MT5 init failed: {mt5.last_error()} — falling back to yfinance backfill"
            )
            return self._preload_bars_yfinance(symbol, start_date, end_date)

        if not mt5.symbol_select(symbol, True):
            self.logger.warning(f"Symbol {symbol} not selectable; continuing anyway")

        all_bars: Dict[str, Dict[str, np.ndarray]] = {}
        bar_dates: List[datetime] = []

        needed_tfs = {long_tf, mid_tf, short_tf, "1D", "1H"}  # 1H always loaded for SL/TP sizing
        # Each timeframe needs enough historical bars for its own indicator computation,
        # independent of the mid_tf granularity chosen for the bar loop.
        # E.g. running balanced with --mid-tf 1m gives _WARM_UP_DAYS=3, but:
        #   D1 needs ≥20 bars  → requires ≥60 calendar days of warmup
        #   15m needs ≥250 bars → requires ≥7 calendar days of warmup
        #   1H  needs ≥250 bars → requires ≥30 calendar days (already handled below)
        # Without this, long-tier features are always None when --mid-tf 1m is used
        # for balanced/safe/risky profiles, resulting in 0 trades.
        _MIN_WARMUP_BY_TF = {
            "1m":   3,
            "5m":   3,
            "15m":  7,    # 7 days × 24h × 4 = 672 bars >> 250 minimum
            "30m":  7,
            "1H":   30,   # handled separately below as _WARM_UP_DAYS_1H
            "4H":   30,   # 30 days × 6 sessions = 180 4H bars >> 40 minimum
            "1D":   60,   # 60 days >> 20 minimum; buffer for weekends & holidays
            "1W":   120,  # 16+ weeks
        }
        # 1H needs at least 250 bars for feature computation (_MIN_BARS=250).
        # 250 1H bars = ~11 trading days. Use 30 days warmup to guarantee this
        # even when the primary mid_tf would only request a shorter warmup.
        _WARM_UP_DAYS_1H = max(_WARM_UP_DAYS, 30)
        for tf_name in needed_tfs:
            mt5_tf = tf_map.get(tf_name)
            if mt5_tf is None:
                continue
            # Use the larger of: profile-driven warmup OR the per-TF minimum needed
            # to satisfy the _tf_min_bars check in _BacktestDataLoader._load_bars.
            _tf_min_warmup = _MIN_WARMUP_BY_TF.get(tf_name, 30)
            if tf_name == "1H":
                tf_start_dt = raw_start_dt - timedelta(days=_WARM_UP_DAYS_1H)
            else:
                tf_start_dt = raw_start_dt - timedelta(days=max(_WARM_UP_DAYS, _tf_min_warmup))
            rates = mt5.copy_rates_range(symbol, mt5_tf, tf_start_dt, end_dt)
            if rates is None or len(rates) == 0:
                self.logger.warning(f"No bars for {symbol}/{tf_name} in range")
                continue
            key = f"{symbol}_{tf_name}"
            # MT5 returns a numpy structured array
            all_bars[key] = {
                "close":  rates["close"].astype(float),
                "high":   rates["high"].astype(float),
                "low":    rates["low"].astype(float),
                "open":   rates["open"].astype(float),
                "volume": rates["tick_volume"].astype(float),
                "time":   rates["time"].astype(float),
            }
            if tf_name == mid_tf:
                bar_dates = [
                    datetime.fromtimestamp(int(t), tz=timezone.utc)
                    for t in rates["time"]
                ]

        # ── Load peer-symbol D1 bars for CorrelationAgent ─────────────────
        peer = DataLoader._CORR_PEERS.get(symbol.upper().replace("m", ""))
        if peer and f"{peer}_1D" not in all_bars:
            mt5_tf_d1 = tf_map.get("1D")
            if mt5_tf_d1 is not None:
                if mt5.symbol_select(peer, True):
                    peer_start = raw_start_dt - timedelta(days=120)
                    peer_rates = mt5.copy_rates_range(peer, mt5_tf_d1, peer_start, end_dt)
                    if peer_rates is not None and len(peer_rates) > 0:
                        all_bars[f"{peer}_1D"] = {
                            "close":  peer_rates["close"].astype(float),
                            "high":   peer_rates["high"].astype(float),
                            "low":    peer_rates["low"].astype(float),
                            "open":   peer_rates["open"].astype(float),
                            "volume": peer_rates["tick_volume"].astype(float),
                            "time":   peer_rates["time"].astype(float),
                        }
                        self.logger.debug(f"Loaded {len(peer_rates)} D1 peer bars for {peer}")

        mt5.shutdown()

        if not bar_dates:
            self.logger.warning(
                f"MT5 returned no {mid_tf} bars for {symbol} — "
                f"falling back to yfinance backfill"
            )
            return self._preload_bars_yfinance(symbol, start_date, end_date)

        self.logger.info(
            f"Loaded {len(bar_dates)} {mid_tf} bars for {symbol} "
            f"({start_date}→{end_date})"
        )
        return all_bars, mid_tf, bar_dates

    # ── Exit evaluation ───────────────────────────────────────────────────────

    def _check_exit(
        self,
        pos: SimPosition,
        bar_high: float,
        bar_low: float,
        step: int,
        tick_val: float,
        pip_sz: float,
        closed_trades: List[ClosedTrade],
        bar_open: float = 0.0,
        bar_close: float = 0.0,
    ) -> bool:
        """Check whether *pos* is stopped out or takes profit on this bar.

        Same-bar SL/TP resolution
        -------------------------
        When both SL and TP are touched within the same bar, the old logic
        always awarded SL (pessimistic).  MT5 fills whichever price level
        the market reaches **first** within the bar.

        We infer intra-bar direction from the OHLC shape:
          * Bullish bar (C ≥ O): price likely dipped first → O→L→H→C
          * Bearish bar (C < O): price likely spiked first → O→H→L→C

        For **longs**:  SL is below entry, TP above.
          - Bullish bar: hits L first → SL tested first; if SL survives, TP tested at H.
          - Bearish bar: hits H first → TP tested first; if TP survives, SL tested at L.

        For **shorts**: SL is above entry, TP below.
          - Bullish bar: hits L first → TP tested first; if TP survives, SL tested at H.
          - Bearish bar: hits H first → SL tested first; if SL survives, TP tested at L.

        This is the same approach the IBS (intra-bar simulation) path already
        uses, now applied to the standard non-IBS exit check as well.
        """
        sl_hit = tp_hit = False
        if pos.direction == "long":
            sl_hit = bar_low  <= pos.stop_loss
            tp_hit = bar_high >= pos.take_profit > 0
        else:
            sl_hit = bar_high >= pos.stop_loss > 0
            tp_hit = bar_low  <= pos.take_profit

        # ── Path-based resolution when both are hit on the same bar ──────
        if sl_hit and tp_hit:
            bullish = (bar_close >= bar_open) if (bar_open > 0 and bar_close > 0) else True
            if pos.direction == "long":
                # Bullish: dip first (L before H) → SL tested first
                # Bearish: spike first (H before L) → TP tested first
                if bullish:
                    tp_hit = False   # SL hit at the dip, TP never reached
                else:
                    sl_hit = False   # TP hit at the spike, SL never reached
            else:  # short
                # Bullish: dip first (L before H) → TP tested first (L=TP for short)
                # Bearish: spike first (H before L) → SL tested first (H=SL for short)
                if bullish:
                    sl_hit = False   # TP hit at the dip, SL never reached
                else:
                    tp_hit = False   # SL hit at the spike, TP never reached

        if sl_hit:
            exit_price, exit_reason = pos.stop_loss, "sl"
        elif tp_hit:
            exit_price, exit_reason = pos.take_profit, "tp"
        else:
            return False

        pnl = self._pnl_at(pos, exit_price, tick_val, pip_sz, close_step=step)
        _pnl_r = pnl / max(pos.risk_amount, 1e-9)
        _one_r_exit = abs(pos.entry_price - (pos.entry_sl or pos.stop_loss))
        _sl_dist_exit = abs(pos.entry_price - pos.stop_loss)
        logger.info(
            f"[EXIT-{exit_reason.upper()}] {pos.symbol} {pos.direction.upper()} "
            f"entry={pos.entry_price:.5f} exit={exit_price:.5f} "
            f"pnl={pnl:+.2f} ({_pnl_r:+.2f}R) "
            f"bars_held={step - pos.open_bar} (ticket={pos.ticket})"
        )
        if exit_reason == "sl":
            logger.debug(
                f"[SL-DIAG] ticket={pos.ticket} "
                f"entry_sl={pos.entry_sl:.5f} current_sl={pos.stop_loss:.5f} "
                f"1R_dist={_one_r_exit/pip_sz:.1f}pip sl_dist={_sl_dist_exit/pip_sz:.1f}pip "
                f"qty={pos.quantity:.2f} risk_amt=${pos.risk_amount:.2f} "
                f"partial1={'Y' if pos.partial_tp1_fired else 'N'} "
                f"partial2={'Y' if pos.partial_tp2_fired else 'N'}"
            )
        closed_trades.append(ClosedTrade(
            symbol=pos.symbol, direction=pos.direction,
            entry_price=pos.entry_price, exit_price=exit_price,
            stop_loss=pos.stop_loss, take_profit=pos.take_profit,
            quantity=pos.quantity, risk_amount=pos.risk_amount,
            pnl=pnl, pnl_r=_pnl_r,
            open_bar=pos.open_bar, close_bar=step,
            exit_reason=exit_reason, confidence=pos.confidence,
            win_probability=pos.win_probability, agent_votes=pos.agent_votes,
        ))
        return True

    # ── P&L helpers ───────────────────────────────────────────────────────────

    def _pnl_at(
        self,
        pos: SimPosition,
        price: float,
        tick_val: float,
        pip_sz: float,
        close_step: int = -1,
    ) -> float:
        pips = (price - pos.entry_price) / pip_sz
        if pos.direction == "short":
            pips = -pips
        gross = pips * tick_val * pos.quantity
        # Commission: round-trip cost per lot (mirrors real ECN broker)
        gross -= self._commission(pos.symbol, pos.quantity)
        # Swap cost: direction-aware overnight financing.
        # Trading days → calendar days via 7/5 (accounts for Wed triple-swap).
        if close_step >= 0:
            bars_held = max(0, close_step - pos.open_bar)
            if bars_held > 0:
                hrs_per_bar = getattr(self, "_hrs_per_bar", 1.0)
                trading_days = int(bars_held * hrs_per_bar / 24)
                if trading_days > 0:
                    # Calendar correction: 7 swap charges per 5 trading days
                    calendar_days = trading_days * 7.0 / 5.0
                    sym = pos.symbol.upper().rstrip("Mm")
                    rates = self._SWAP_RATE.get(sym, (-3.0, -1.0))
                    rate = rates[0] if pos.direction == "long" else rates[1]
                    swap = rate * pos.quantity * calendar_days
                    gross += swap
        return gross

    def _pnl_at_qty(
        self,
        pos: SimPosition,
        price: float,
        qty: float,
        tick_val: float,
        pip_sz: float,
    ) -> float:
        """P&L for a *partial* close of `qty` lots (no swap — short-term partial)."""
        pips = (price - pos.entry_price) / pip_sz
        if pos.direction == "short":
            pips = -pips
        gross = pips * tick_val * qty
        # Commission on the partial quantity
        gross -= self._commission(pos.symbol, qty)
        return gross

    def _pip_size(self, symbol: str) -> float:
        sym = symbol.upper()
        if "JPY" in sym:
            return 0.01
        if any(x in sym for x in ("DAX", "UK100", "US30", "US500", "USTEC", "XAU", "GOLD")):
            return 1.0
        return 0.0001

    def _spread(self, symbol: str) -> float:
        """Return the base (calm-market) spread for *symbol*."""
        return self._BASE_SPREAD.get(symbol.upper().rstrip("Mm"), 0.0001)

    def _dynamic_spread(
        self, symbol: str, bar_range: float, atr_14: float,
    ) -> float:
        """Spread widened proportionally to bar volatility vs ATR-14.

        When a bar's range (hi-lo) exceeds the recent median range (≈ ATR),
        the spread widens linearly up to ``_SPREAD_WIDEN_CAP × base``.
        This models real broker behaviour: during fast moves/news, spreads
        widen and slippage increases.

        Parameters
        ----------
        bar_range : float   hi - lo of the current bar
        atr_14    : float   14-period ATR (median bar range proxy)
        """
        base = self._spread(symbol)
        if atr_14 <= 0 or bar_range <= 0:
            return base
        ratio = bar_range / atr_14  # 1.0 = average bar; >1 = volatile
        # multiplier: 1.0× at ratio ≤ 1, linear to cap at ratio = cap
        mult = min(max(ratio, 1.0), self._SPREAD_WIDEN_CAP)
        return base * mult

    def _commission(self, symbol: str, quantity: float) -> float:
        """Round-trip commission cost in account currency."""
        per_lot = self._COMMISSION_PER_LOT.get(
            symbol.upper().rstrip("Mm"), 7.0
        )
        return per_lot * quantity

    def _tick_value(self, symbol: str) -> float:
        return self._TICK_VALUE.get(symbol.upper().rstrip("Mm"), 10.0)

    def _build_positions_map(
        self,
        open_positions: List["SimPosition"],
        mid_bars: Dict[str, np.ndarray],
        step: int,
        tick_val: float,
        pip_sz: float,
    ) -> dict:
        """Build a symbol→[positions] map that correctly accumulates
        multiple positions on the same symbol (e.g. after scale-in).
        The old dict comprehension silently dropped all but the last
        position per symbol."""
        result: dict = {}
        close_price = float(
            mid_bars["close"][min(step, len(mid_bars["close"]) - 1)]
        )
        for p in open_positions:
            profit = self._pnl_at(p, close_price, tick_val, pip_sz, close_step=step)
            entry = {
                "ticket": p.ticket,
                "type": "BUY" if p.direction == "long" else "SELL",
                "profit": profit,
                "price_open": p.entry_price,
            }
            result.setdefault(p.symbol, []).append(entry)
        return result

    def _make_portfolio(
        self,
        equity: float,
        initial: float,
        open_positions: List[SimPosition],
        mid_bars: Dict[str, np.ndarray],
        step: int,
        tick_val: float,
        pip_sz: float,
    ) -> PortfolioState:
        dd = (equity - initial) / max(initial, 1.0)
        return PortfolioState(
            equity=equity,
            margin_used=0.0,
            free_margin=equity,
            daily_pnl=equity - initial,
            daily_drawdown=dd,
            open_positions=[p.symbol for p in open_positions],
            open_positions_map=self._build_positions_map(
                open_positions, mid_bars, step, tick_val, pip_sz
            ),
            max_daily_drawdown=dd,
            leverage_used=0.0,
            n_portfolio_symbols=getattr(self, "_n_portfolio_symbols", 1),
        )

    def _make_portfolio_multi(
        self,
        equity: float,
        initial: float,
        open_positions: List["SimPosition"],
        sym_ctx: Dict[str, dict],
        sym_ts_to_idx: Dict[str, dict],
        bar_ts: float,
    ) -> "PortfolioState":
        """Portfolio state with correct per-symbol pricing (portfolio mode)."""
        dd = (equity - initial) / max(initial, 1.0)
        pos_map: dict = {}
        for p in open_positions:
            ctx = sym_ctx.get(p.symbol)
            if ctx is None:
                continue
            _idx = sym_ts_to_idx[p.symbol].get(bar_ts)
            if _idx is None:
                _idx = len(ctx["mid_bars"]["close"]) - 1
            _cl = float(ctx["mid_bars"]["close"][min(_idx, len(ctx["mid_bars"]["close"]) - 1)])
            _tv = ctx["tick_val"]
            _ps = ctx["pip_sz"]
            profit = self._pnl_at(p, _cl, _tv, _ps, close_step=_idx)
            entry = {
                "ticket": p.ticket,
                "type": "BUY" if p.direction == "long" else "SELL",
                "profit": profit,
                "price_open": p.entry_price,
            }
            pos_map.setdefault(p.symbol, []).append(entry)
        return PortfolioState(
            equity=equity,
            margin_used=0.0,
            free_margin=equity,
            daily_pnl=equity - initial,
            daily_drawdown=dd,
            open_positions=[p.symbol for p in open_positions],
            open_positions_map=pos_map,
            max_daily_drawdown=dd,
            leverage_used=0.0,
            n_portfolio_symbols=getattr(self, "_n_portfolio_symbols", 1),
        )
