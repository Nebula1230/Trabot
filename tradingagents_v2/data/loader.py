"""
Market data loader using MetaTrader 5.
Fetches multi-timeframe OHLCV bars and computes TechnicalFeatures.
"""

from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timezone
import logging
import time as _time
import numpy as np

from ..core.types import TechnicalFeatures, MarketData


def _ema(series: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA using numpy."""
    alpha = 2.0 / (period + 1)
    result = np.empty_like(series)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def _rsi(close: np.ndarray, period: int) -> float:
    """Compute RSI for the latest bar."""
    deltas = np.diff(close[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Compute ATR series."""
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    atr = np.empty(len(tr))
    atr[0] = tr[0]
    alpha = 1.0 / period
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return atr


def _macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram) as scalar values for latest bar."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line[-1], signal_line[-1], histogram[-1]


def _rsi_series(close: np.ndarray, period: int) -> np.ndarray:
    """
    Compute the full RSI array using Wilder's smoothing.
    Returns an array of length len(close)-1 (one element shorter than close).
    Index i of the result corresponds to close[i+1].
    """
    deltas = np.diff(close)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = np.zeros(len(gains), dtype=float)
    avg_l  = np.zeros(len(losses), dtype=float)
    if len(gains) >= period:
        avg_g[period - 1] = gains[:period].mean()
        avg_l[period - 1] = losses[:period].mean()
        for i in range(period, len(gains)):
            avg_g[i] = (avg_g[i - 1] * (period - 1) + gains[i]) / period
            avg_l[i] = (avg_l[i - 1] * (period - 1) + losses[i]) / period
    rs  = np.where(avg_l > 0, avg_g / np.maximum(avg_l, 1e-12), 100.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:max(period - 1, 0)] = 50.0   # warm-up period
    return rsi


def _divergence_score(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    lookback: int = 5,
    rsi_period: int = 14,
) -> tuple:
    """
    Detect regular and hidden RSI divergences on recent price bars.

    Returns (bull_score, bear_score) both in [0, 1].

    Regular bullish  — price lower-low  + RSI higher-low  → reversal up signal.
    Regular bearish  — price higher-high + RSI lower-high  → reversal down signal.
    Hidden  bullish  — price higher-low  + RSI lower-low   → continuation up.
    Hidden  bearish  — price lower-high  + RSI higher-high → continuation down.
    """
    n = len(close)
    if n < rsi_period + 2 * lookback + 5:
        return 0.0, 0.0

    # Full RSI series aligned with price: rsi_arr[j] ↔ close[j+1]
    rsi_arr = _rsi_series(close, rsi_period)

    def _rsi_at(idx: int) -> float:
        """RSI value at bar index *idx* of the close/high/low arrays."""
        ri = idx - 1
        if 0 <= ri < len(rsi_arr):
            return float(rsi_arr[ri])
        return 50.0

    # Find swing-low and swing-high bar indices.
    # confirmed_end: bars at index >= confirmed_end cannot yet be confirmed as swings
    # because doing so would require future (unseen) bars — look-ahead guard.
    confirmed_end = n - lookback
    sw_low_idx:  list = []
    sw_high_idx: list = []
    for i in range(lookback, confirmed_end):
        if low[i]  == low[i - lookback:i + lookback + 1].min():
            sw_low_idx.append(i)
        if high[i] == high[i - lookback:i + lookback + 1].max():
            sw_high_idx.append(i)

    bull_score = 0.0
    bear_score = 0.0

    # ── Bullish signals (swing lows) ──────────────────────────────────────
    if len(sw_low_idx) >= 2:
        i1, i2       = sw_low_idx[-2], sw_low_idx[-1]   # older, newer
        p1, r1       = float(low[i1]),  _rsi_at(i1)
        p2, r2       = float(low[i2]),  _rsi_at(i2)
        prange       = max(abs(p1), abs(p2), 1e-9)

        # Regular bullish: price LL + RSI HL
        if p2 < p1 and r2 > r1:
            depth    = (p1 - p2) / prange             # how deep the new low is
            recovery = (r2 - r1) / 100.0              # RSI recovery
            bull_score = max(bull_score, float(np.clip((depth + recovery) * 4.0, 0.0, 1.0)))

        # Hidden bullish: price HL + RSI LL (continuation in uptrend)
        elif p2 > p1 and r2 < r1:
            rsi_drop  = (r1 - r2) / 100.0
            bull_score = max(bull_score, float(np.clip(rsi_drop * 3.0, 0.0, 0.75)))

    # ── Bearish signals (swing highs) ─────────────────────────────────────
    if len(sw_high_idx) >= 2:
        i1, i2       = sw_high_idx[-2], sw_high_idx[-1]
        p1, r1       = float(high[i1]), _rsi_at(i1)
        p2, r2       = float(high[i2]), _rsi_at(i2)
        prange       = max(abs(p1), abs(p2), 1e-9)

        # Regular bearish: price HH + RSI LH
        if p2 > p1 and r2 < r1:
            surge      = (p2 - p1) / prange
            fade       = (r1 - r2) / 100.0
            bear_score = max(bear_score, float(np.clip((surge + fade) * 4.0, 0.0, 1.0)))

        # Hidden bearish: price LH + RSI HH (continuation in downtrend)
        elif p2 < p1 and r2 > r1:
            rsi_surge  = (r2 - r1) / 100.0
            bear_score = max(bear_score, float(np.clip(rsi_surge * 3.0, 0.0, 0.75)))

    return float(bull_score), float(bear_score)


def _hurst(close: np.ndarray, min_lag: int = 2, max_lag: int = 20) -> float:
    """Estimate Hurst exponent via R/S analysis on the last 100 bars."""
    series = close[-100:] if len(close) >= 100 else close
    lags = range(min_lag, min(max_lag, len(series) // 2))
    rs_values = []
    for lag in lags:
        chunks = [series[i:i + lag] for i in range(0, len(series) - lag, lag)]
        if not chunks:
            continue
        rs_chunk = []
        for chunk in chunks:
            mean = chunk.mean()
            deviations = np.cumsum(chunk - mean)
            r = deviations.max() - deviations.min()
            s = chunk.std()
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            rs_values.append((lag, np.mean(rs_chunk)))
    if len(rs_values) < 2:
        return 0.5
    lags_arr = np.log([v[0] for v in rs_values])
    rs_arr = np.log([v[1] for v in rs_values])
    hurst = np.polyfit(lags_arr, rs_arr, 1)[0]
    return float(np.clip(hurst, 0.0, 1.0))


def _detect_swings(high: np.ndarray, low: np.ndarray, lookback: int = 5):
    """Detect swing highs/lows; return (swing_highs, swing_lows, hh_hl, lh_ll, last_break).

    Look-ahead guard: only bars with at least *lookback* subsequent closed bars are
    evaluated.  The variable ``confirmed_end = n - lookback`` is the exclusive upper
    bound — bars at or beyond this index would need future price data to be confirmed
    as swings, so they are intentionally excluded.
    """
    n = len(high)
    sh, sl = [], []
    confirmed_end = n - lookback  # bars beyond here need future bars to confirm
    for i in range(lookback, confirmed_end):
        if high[i] == high[i - lookback:i + lookback + 1].max():
            sh.append(float(high[i]))
        if low[i] == low[i - lookback:i + lookback + 1].min():
            sl.append(float(low[i]))

    hh_hl = 0
    if len(sh) >= 2 and sh[-1] > sh[-2]:
        hh_hl += 1
    if len(sl) >= 2 and sl[-1] > sl[-2]:
        hh_hl += 1

    lh_ll = 0
    if len(sh) >= 2 and sh[-1] < sh[-2]:
        lh_ll += 1
    if len(sl) >= 2 and sl[-1] < sl[-2]:
        lh_ll += 1

    last_break = None
    if hh_hl > lh_ll:
        last_break = "bullish"
    elif lh_ll > hh_hl:
        last_break = "bearish"

    return sh[-5:], sl[-5:], hh_hl, lh_ll, last_break


# ------------------------------------------------------------------
# Minimum bars needed per computation
_MIN_BARS = 250

# Annualization factors: bars per calendar year for 24-hour FX markets.
# Used to scale per-bar realized volatility to an annual figure so that
# agents can reason in annual-vol terms (e.g. "EUR/USD ~ 8% pa").
_BARS_PER_YEAR: dict = {
    "1m":  362_880,   # 252 * 1440
    "5m":  72_576,    # 252 * 288
    "15m": 24_192,    # 252 * 96
    "30m": 12_096,    # 252 * 48
    "1H":  6_048,     # 252 * 24
    "4H":  1_512,     # 252 * 6
    "1D":  252,
}


class DataLoader:
    """
    Loads OHLCV bars from MT5 (or simulation) and computes TechnicalFeatures.
    """

    def __init__(self, simulation: bool = False, executor=None, timeframe_map: Dict[str, str] = None):
        self._executor = executor
        # Per-profile tier → MT5 timeframe override (e.g. scalp: short="1m")
        # Format: {"long": "1H", "mid": "15m", "short": "1m"}
        self._tier_tf_map: Dict[str, str] = timeframe_map or {"long": "1D", "mid": "1H", "short": "15m"}
        # Resolve simulation mode: explicit flag, executor says so, or no broker available
        if simulation:
            self.simulation = True
        elif executor is not None:
            self.simulation = executor.simulation_mode
        else:
            # No executor: fall back to simulation (no direct MT5 import needed)
            self.simulation = True

        # Build timeframe map from executor (works for both pymt5linux and official MT5)
        self._tf_map: Dict[str, Any] = {}
        if not self.simulation and executor is not None:
            self._tf_map = executor.get_tf_map()

        self.logger = logging.getLogger("DataLoader")
        if self.simulation:
            self.logger.warning("DataLoader: running in simulation mode with synthetic data.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_features(self, symbol: str, primary_tf: str = "1H",
                     n_bars: int = 300) -> Optional[TechnicalFeatures]:
        """
        Load bars for *symbol* on *primary_tf* and compute all TechnicalFeatures.
        Breadth features use the D1 timeframe.
        """
        bars = self._load_bars(symbol, primary_tf, n_bars)
        if bars is None:
            return None

        close = bars["close"]
        high = bars["high"]
        low = bars["low"]
        volume = bars["volume"]

        if len(close) < _MIN_BARS:
            # For longer timeframes (D1, 4H) the nominal 250-bar minimum cannot be
            # met in a short backtest — apply a per-TF minimum instead.
            _tf_min_required = {
                "1D": 25,   # EMA20 needs 20 bars; 25 gives modest MACD window too
                "1W": 15,
                "4H": 50,   # EMA50 needs 50 bars
            }.get(primary_tf, _MIN_BARS)
            if len(close) < _tf_min_required:
                self.logger.error(
                    f"Not enough bars for {symbol}/{primary_tf}: "
                    f"got {len(close)}, need {_tf_min_required}"
                )
                return None
            # For D1/4H with sufficient but sub-250 bars: proceed with clamped periods

        # --- Moving averages ------------------------------------------------
        # Clamp EMA periods to available data so that longer-timeframe arrays
        # (D1, 4H) don't fail when they have fewer bars than the nominal period.
        # This is intentional: in a 5-day backtest you only have ~47 D1 bars;
        # EMA47 gives a valid long-term trend signal on that scale.
        _n = len(close)
        _p200 = min(200, _n - 1)
        _p50  = min(50, _n - 1)
        _p20  = min(20, _n - 1)
        ema20_arr = _ema(close, _p20)
        ema50_arr = _ema(close, _p50)
        ema200_arr = _ema(close, _p200)
        ema20 = float(ema20_arr[-1])
        ema50 = float(ema50_arr[-1])
        ema200 = float(ema200_arr[-1])
        _slope_lb = min(6, len(ema20_arr) - 1)  # clamp lookback for slope
        ema20_slope = float(ema20_arr[-1] - ema20_arr[-(_slope_lb)]) / max(ema20_arr[-(_slope_lb)], 1e-9)
        ema50_slope = float(ema50_arr[-1] - ema50_arr[-(_slope_lb)]) / max(ema50_arr[-(_slope_lb)], 1e-9)
        ema200_slope = float(ema200_arr[-1] - ema200_arr[-(_slope_lb)]) / max(ema200_arr[-(_slope_lb)], 1e-9)

        # --- RSI -------------------------------------------------------------
        rsi_14 = _rsi(close, 14)
        rsi_4 = _rsi(close, 4)

        # --- MACD ------------------------------------------------------------
        _, _, macd_hist = _macd(close)
        _, _, macd_hist_prev = _macd(close[:-1])
        macd_hist_delta = float(macd_hist - macd_hist_prev)

        # --- ROC -------------------------------------------------------------
        roc_10 = float((close[-1] - close[-11]) / max(close[-11], 1e-9))

        # --- ATR -------------------------------------------------------------
        atr_arr = _atr(high, low, close, 14)
        atr_14 = float(atr_arr[-1])
        atr_arr5 = _atr(high, low, close, 5)
        atr_5 = float(atr_arr5[-1])

        # --- Bollinger Bands -------------------------------------------------
        sma20 = float(close[-20:].mean())
        std20 = float(close[-20:].std())
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_width = float((bb_upper - bb_lower) / max(sma20, 1e-9))
        bb_percent_b = float((close[-1] - bb_lower) / max(bb_upper - bb_lower, 1e-9))

        # --- Keltner Channel width -------------------------------------------
        keltner_width = float(atr_14 * 2 / max(ema20, 1e-9))

        # --- Realized vol (20-bar, annualised using timeframe-aware factor) ---
        # sqrt(bars_per_year) correctly scales per-bar vol to an annual figure
        # so that e.g. EUR/USD on 1m or 1H both produce ~7-10% annual vol.
        log_rets = np.diff(np.log(close[-21:]))
        _bpy = _BARS_PER_YEAR.get(primary_tf, 252)
        realized_vol = float(log_rets.std() * np.sqrt(_bpy))

        # --- ADX -------------------------------------------------------------
        adx_14 = self._adx(high, low, close, 14)

        # --- RVI (Relative Vigor Index, simplified) ---------------------------
        rvi = self._rvi(close, high, low, 10)

        # --- Hurst -----------------------------------------------------------
        hurst_exponent = _hurst(close)

        # --- ATR/price ratio & VWAP distance ---------------------------------
        atr_price_ratio = float(atr_14 / max(close[-1], 1e-9))

        # Intraday session-reset VWAP:
        # Use bars from 00:00 UTC of the current trading day when timestamps are
        # available (live mode).  Fall back to a rolling session window for
        # simulation / when bar timestamps are missing.  This matches the VWAP
        # anchor used by institutional algorithms and gives proper mean-reversion
        # and breakout signals for intraday scalping.
        bar_times = bars.get("time")   # seconds-since-epoch, may be None in sim
        if bar_times is not None and len(bar_times) > 1:
            import datetime as _dt
            today_utc = _dt.datetime.utcfromtimestamp(int(bar_times[-1])).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            session_start_ts = int(today_utc.timestamp())
            # Find the first bar at or after midnight UTC
            day_mask = bar_times >= session_start_ts
            if day_mask.sum() > 0:
                _c = close[day_mask]
                _v = volume[day_mask]
            else:
                # All bars are from previous day — use full window as fallback
                _c = close[-60:]
                _v = volume[-60:]
        else:
            # Simulation or no timestamps: use last ~6.5h (390 × 1m bars = one session)
            _c = close[-390:]
            _v = volume[-390:]
        vwap = float((_c * _v).sum() / max(_v.sum(), 1e-9))
        vwap_distance = float((close[-1] - vwap) / max(atr_14, 1e-9))

        # --- Structure -------------------------------------------------------
        sh, sl, hh_hl, lh_ll, last_break = _detect_swings(high, low)

        # --- Breadth (index level, D1) ----------------------------------------
        index_trend, adec, above50, above200, sector_dir = self._breadth_features(symbol)

        # --- Macro / intermarket (yfinance, cached) ---------------------------
        macro = self._macro_features()
        dxy_dir, vix_dir, crude_dir, yield_dir = self._intermarket_scores(symbol, macro)

        # --- Session breakout (Asian range, only meaningful on H1) ------------
        session_break = 0.0
        if primary_tf == "1H":
            session_break = self._session_break_score(high, low, close, bars.get("time"))

        # --- RSI Divergences (mid/1H most informative; computed for all TFs) ----
        bull_div, bear_div = _divergence_score(close, high, low)

        # --- Weekly pivot level proximity (ATR-normalized) --------------------
        nearest_support_atr, nearest_resist_atr = self._pivot_levels(
            symbol, float(close[-1]), atr_14)

        return TechnicalFeatures(
            swing_highs=sh,
            swing_lows=sl,
            hh_hl_count=hh_hl,
            lh_ll_count=lh_ll,
            last_break=last_break,
            ema20=ema20,
            ema50=ema50,
            ema200=ema200,
            ema20_slope=ema20_slope,
            ema50_slope=ema50_slope,
            ema200_slope=ema200_slope,
            rsi_14=rsi_14,
            rsi_4=rsi_4,
            macd_hist=float(macd_hist),
            macd_hist_delta=macd_hist_delta,
            roc_10=roc_10,
            bb_percent_b=bb_percent_b,
            atr_14=atr_14,
            atr_5=atr_5,
            realized_vol=realized_vol,
            bb_width=bb_width,
            keltner_width=keltner_width,
            adx_14=adx_14,
            rvi=rvi,
            hurst_exponent=hurst_exponent,
            atr_price_ratio=atr_price_ratio,
            vwap_distance=vwap_distance,
            index_trend=index_trend,
            advance_decline=adec,
            above_50ema_pct=above50,
            above_200ema_pct=above200,
            sector_direction=sector_dir,
            dxy_dir=dxy_dir,
            vix_dir=vix_dir,
            crude_dir=crude_dir,
            yield_dir=yield_dir,
            session_break_score=session_break,
            bull_div_score=bull_div,
            bear_div_score=bear_div,
            nearest_support_atr=nearest_support_atr,
            nearest_resist_atr=nearest_resist_atr,
            bars_per_year=float(_BARS_PER_YEAR.get(primary_tf, 252)),
        )

    def get_pivot_prices(self, symbol: str, current_price: float, atr_14: float
                         ) -> Dict[str, float]:
        """
        Return the absolute weekly pivot prices (PP, R1, R2, S1, S2) plus the
        nearest_support and nearest_resistance prices (not ATR-normalised).

        Falls back to ±2×ATR pseudo-levels when data is unavailable (simulation
        or insufficient history) so callers don't have to special-case None.
        """
        if self.simulation:
            return {
                "pp": current_price,
                "r1": current_price + 2 * atr_14,
                "r2": current_price + 4 * atr_14,
                "s1": current_price - 2 * atr_14,
                "s2": current_price - 4 * atr_14,
                "nearest_support": current_price - 2 * atr_14,
                "nearest_resist":  current_price + 2 * atr_14,
            }

        bars = self._load_bars(symbol, "1D", 12)
        if bars is None or len(bars["close"]) < 7:
            return {
                "pp": current_price,
                "r1": current_price + 2 * atr_14,
                "r2": current_price + 4 * atr_14,
                "s1": current_price - 2 * atr_14,
                "s2": current_price - 4 * atr_14,
                "nearest_support": current_price - 2 * atr_14,
                "nearest_resist":  current_price + 2 * atr_14,
            }

        week_high  = float(bars["high"][-6:-1].max())
        week_low   = float(bars["low"][-6:-1].min())
        week_close = float(bars["close"][-2])

        pp = (week_high + week_low + week_close) / 3.0
        r1 = 2.0 * pp - week_low
        r2 = pp + (week_high - week_low)
        s1 = 2.0 * pp - week_high
        s2 = pp - (week_high - week_low)

        resist_above  = [l for l in (pp, r1, r2) if l > current_price]
        support_below = [l for l in (pp, s1, s2) if l < current_price]

        nearest_resist  = min(resist_above)  if resist_above  else current_price + 2 * atr_14
        nearest_support = max(support_below) if support_below else current_price - 2 * atr_14

        return {
            "pp": pp, "r1": r1, "r2": r2, "s1": s1, "s2": s2,
            "nearest_support": nearest_support,
            "nearest_resist":  nearest_resist,
        }

    def get_structural_levels(self, symbol: str, current_price: float, atr_14: float
                              ) -> Dict[str, float]:
        """
        Combined structural levels: weekly formula pivots PLUS D1 swing highs/lows.

        This is the preferred method for SL/TP placement.  It merges two sources:

          1. Weekly PP/R1/R2/S1/S2  — formula-derived, always present
          2. D1 swing nodes          — actual price memory from ~3 months of daily
                                       bars; lookback=10 bars for major swings only

        The merged set is de-duplicated (levels within 0.3×ATR are collapsed),
        then sorted so that 'r1' is always the nearest resistance above the current
        price (whether that came from a pivot or a D1 swing high), and 's1' is the
        nearest support below.

        Returns the same key set as get_pivot_prices() so all callers are
        interchangeable:
          pp, r1, r2, s1, s2, nearest_support, nearest_resist
        """
        atr = max(atr_14, 1e-9)
        fallback = {
            "pp": current_price,
            "r1": current_price + 2 * atr,
            "r2": current_price + 4 * atr,
            "s1": current_price - 2 * atr,
            "s2": current_price - 4 * atr,
            "nearest_support": current_price - 2 * atr,
            "nearest_resist":  current_price + 2 * atr,
        }

        if self.simulation:
            return fallback

        # ── 1. Weekly formula pivots ──────────────────────────────────────
        pivot_dict = self.get_pivot_prices(symbol, current_price, atr_14)
        pp = pivot_dict["pp"]
        piv_r1, piv_r2 = pivot_dict["r1"], pivot_dict["r2"]
        piv_s1, piv_s2 = pivot_dict["s1"], pivot_dict["s2"]

        # ── 2. D1 swing highs / lows (last ~3 months, lookback=10 bars) ──
        # lookback=10 means a bar must be the highest/lowest of a 21-bar window
        # to qualify — this filters out minor 1-2 day wiggles and keeps only
        # the major structural turning points visible on the daily chart.
        d1_bars = self._load_bars(symbol, "1D", 60)
        d1_swing_highs: list = []
        d1_swing_lows:  list = []
        if d1_bars is not None and len(d1_bars["close"]) >= 25:
            sh, sl, _, _, _ = _detect_swings(d1_bars["high"], d1_bars["low"], lookback=10)
            d1_swing_highs = [float(h) for h in sh]
            d1_swing_lows  = [float(l) for l in sl]

        # ── 3. Merge into sorted support / resistance lists ───────────────
        raw_supports = sorted(
            set(
                [l for l in (pp, piv_s1, piv_s2) if l < current_price]
                + [l for l in d1_swing_lows if l < current_price]
            ),
            reverse=True,  # highest first = nearest below price
        )
        raw_resists = sorted(
            set(
                [l for l in (pp, piv_r1, piv_r2) if l > current_price]
                + [l for l in d1_swing_highs if l > current_price]
            )
            # lowest first = nearest above price
        )

        # ── 4. De-duplicate: collapse levels within 0.3×ATR of each other ─
        # When a pivot and a D1 swing are within the same cluster, keep the
        # one that is closest to the current price (already guaranteed by the
        # sort order) and skip the rest.
        def _dedup(levels: list) -> list:
            out: list = []
            for lvl in levels:
                if not out or abs(lvl - out[-1]) > 0.3 * atr:
                    out.append(lvl)
            return out

        supports = _dedup(raw_supports)
        resists  = _dedup(raw_resists)

        # ── 5. Build output ───────────────────────────────────────────────
        nearest_support = supports[0] if supports else current_price - 2 * atr
        nearest_resist  = resists[0]  if resists  else current_price + 2 * atr

        out_r1 = resists[0] if len(resists) >= 1 else piv_r1
        out_r2 = resists[1] if len(resists) >= 2 else piv_r2
        out_s1 = supports[0] if len(supports) >= 1 else piv_s1
        out_s2 = supports[1] if len(supports) >= 2 else piv_s2

        return {
            "pp": pp,
            "r1": out_r1, "r2": out_r2,
            "s1": out_s1, "s2": out_s2,
            "nearest_support": nearest_support,
            "nearest_resist":  nearest_resist,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def get_multi_features(self, symbol: str) -> Dict[str, Optional["TechnicalFeatures"]]:
        """
        Fetch TechnicalFeatures for all three timeframe tiers in parallel (threads).

        The tier→timeframe mapping comes from `self._tier_tf_map`, which is set
        at construction from the profile's `timeframes` config.  Defaults:
          long  → 1D   mid → 1H   short → 15m
        Scalp profile overrides to:
          long  → 1H   mid → 15m  short → 1m

        Returns dict with keys "long", "mid", "short".
        """
        import concurrent.futures
        tf_map = self._tier_tf_map

        def _fetch(tier_tf):
            tier, tf = tier_tf
            return tier, self.get_features(symbol, tf, n_bars=300)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            results = dict(pool.map(_fetch, tf_map.items()))

        return results

    def _load_bars(self, symbol: str, timeframe: str, n: int) -> Optional[Dict[str, np.ndarray]]:
        if self.simulation:
            return self._synthetic_bars(n, timeframe)

        tf = self._tf_map.get(timeframe)
        if tf is None:
            self.logger.error(f"Unknown timeframe: {timeframe}")
            return None

        rates = self._executor.copy_rates(symbol, timeframe, n)
        if rates is None or len(rates) == 0:
            self.logger.error(f"No data from MT5 for {symbol}/{timeframe}")
            return None

        return {
            "open":   np.array([r["open"]        for r in rates], dtype=float),
            "high":   np.array([r["high"]        for r in rates], dtype=float),
            "low":    np.array([r["low"]         for r in rates], dtype=float),
            "close":  np.array([r["close"]       for r in rates], dtype=float),
            "volume": np.array([r["tick_volume"] for r in rates], dtype=float),
            "time":   np.array([r["time"]        for r in rates], dtype=np.int64),
        }

    def _synthetic_bars(self, n: int, timeframe: str = "1H") -> Dict[str, np.ndarray]:
        """Generate plausible random walk bars for simulation."""
        # Different seed per timeframe so each tier has distinct price action
        # Keep 1H = 42 (original seed) for backward compatibility
        _tf_seeds = {"1W": 40, "1D": 41, "4H": 43, "1H": 42, "15m": 44, "5m": 45, "1m": 46}
        np.random.seed(_tf_seeds.get(timeframe, 42))
        log_returns = np.random.normal(0.0002, 0.01, n)
        close = 100.0 * np.exp(np.cumsum(log_returns))
        noise = np.abs(np.random.normal(0, 0.005, n)) * close
        high = close + noise
        low = close - noise
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        volume = np.random.randint(1000, 10000, n).astype(float)
        _tf_seconds = {"1W": 604800, "1D": 86400, "4H": 14400, "1H": 3600,
                       "15m": 900, "5m": 300, "1m": 60}
        bar_secs = _tf_seconds.get(timeframe, 3600)
        now_ts = int(_time.time())
        timestamps = np.array([now_ts - (n - 1 - i) * bar_secs for i in range(n)], dtype=np.int64)
        return {"open": open_, "high": high, "low": low, "close": close,
                "volume": volume, "time": timestamps}

    def _adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        """Compute ADX for latest bar."""
        up_move = np.diff(high)
        down_move = -np.diff(low)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        atr = _atr(high, low, close, period)
        if len(atr) < period:
            return 25.0
        smooth_plus = _ema(plus_dm[-len(atr):], period)
        smooth_minus = _ema(minus_dm[-len(atr):], period)
        atr_s = atr
        di_plus = 100 * smooth_plus / np.maximum(atr_s, 1e-9)
        di_minus = 100 * smooth_minus / np.maximum(atr_s, 1e-9)
        dx = 100 * np.abs(di_plus - di_minus) / np.maximum(di_plus + di_minus, 1e-9)
        adx_arr = _ema(dx, period)
        return float(np.clip(adx_arr[-1], 0, 100))

    def _rvi(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int) -> float:
        """Simplified RVI: ratio of (close-open) range over (high-low) range, smoothed."""
        numerator = np.diff(close[-period - 1:])
        hl_range = high[-period:] - low[-period:]
        rvi_raw = numerator / np.maximum(hl_range, 1e-9)
        return float(np.clip(rvi_raw.mean(), -1.0, 1.0))

    # ── Intermarket correlation weights per symbol ────────────────────────
    # Tuple: (dxy_w, vix_w, crude_w, yield_w)
    # Positive weight = that macro indicator bullish = bullish for this pair.
    # Sign convention:
    #   dxy_trend positive  → USD gaining strength
    #   vix_signal positive → fear/risk-off dominant
    #   crude_trend positive → WTI oil rising
    #   yield_trend positive → US 10Y yield rising
    _INTERMARKET_WEIGHTS: Dict[str, Tuple[float, float, float, float]] = {
        # ── Major USD pairs ──────────────────────────────────────────────────
        "EURUSD": (-0.75, -0.15,  0.00, -0.40),
        "GBPUSD": (-0.65, -0.15,  0.00, -0.30),
        "USDJPY": (+0.55, -0.55,  0.00, +0.65),
        "USDCHF": (+0.55, -0.35,  0.00, +0.40),
        "AUDUSD": (-0.50, -0.55, +0.20, -0.20),
        "USDCAD": (+0.40, +0.45, -0.70, +0.20),
        "NZDUSD": (-0.45, -0.50,  0.00, -0.15),
        # ── Metals ───────────────────────────────────────────────────────────
        "XAUUSD": (-0.65, +0.65,  0.00, -0.55),
        # Silver: mirrors gold but with more industrial/crude correlation
        "XAGUSD": (-0.60, +0.55, +0.15, -0.45),
        # ── US equity indices ────────────────────────────────────────────────
        "US30":   (-0.20, -0.80, +0.10, -0.25),
        "US500":  (-0.20, -0.80, +0.10, -0.25),
        "USTEC":  (-0.25, -0.75,  0.00, -0.40),
        # ── European equity indices ──────────────────────────────────────────
        "DAX":    (+0.15, -0.70,  0.00, -0.20),
        "UK100":  (+0.15, -0.65, +0.25, -0.20),
        # ── Asia-Pacific index ───────────────────────────────────────────────
        # ASX 200: commodity-heavy, risk-on, AUD-correlated
        "AUS200": (-0.15, -0.75, +0.30, -0.20),
        # Nikkei 225: export-driven, USDJPY-correlated (JPY weak = Nikkei bull)
        "JP225":  (+0.45, -0.65,  0.00, +0.25),
        # ── Cross currency pairs (no direct DXY link, mainly risk-on/off) ───
        # EURJPY: DXY has mild effect via EUR; VIX dominant (risk-on cross)
        "EURJPY": (-0.15, -0.65, +0.10, +0.25),
        # GBPJPY: higher-beta risk-on cross, VIX most important
        "GBPJPY": (-0.10, -0.70, +0.10, +0.20),
        # AUDJPY: purest risk-on/off barometer in FX, crude adds commodity angle
        "AUDJPY": (-0.20, -0.80, +0.25, +0.10),
        # EURGBP: relative EU/UK macro, modest macro driver
        "EURGBP": (-0.20, -0.10,  0.00, -0.15),
        # EURCHF: EUR vs safe-haven CHF; flight-to-safety = bearish
        "EURCHF": (-0.10, -0.50,  0.05, +0.10),
        # CHFJPY: safe-haven pair (CHF/JPY); fear = CHFJPY falls (both safe havens,
        #   but CHF is stronger haven so pair falls in risk-off)
        "CHFJPY": (+0.10, +0.40,  0.05, +0.15),
        # ── Energy ───────────────────────────────────────────────────────────
        # WTI Crude Oil (MetaQuotes-Demo symbol: USOIL or XTIUSD)
        "USOIL":  (-0.40, -0.30, +1.00, +0.20),
        "XTIUSD": (-0.40, -0.30, +1.00, +0.20),
    }

    # Class-level macro data cache (shared across all DataLoader instances)
    _macro_cache: Dict[str, float] = {}
    _macro_cache_ts: float = 0.0
    _MACRO_CACHE_TTL: float = 300.0   # 5 minutes

    # Class-level real-breadth cache — yfinance data, refreshed every hour
    _breadth_cache: Dict[str, float] = {}
    _breadth_cache_ts: float = 0.0
    _BREADTH_CACHE_TTL: float = 3600.0   # 1 hour (daily data changes intraday)

    # Forex pairs where USD is the quote currency (e.g. EURUSD = EUR/USD).
    # A BUY on these = long foreign currency / short USD.
    _USD_QUOTE_PAIRS = frozenset([
        "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "XAUUSD",
        "CHFUSD", "CADUSD", "XAGUSD",
    ])
    # Forex pairs where USD is the base currency (e.g. USDJPY = USD/JPY).
    _USD_BASE_PAIRS = frozenset([
        "USDJPY", "USDCHF", "USDCAD", "USDSGD", "USDHKD",
        "USDMXN", "USDNOK", "USDSEK",
    ])
    # Cross pairs — no USD leg.  Mapped to an MT5 proxy for breadth context.
    # The proxy should reflect the risk-on/off bias of the pair.
    _CROSS_BREADTH: Dict[str, List[str]] = {
        # JPY crosses: prefer indices available on MetaQuotes-Demo; JP225 as last resort
        "EURJPY": ["DAX",   "US500", "SP500", "JP225"],
        "GBPJPY": ["UK100", "US500", "SP500", "JP225"],
        "AUDJPY": ["US500", "SP500", "AUS200", "JP225"],
        "NZDJPY": ["US500", "SP500", "AUS200"],
        "CADJPY": ["US500", "SP500"],
        "CHFJPY": ["US500", "SP500", "JP225"],
        # EUR crosses
        "EURGBP": ["DAX", "UK100", "US500", "SP500"],
        "EURCHF": ["DAX", "US500", "SP500"],
        # Commodity / energy
        "USOIL":  ["US500", "SP500"],
        "XTIUSD": ["US500", "SP500"],
    }

    def _breadth_features(self, symbol: str):
        """
        Compute breadth features using a macro proxy appropriate for the symbol type.

        When yfinance is available and not in simulation mode, this method first
        tries ``_breadth_features_real()`` which uses real NYSE breadth data
        (``^NYADD``, SPY, sector ETFs) with a 1-hour cache.  On failure it falls
        back to the MT5-proxy approach below.

        Forex USD pairs  → USDJPY D1 as USD-strength proxy (USDJPY up = USD strong).
          USD-quote (EURUSD etc.): USD strength opposes buying → proxy used as-is.
          USD-base  (USDJPY etc.): USD strength favours buying → same direction.
        European indices → DAX or UK100 as regional breadth.
        Other indices / commodities → SP500 / US500.
        """
        if self.simulation:
            return 0.1, 0.52, 0.60, 0.45, 0.1

        # ── Try real NYSE/SPY breadth first (yfinance, 1-hour cache) ─────────
        try:
            real = self._breadth_features_real(symbol)
            if real is not None:
                return real
        except Exception:
            pass  # fall through to MT5-proxy method

        sym_up = symbol.upper()

        # ── Forex USD pairs: use USDJPY as USD-strength proxy ─────────────────
        is_usd_quote = sym_up in self._USD_QUOTE_PAIRS
        is_usd_base  = sym_up in self._USD_BASE_PAIRS
        if is_usd_quote or is_usd_base:
            usd_bars = None
            for proxy in ["USDJPY", "USDJPYm"]:
                usd_bars = self._load_bars(proxy, "1D", 60)
                if usd_bars is not None:
                    break
            if usd_bars is None:
                # Fallback: try EURUSD inverted
                eu_bars = self._load_bars("EURUSD", "1D", 60)
                if eu_bars is not None:
                    # Invert: strong USD = EURUSD falling
                    usd_bars = {k: 1.0 / np.maximum(eu_bars[k], 1e-9)
                                for k in ("open", "high", "low", "close", "volume")}
                    usd_bars["volume"] = eu_bars["volume"]
            if usd_bars is None:
                return 0.0, 0.0, 0.0, 0.0, 0.0

            close_usd = usd_bars["close"]
            ema50_u   = _ema(close_usd, 50)
            ema200_u  = _ema(close_usd, 200)

            # USD trending up = USDJPY EMA50 > EMA200
            usd_trend     = float(np.sign(ema50_u[-1] - ema200_u[-1]) * 0.5)
            up_days       = float(np.sum(np.diff(close_usd[-21:]) > 0) / 20)
            adec          = up_days - 0.5
            above50_u     = float(np.clip((close_usd[-1] - ema50_u[-1])  / max(ema50_u[-1],  1e-9), -1, 1))
            above200_u    = float(np.clip((close_usd[-1] - ema200_u[-1]) / max(ema200_u[-1], 1e-9), -1, 1))

            # For USD-quote pairs (EURUSD, GBPUSD …): USD strong = bearish for pair.
            # The BreadthAgent will interpret a positive index_trend as "bullish context."
            # We want positive to mean "USD context supports this direction," so:
            #   USD-quote: index_trend = usd_trend (USD strong = sell pair = negative dir bias)
            #   USD-base:  index_trend = usd_trend (USD strong = buy pair = positive dir bias)
            # Both cases: leave sign as-is — BreadthAgent receives raw USD-strength signal.
            return usd_trend, adec, above50_u, above200_u, usd_trend

        # ── Cross currency pairs (no USD leg) ─────────────────────────────────
        # Use CROSS_BREADTH proxies defined in class dict (risk-on/off context)
        if sym_up in self._CROSS_BREADTH:
            proxies = self._CROSS_BREADTH[sym_up]
            bars = None
            for proxy in proxies:
                bars = self._load_bars(proxy, "1D", 60)
                if bars is not None:
                    break
            if bars is None:
                return 0.0, 0.5, 0.5, 0.5, 0.0
            close = bars["close"]
            ema50_c  = _ema(close, 50)
            ema200_c = _ema(close, 200)
            cross_trend  = float(np.sign(ema50_c[-1] - ema200_c[-1]) * 0.5)
            up_days      = float(np.sum(np.diff(close[-21:]) > 0) / 20)
            adec         = up_days - 0.5
            above50_c    = float(np.clip((close[-1] - ema50_c[-1])  / max(ema50_c[-1],  1e-9), -1, 1))
            above200_c   = float(np.clip((close[-1] - ema200_c[-1]) / max(ema200_c[-1], 1e-9), -1, 1))
            return cross_trend, adec, above50_c, above200_c, cross_trend

        # ── Equity index / commodity proxies ──────────────────────────────────
        if any(x in sym_up for x in ["DAX", "GER", "CAC"]):
            proxies = ["DAX", "US500", "SP500"]
        elif any(x in sym_up for x in ["UK", "FTSE"]):
            proxies = ["UK100", "US500", "SP500"]
        elif any(x in sym_up for x in ["AUS", "AUS200"]):
            proxies = ["AUS200", "US500", "SP500"]
        elif any(x in sym_up for x in ["JP225", "NIK", "JPN"]):
            proxies = ["JP225", "US500", "SP500"]
        elif any(x in sym_up for x in ["OIL", "XTI", "WTI", "BRENT", "NGAS"]):
            proxies = ["US500", "SP500"]
        else:
            proxies = ["SP500", "US500"]

        bars = None
        for proxy in proxies:
            bars = self._load_bars(proxy, "1D", 60)
            if bars is not None:
                break
        if bars is None:
            return 0.0, 0.5, 0.5, 0.5, 0.0

        close = bars["close"]
        ema50_idx = _ema(close, 50)
        ema200_idx = _ema(close, 200)

        index_trend = float(np.sign(ema50_idx[-1] - ema200_idx[-1]) * 0.5)
        # Approximate A/D as fraction of bars up over last 20
        up_days = float(np.sum(np.diff(close[-21:]) > 0) / 20)
        adec = up_days - 0.5  # centre around 0

        above50 = float(np.clip((close[-1] - ema50_idx[-1]) / max(ema50_idx[-1], 1e-9), -1, 1))
        above200 = float(np.clip((close[-1] - ema200_idx[-1]) / max(ema200_idx[-1], 1e-9), -1, 1))
        sector_dir = index_trend  # simplified: same as index

        return index_trend, adec, above50, above200, sector_dir

    # ------------------------------------------------------------------
    # Real breadth features via yfinance (NYSE A/D + SPY + sector ETFs)
    # ------------------------------------------------------------------

    def _breadth_features_real(self, symbol: str):
        """
        Fetch genuine market breadth from yfinance with a 1-hour TTL cache.

        Tickers used:
          ^NYADD  — NYSE net advance/decline (raw breadth)
          SPY     — S&P 500 ETF (trend + % above EMA)
          XLF, XLK, XLE, XLV, XLI — sector ETFs (risk-on direction)

        Returns the same 5-tuple as ``_breadth_features``:
          (index_trend, advance_decline, above_50ema, above_200ema, sector_dir)
        all values in [-1, 1], or None if yfinance is unavailable.

        All symbols receive the same SPY-centric breadth because:
          • For USD-quote FX pairs (EURUSD etc.) risk-on = USD weak = bullish
          • For equity indices the S&P 500 breadth directly applies
          • Cross pairs (EURJPY etc.) react most strongly to risk-on/off

        The BreadthAgent weights this signal appropriately per symbol via
        its internal direction bias, so a single risk-on reading is correct
        to pass for all symbols.
        """
        now = _time.time()
        if (DataLoader._breadth_cache
                and (now - DataLoader._breadth_cache_ts) < DataLoader._BREADTH_CACHE_TTL):
            cached = DataLoader._breadth_cache
            # Return cached tuple
            return (
                cached["index_trend"],
                cached["advance_decline"],
                cached["above_50ema"],
                cached["above_200ema"],
                cached["sector_dir"],
            )

        try:
            import yfinance as yf

            # Download last 60 trading days of daily data.
            # ^NYADD was removed from Yahoo Finance; use RSP (equal-weight S&P 500)
            # vs SPY spread as a breadth proxy: RSP outperforming SPY = broad advance.
            tickers = ["SPY", "RSP", "XLF", "XLK", "XLE", "XLV", "XLI"]
            data = yf.download(
                tickers, period="65d", interval="1d",
                progress=False, auto_adjust=True, threads=True,
            )
            closes = data.get("Close", data)

            # ── SPY trend: EMA50 > EMA200 ────────────────────────────────────
            spy_col = closes.get("SPY")
            if spy_col is None:
                return None
            spy_vals = spy_col.dropna().values
            if len(spy_vals) < 51:
                return None

            ema50  = _ema(spy_vals, 50)
            ema200 = _ema(spy_vals, min(200, len(spy_vals) - 1))

            index_trend = float(np.sign(ema50[-1] - ema200[-1]) * 0.5)
            above_50ema = float(np.clip(
                (spy_vals[-1] - ema50[-1]) / max(ema50[-1], 1e-9), -1.0, 1.0
            ))
            above_200ema = float(np.clip(
                (spy_vals[-1] - ema200[-1]) / max(ema200[-1], 1e-9), -1.0, 1.0
            ))

            # ── Breadth: RSP/SPY ratio momentum (equal-weight vs cap-weight) -
            # RSP outperforming SPY → broad advance (many stocks rising).
            # RSP underperforming → narrow leadership (few mega-caps carrying).
            advance_decline = 0.0
            rsp_col = closes.get("RSP")
            if rsp_col is not None:
                rsp_vals = rsp_col.dropna().values
                if len(rsp_vals) >= 6 and len(spy_vals) >= 6:
                    # 5-day relative performance, scaled: ±5% → ±1
                    rsp_ret = (rsp_vals[-1] - rsp_vals[-6]) / max(abs(rsp_vals[-6]), 1e-9)
                    spy_ret = (spy_vals[-1] - spy_vals[-6]) / max(abs(spy_vals[-6]), 1e-9)
                    advance_decline = float(np.clip((rsp_ret - spy_ret) / 0.05, -1.0, 1.0))

            # ── Sector direction: average 5-day return of sector ETFs ────────
            sector_rets: list = []
            for etf in ("XLF", "XLK", "XLE", "XLV", "XLI"):
                col = closes.get(etf)
                if col is None:
                    continue
                vals = col.dropna().values
                if len(vals) >= 6:
                    ret = (vals[-1] - vals[-6]) / max(abs(vals[-6]), 1e-9)
                    sector_rets.append(float(np.clip(ret / 0.05, -1.0, 1.0)))
            sector_dir = float(np.mean(sector_rets)) if sector_rets else 0.0

            result = {
                "index_trend":    index_trend,
                "advance_decline": advance_decline,
                "above_50ema":   above_50ema,
                "above_200ema":  above_200ema,
                "sector_dir":    sector_dir,
            }
            DataLoader._breadth_cache    = result
            DataLoader._breadth_cache_ts = now

            return index_trend, advance_decline, above_50ema, above_200ema, sector_dir

        except Exception as exc:
            self.logger.debug(f"_breadth_features_real: yfinance failed ({exc})")
            return None

    # ------------------------------------------------------------------
    # Weekly pivot level proximity
    # ------------------------------------------------------------------

    def _pivot_levels(
        self, symbol: str, current_price: float, atr_14: float
    ) -> tuple:
        """
        Classic weekly pivot points (PP, R1/R2, S1/S2) from the last 5 D1 bars.
        Returns (nearest_support_atr, nearest_resist_atr) — ATR-normalised distances.

        10.0 means "far from any pivot level" (safe to enter).
        < 0.5 means "price is very close to a key level" (risky entry).
        """
        if self.simulation:
            return 5.0, 5.0   # safe default in simulation

        bars = self._load_bars(symbol, "1D", 12)
        if bars is None or len(bars["close"]) < 7:
            return 5.0, 5.0

        # Use the last 5 closed D1 bars as a proxy for the previous week.
        # Exclude the last/current bar which may be incomplete.
        week_high  = float(bars["high"][-6:-1].max())
        week_low   = float(bars["low"][-6:-1].min())
        week_close = float(bars["close"][-2])          # yesterday's close

        pp = (week_high + week_low + week_close) / 3.0
        r1 = 2.0 * pp - week_low
        r2 = pp + (week_high - week_low)
        s1 = 2.0 * pp - week_high
        s2 = pp - (week_high - week_low)

        atr = max(atr_14, 1e-9)

        # Nearest resistance above current price
        resist_above = [l for l in (pp, r1, r2) if l > current_price]
        resist_atr   = (min(resist_above) - current_price) / atr if resist_above else 10.0

        # Nearest support below current price
        support_below = [l for l in (pp, s1, s2) if l < current_price]
        support_atr   = (current_price - max(support_below)) / atr if support_below else 10.0

        return float(np.clip(support_atr, 0.0, 20.0)), float(np.clip(resist_atr, 0.0, 20.0))

    # ------------------------------------------------------------------
    # Macro / intermarket features (yfinance, cached 5 min)
    # ------------------------------------------------------------------

    def _macro_features(self) -> Dict[str, float]:
        """
        Fetch DXY, VIX, WTI crude, US 10Y yield via yfinance.
        Returns dict: {dxy_trend, vix_signal, crude_trend, yield_trend}  values in [-1, 1].
        Cached for 5 minutes (class-level cache shared across all instances).
        """
        now = _time.time()
        if (DataLoader._macro_cache
                and (now - DataLoader._macro_cache_ts) < DataLoader._MACRO_CACHE_TTL):
            return DataLoader._macro_cache

        result: Dict[str, float] = {
            "dxy_trend": 0.0, "vix_signal": 0.0,
            "crude_trend": 0.0, "yield_trend": 0.0,
        }
        if self.simulation:
            DataLoader._macro_cache = result
            DataLoader._macro_cache_ts = now
            return result

        try:
            import yfinance as yf
            tickers = ["DX-Y.NYB", "^VIX", "CL=F", "^TNX"]
            data = yf.download(
                tickers, period="20d", interval="1d",
                progress=False, auto_adjust=True, threads=True,
            )
            closes = data.get("Close", data)

            def _mom(col: str, n: int = 5) -> float:
                """n-day momentum, clipped to [-1, 1], scaled so ±5% → ±1."""
                if col not in closes.columns:
                    return 0.0
                s = closes[col].dropna().values
                if len(s) < n + 1:
                    return 0.0
                ret = (s[-1] - s[-(n + 1)]) / max(abs(s[-(n + 1)]), 1e-9)
                return float(np.clip(ret / 0.05, -1.0, 1.0))  # 5% = magnitude 1

            result["dxy_trend"]   = _mom("DX-Y.NYB")
            result["crude_trend"] = _mom("CL=F")
            result["yield_trend"] = _mom("^TNX")

            # VIX: combine level (fear above 20) + 5-day direction
            if "^VIX" in closes.columns:
                vix = closes["^VIX"].dropna().values
                if len(vix) >= 2:
                    vix_level  = float(np.clip((vix[-1] - 20.0) / 15.0, -1.0, 1.0))
                    vix_change = float(np.clip((vix[-1] - vix[-min(6, len(vix))]) / max(vix[-1], 1e-9) / 0.20, -1.0, 1.0))
                    result["vix_signal"] = float(np.clip(vix_level * 0.6 + vix_change * 0.4, -1.0, 1.0))

        except Exception as exc:
            self.logger.warning(f"_macro_features: yfinance failed ({exc}), using neutral zeros")

        DataLoader._macro_cache    = result
        DataLoader._macro_cache_ts = now
        return result

    def _intermarket_scores(
        self, symbol: str, macro: Dict[str, float]
    ) -> Tuple[float, float, float, float]:
        """
        Apply per-symbol correlation weights to raw macro signals.
        Returns (dxy_dir, vix_dir, crude_dir, yield_dir) each in [-1, 1].
        """
        sym = symbol.upper().replace("m", "")   # strip mini-contract suffix
        weights = self._INTERMARKET_WEIGHTS.get(sym)
        if weights is None:
            return 0.0, 0.0, 0.0, 0.0
        dw, vw, cw, yw = weights
        return (
            float(np.clip(macro["dxy_trend"]   * dw, -1.0, 1.0)),
            float(np.clip(macro["vix_signal"]  * vw, -1.0, 1.0)),
            float(np.clip(macro["crude_trend"] * cw, -1.0, 1.0)),
            float(np.clip(macro["yield_trend"] * yw, -1.0, 1.0)),
        )

    # ------------------------------------------------------------------
    # Session breakout (Asian range)
    # ------------------------------------------------------------------

    def _session_break_score(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        timestamps: Optional[np.ndarray],
    ) -> float:
        """
        Asian-range breakout signal using H1 bars.

        1. Identify the most-recent complete Asian session (00:00–08:00 UTC)
           using the last 48 bars.
        2. Compute current price position relative to the Asian high/low.
        3. Scale by session multiplier: strongest during London/NY overlap,
           near-zero while the Asian range is still forming.

        Returns a value in [-1, 1]:
          +1 → strong bullish London breakout above Asian high
          -1 → strong bearish London breakdown below Asian low
           0 → price inside range or Asian session forming
        """
        if timestamps is None or len(timestamps) < 10:
            return 0.0

        # Work with the last 48 bars (covers the current + previous Asian sessions)
        n_look = min(48, len(timestamps))
        ts   = timestamps[-n_look:]
        hi   = high[-n_look:]
        lo   = low[-n_look:]
        cl   = close[-n_look:]

        # UTC hours for each bar
        hours = np.array(
            [datetime.fromtimestamp(int(t), tz=timezone.utc).hour for t in ts],
            dtype=int,
        )

        # Asian session mask: bars whose open is in 00:00–07:59 UTC
        asian_mask = hours < 8
        if asian_mask.sum() < 3:
            return 0.0

        asian_high = float(hi[asian_mask].max())
        asian_low  = float(lo[asian_mask].min())
        asian_range = asian_high - asian_low
        if asian_range < 1e-10:
            return 0.0

        current_price = float(cl[-1])
        current_hour  = int(hours[-1])

        # Raw score: distance above/below range as fraction of range width
        if current_price > asian_high:
            raw = min((current_price - asian_high) / asian_range, 1.5)   # cap at 1.5× range
            raw_score = min(raw, 1.0)
        elif current_price < asian_low:
            raw = min((asian_low - current_price) / asian_range, 1.5)
            raw_score = -min(raw, 1.0)
        else:
            # Inside range: weak position signal
            pos = (current_price - asian_low) / asian_range   # 0..1
            raw_score = (pos - 0.5) * 0.25  # ±0.125 max

        # Session weighting: London/NY hours = full signal; Asian formation = minimal
        if 8 <= current_hour < 17:
            mult = 1.0    # London open → NY close
        elif 17 <= current_hour < 21:
            mult = 0.55   # NY afternoon
        elif 21 <= current_hour or current_hour < 1:
            mult = 0.15   # late NY / early Asia — range not set yet
        else:
            mult = 0.25   # Asian session forming: range still being built

        return float(np.clip(raw_score * mult, -1.0, 1.0))
