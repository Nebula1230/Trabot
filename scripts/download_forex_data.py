#!/usr/bin/env python3
"""
Download historical forex data from Dukascopy's public CDN.

Dukascopy stores tick data in compressed binary chunks (one file per hour).
This script downloads them, decodes ticks, and resamples to OHLCV bars at
the requested granularity (default: 15m).

Output: one CSV per symbol in the format the backtest engine expects:
    time,open,high,low,close,volume

Usage:
    python scripts/download_forex_data.py                          # defaults
    python scripts/download_forex_data.py --symbols EURUSD GBPUSD USDJPY
    python scripts/download_forex_data.py --start 2024-01-01 --end 2026-03-31
    python scripts/download_forex_data.py --tf 5m --out ./data/5m

The downloaded data can be used with the backtester:
    python -m tradingagents_v2.backtesting --data-dir ./data/15m ...

Dukascopy CDN is publicly accessible — no API key or account required.
"""

import argparse
import io
import logging
import struct
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import lzma
except ImportError:
    lzma = None

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is required.  pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: 'numpy' is required.  pip install numpy", file=sys.stderr)
    sys.exit(1)


logger = logging.getLogger("DukascopyLoader")

# ─────────────────────────────────────────────────────────────────────────────
# Dukascopy CDN constants
# ─────────────────────────────────────────────────────────────────────────────

# URL pattern: one compressed binary file per hour.
# Months are 0-indexed (January=0, February=1, ..., December=11).
_CDN_URL = (
    "https://datafeed.dukascopy.com/datafeed/{symbol}/"
    "{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
)

# Point value (pip divisor) for decoding raw integer prices.
# Dukascopy stores prices as int32 relative to a base; dividing by _POINT
# gives the real price.  Most FX pairs use 1e5 (5 decimal places).
_POINT: Dict[str, float] = {
    "EURUSD": 1e5, "GBPUSD": 1e5, "AUDUSD": 1e5, "NZDUSD": 1e5,
    "USDCHF": 1e5, "USDCAD": 1e5, "EURGBP": 1e5, "EURCHF": 1e5,
    "EURCAD": 1e5, "EURAUD": 1e5, "GBPJPY": 1e3, "EURJPY": 1e3,
    "USDJPY": 1e3, "AUDJPY": 1e3, "CADJPY": 1e3, "CHFJPY": 1e3,
    "NZDJPY": 1e3, "GBPAUD": 1e5, "GBPCAD": 1e5, "GBPCHF": 1e5,
    "AUDCAD": 1e5, "AUDCHF": 1e5, "AUDNZD": 1e5, "CADCHF": 1e5,
    "NZDCAD": 1e5, "NZDCHF": 1e5,
    "XAUUSD": 1e2, "XAGUSD": 1e3,
}

# Resampling rule string for pandas
_RESAMPLE_RULE: Dict[str, str] = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1H": "1h", "4H": "4h", "1D": "1D",
}


# ─────────────────────────────────────────────────────────────────────────────
# Tick decoding
# ─────────────────────────────────────────────────────────────────────────────

def _decode_bi5(data: bytes, hour_dt: datetime, point: float) -> List[Tuple]:
    """Decode a Dukascopy .bi5 (LZMA-compressed) tick buffer.

    Each tick is 20 bytes:
        4 bytes: milliseconds since the start of the hour (uint32, big-endian)
        4 bytes: ask price as integer (uint32, big-endian)
        4 bytes: bid price as integer (uint32, big-endian)
        4 bytes: ask volume (float32, big-endian)
        4 bytes: bid volume (float32, big-endian)

    Returns list of (unix_timestamp, bid, ask, volume).
    """
    if not data:
        return []

    try:
        raw = lzma.decompress(data)
    except Exception:
        return []

    if len(raw) == 0 or len(raw) % 20 != 0:
        return []

    n_ticks = len(raw) // 20
    ticks = []
    base_ts = hour_dt.timestamp()

    for i in range(n_ticks):
        offset = i * 20
        ms_offset, ask_int, bid_int, ask_vol, bid_vol = struct.unpack(
            ">IIIff", raw[offset : offset + 20]
        )
        ts = base_ts + ms_offset / 1000.0
        bid = bid_int / point
        ask = ask_int / point
        vol = ask_vol + bid_vol
        ticks.append((ts, bid, ask, vol))

    return ticks


def _download_hour(
    symbol: str, dt: datetime, session: requests.Session
) -> List[Tuple]:
    """Download and decode one hour of tick data."""
    # Dukascopy months are 0-indexed
    url = _CDN_URL.format(
        symbol=symbol.upper(),
        year=dt.year,
        month=dt.month - 1,  # 0-indexed!
        day=dt.day,
        hour=dt.hour,
    )
    point = _POINT.get(symbol.upper(), 1e5)

    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 200 and len(resp.content) > 0:
            return _decode_bi5(resp.content, dt, point)
        return []
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Main download + resample
# ─────────────────────────────────────────────────────────────────────────────

def download_symbol(
    symbol: str,
    start: datetime,
    end: datetime,
    tf: str = "15m",
    out_dir: Path = Path("data"),
    max_workers: int = 8,
) -> Optional[Path]:
    """Download tick data from Dukascopy and resample to OHLCV bars.

    Args:
        symbol:      Forex pair (e.g. "EURUSD").
        start:       Start datetime (UTC).
        end:         End datetime (UTC).
        tf:          Target timeframe (e.g. "15m", "1H", "5m").
        out_dir:     Output directory for CSV files.
        max_workers: Parallel download threads.

    Returns:
        Path to the written CSV, or None on failure.
    """
    if lzma is None:
        logger.error("lzma module not available — cannot decompress Dukascopy data")
        return None

    symbol = symbol.upper()
    if symbol not in _POINT:
        logger.warning(f"{symbol} not in known symbols — using default point=1e5")

    logger.info(f"Downloading {symbol} ticks {start:%Y-%m-%d} → {end:%Y-%m-%d}")

    # ── Incremental mode: detect existing data and skip already-covered hours
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_{tf}.csv"
    existing_df = None

    # Map tf string → timedelta for bar-level gap detection
    _TF_DELTA = {
        "1m": timedelta(minutes=1), "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15), "30m": timedelta(minutes=30),
        "1H": timedelta(hours=1), "4H": timedelta(hours=4),
        "1D": timedelta(days=1),
    }
    bar_delta = _TF_DELTA.get(tf, timedelta(hours=1))

    # Work with naive UTC datetimes for reliable set comparisons
    _start = start.replace(second=0, microsecond=0, tzinfo=None)
    _end = end.replace(tzinfo=None)

    # Build the FULL list of expected bar slots from start → end (weekdays only)
    all_slots: List[datetime] = []
    dt = _start
    while dt < _end:
        if dt.weekday() < 5:  # skip weekends
            all_slots.append(dt)
        dt += bar_delta

    # Build the full list of hours (Dukascopy downloads are per-hour)
    all_hours_set: set = set()
    h = _start.replace(minute=0)
    while h < _end:
        if h.weekday() < 5:
            all_hours_set.add(h)
        h += timedelta(hours=1)

    # Determine which bar slots already have data
    existing_slots: set = set()
    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_path, index_col="time")
            existing_df.index = pd.to_datetime(existing_df.index, utc=True)
            if len(existing_df) > 0:
                for ts in existing_df.index:
                    # Build a plain naive datetime for reliable set-lookup
                    existing_slots.add(
                        datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, 0)
                    )
                logger.info(
                    f"  {symbol}_{tf}: {len(existing_df):,} existing bars, "
                    f"{len(existing_slots):,}/{len(all_slots):,} expected slots covered"
                )
        except Exception as exc:
            logger.warning(f"  Could not read existing {out_path} ({exc}) — full re-download")
            existing_df = None

    # Find missing bar slots, then map them back to the hours we need to fetch
    missing_slots = [s for s in all_slots if s not in existing_slots]

    if not missing_slots:
        logger.info(
            f"  {symbol}_{tf}: all {len(all_slots):,} bar slots already covered "
            f"({len(existing_df):,} bars) — nothing to download"
        )
        return out_path

    # Each missing bar slot requires its enclosing hour to be downloaded
    needed_hours: set = set()
    for slot in missing_slots:
        needed_hours.add(slot.replace(minute=0, second=0, microsecond=0))
    # Only keep hours that are valid trading hours
    hours = sorted(h for h in needed_hours if h in all_hours_set)

    logger.info(
        f"  {len(missing_slots):,} missing {tf} bar slots → "
        f"{len(hours):,} hours to fetch "
        f"(out of {len(all_hours_set):,} total hours, "
        f"{len(all_hours_set) - len(hours):,} already covered)"
    )

    # Parallel download
    all_ticks: List[Tuple] = []
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"

    done = 0
    total = len(hours)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_hour, symbol, h, session): h for h in hours}
        for future in as_completed(futures):
            ticks = future.result()
            all_ticks.extend(ticks)
            done += 1
            if done % 200 == 0 or done == total:
                pct = 100 * done / total
                print(
                    f"  [{symbol}] {done}/{total} hours ({pct:.0f}%)",
                    file=sys.stderr,
                    flush=True,
                )

    session.close()

    if not all_ticks:
        logger.error(f"No ticks downloaded for {symbol}")
        return None

    # Sort by timestamp (parallel downloads arrive out of order)
    all_ticks.sort(key=lambda t: t[0])
    logger.info(f"  {len(all_ticks):,} ticks downloaded")

    # Convert to numpy arrays
    timestamps = np.array([t[0] for t in all_ticks], dtype=np.float64)
    bids = np.array([t[1] for t in all_ticks], dtype=np.float64)
    asks = np.array([t[2] for t in all_ticks], dtype=np.float64)
    volumes = np.array([t[3] for t in all_ticks], dtype=np.float64)

    # Mid price for OHLCV
    mids = (bids + asks) / 2.0

    # Resample ticks → OHLCV bars using pandas
    import pandas as pd

    df = pd.DataFrame(
        {"mid": mids, "volume": volumes},
        index=pd.to_datetime(timestamps, unit="s", utc=True),
    )

    rule = _RESAMPLE_RULE.get(tf)
    if rule is None:
        logger.error(f"Unknown timeframe '{tf}'. Supported: {list(_RESAMPLE_RULE)}")
        return None

    bars = df["mid"].resample(rule).ohlc().dropna()
    bars["volume"] = df["volume"].resample(rule).sum()
    bars = bars.dropna()

    if len(bars) == 0:
        if existing_df is not None and len(existing_df) > 0:
            logger.info(f"  No new bars for {symbol} — keeping existing data")
            return out_path
        logger.error(f"No bars after resampling {symbol} to {tf}")
        return None

    # ── Merge with existing data (incremental) ────────────────────────────
    if existing_df is not None and len(existing_df) > 0:
        # Combine: existing bars + new bars — new bars win on overlap
        combined = pd.concat([existing_df, bars])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        bars = combined
        logger.info(
            f"  Merged: {len(existing_df):,} existing + {len(bars) - len(existing_df):,} new "
            f"= {len(bars):,} total bars"
        )

    # Write CSV
    bars.index.name = "time"
    bars.to_csv(out_path)

    logger.info(
        f"  {symbol}_{tf}: {len(bars):,} bars written to {out_path}"
    )
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Stock download via yfinance
# ─────────────────────────────────────────────────────────────────────────────

# yfinance max history by interval (conservative — actual limits may be higher)
_YF_MAX_DAYS = {
    "1m": 6, "5m": 58, "15m": 58, "30m": 58,
    "1h": 725, "4h": 725, "1d": 99999, "1wk": 99999,
}

_TF_TO_YF_INTERVAL = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1H": "1h", "4H": "1h", "1D": "1d",
}


def _is_forex(symbol: str) -> bool:
    """Heuristic: 6-letter all-alpha uppercase → forex pair."""
    s = symbol.upper().replace("/", "")
    if s in _POINT:
        return True
    if len(s) == 6 and s.isalpha():
        return True
    return False


def download_stock(
    symbol: str,
    start: datetime,
    end: datetime,
    tf: str = "1H",
    out_dir: Path = Path("data"),
) -> Optional[Path]:
    """Download stock/ETF/index data from yfinance.

    For 15m/5m/30m: max ~60 days of history.
    For 1H: max ~730 days (2 years).
    For 1D: unlimited.

    Automatically degrades to 1H if the requested range exceeds the 15m limit,
    and to 1D if it exceeds the 1H limit.
    """
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        logger.error("yfinance required for stocks: pip install yfinance")
        return None

    # Strip exchange prefix (NASDAQ:AAPL → AAPL)
    ticker = symbol.split(":")[-1] if ":" in symbol else symbol
    clean_sym = ticker.upper()

    yf_interval = _TF_TO_YF_INTERVAL.get(tf, "1h")
    range_days = (end - start).days

    # Check limits and auto-degrade
    max_days = _YF_MAX_DAYS.get(yf_interval, 60)
    actual_tf = tf
    if range_days > max_days:
        if yf_interval in ("1m", "5m", "15m", "30m"):
            logger.warning(
                f"{clean_sym}: {tf} only available for {max_days} days, "
                f"but {range_days} days requested — degrading to 1H"
            )
            yf_interval = "1h"
            actual_tf = "1H"
            max_days = _YF_MAX_DAYS["1h"]
        if range_days > max_days:
            logger.warning(
                f"{clean_sym}: 1H only available for {max_days} days, "
                f"but {range_days} days requested — degrading to 1D"
            )
            yf_interval = "1d"
            actual_tf = "1D"

    logger.info(f"Downloading {clean_sym} via yfinance ({actual_tf}, {yf_interval})")

    t = yf.Ticker(ticker)
    df = t.history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=yf_interval,
    )

    if df is None or len(df) == 0:
        logger.error(f"No data returned for {clean_sym}")
        return None

    # Normalize columns
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df = df[["open", "high", "low", "close", "volume"]].dropna()

    # Resample 1H → 4H if requested
    if tf == "4H" and yf_interval == "1h":
        bars = df["open"].resample("4h").first().to_frame()
        bars["high"] = df["high"].resample("4h").max()
        bars["low"] = df["low"].resample("4h").min()
        bars["close"] = df["close"].resample("4h").last()
        bars["volume"] = df["volume"].resample("4h").sum()
        df = bars.dropna()

    if len(df) == 0:
        logger.error(f"No bars after processing {clean_sym}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clean_sym}_{actual_tf}.csv"
    df.index.name = "time"
    df.to_csv(out_path)

    logger.info(f"  {clean_sym}_{actual_tf}: {len(df):,} bars written to {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download historical market data (forex from Dukascopy, stocks from yfinance)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Forex: 2 years of 15m data for major pairs (Dukascopy, free)
  python scripts/download_forex_data.py

  # Stocks: 2 years of 1H data (yfinance, free — max ~730 days at 1H)
  python scripts/download_forex_data.py --symbols AAPL MSFT NVDA --tf 1H

  # Mix forex + stocks (auto-detects which source to use)
  python scripts/download_forex_data.py --symbols EURUSD AAPL SPY --tf 1H

  # Then run backtest with downloaded data
  python -m tradingagents_v2.backtesting \\
      --data-dir ./data/1H --symbols AAPL MSFT --start 2024-06-01 --end 2026-03-31
        """,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["EURUSD", "GBPUSD", "USDJPY"],
        help="Symbols to download — forex pairs (EURUSD) or stocks (AAPL, NASDAQ:NVDA)",
    )
    parser.add_argument(
        "--start",
        default="2024-01-01",
        help="Start date YYYY-MM-DD (default: 2024-01-01)",
    )
    parser.add_argument(
        "--end",
        default="2026-03-31",
        help="End date YYYY-MM-DD (default: 2026-03-31)",
    )
    parser.add_argument(
        "--tf",
        default="15m",
        choices=list(_RESAMPLE_RULE.keys()),
        help="Bar timeframe (default: 15m)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: ./data/{tf})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel download threads for Dukascopy (default: 8)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    out_dir = Path(args.out) if args.out else Path(f"data/{args.tf}")

    # Split symbols into forex (Dukascopy) vs stocks (yfinance)
    forex_syms = [s for s in args.symbols if _is_forex(s)]
    stock_syms = [s for s in args.symbols if not _is_forex(s)]

    if forex_syms:
        print(
            f"Forex (Dukascopy): {forex_syms}\n"
            f"  Range: {args.start} → {args.end}  TF: {args.tf}  Workers: {args.workers}"
        )
    if stock_syms:
        print(
            f"Stocks (yfinance): {stock_syms}\n"
            f"  Range: {args.start} → {args.end}  TF: {args.tf}"
        )
    print(f"  Output: {out_dir}/\n")

    # Download forex via Dukascopy
    for sym in forex_syms:
        result = download_symbol(
            sym, start, end, tf=args.tf, out_dir=out_dir, max_workers=args.workers,
        )
        if result is None:
            print(f"  FAILED: {sym}", file=sys.stderr)

    # Download stocks via yfinance
    for sym in stock_syms:
        result = download_stock(sym, start, end, tf=args.tf, out_dir=out_dir)
        if result is None:
            print(f"  FAILED: {sym}", file=sys.stderr)

    # Also generate 1D bars by resampling (needed for D1 agents)
    all_syms = forex_syms + [s.split(":")[-1].upper() for s in stock_syms]
    if args.tf != "1D":
        import pandas as pd
        for sym in all_syms:
            # Try the requested TF first, then fallback to actual TF written
            tf_path = out_dir / f"{sym}_{args.tf}.csv"
            if not tf_path.exists():
                # Stock may have degraded to 1H
                tf_path = out_dir / f"{sym}_1H.csv"
            if not tf_path.exists():
                continue
            df = pd.read_csv(tf_path, index_col="time")
            df.index = pd.to_datetime(df.index, utc=True)
            d1 = df[["open"]].resample("1D").first()
            d1["high"] = df["high"].resample("1D").max()
            d1["low"] = df["low"].resample("1D").min()
            d1["close"] = df["close"].resample("1D").last()
            d1["volume"] = df["volume"].resample("1D").sum()
            d1 = d1.dropna()
            d1_path = out_dir / f"{sym}_1D.csv"
            d1.to_csv(d1_path)
            print(f"  {sym}_1D: {len(d1)} bars → {d1_path}")

    print("\nDone. Run backtest with:")
    print(f"  python -m tradingagents_v2.backtesting \\")
    print(f"      --data-dir {out_dir} \\")
    print(f"      --symbols {' '.join(all_syms)} \\")
    print(f"      --start {args.start} --end {args.end} --profile balanced")


if __name__ == "__main__":
    main()
