"""
Checkpoint / resume support for incremental backtests.

Periodically saves engine state to a JSON file so that an interrupted backtest
can be resumed from the last checkpoint instead of starting over.

Usage (CLI):
    # First run — automatically saves checkpoints to .checkpoint.json:
    python -m tradingagents_v2.backtesting --start 2025-01-01 --end 2026-01-01 ...

    # If interrupted, resume from checkpoint:
    python -m tradingagents_v2.backtesting --resume backtest_report.checkpoint.json ...

Checkpoint file is deleted automatically on successful completion.
"""

import json
import logging
import os
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("BacktestCheckpoint")


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sim_pos_to_dict(pos) -> dict:
    """Serialise a SimPosition to a JSON-safe dict."""
    d = {}
    for f in fields(pos):
        v = getattr(pos, f.name)
        if isinstance(v, datetime):
            d[f.name] = v.isoformat()
        else:
            d[f.name] = v
    return d


def _dict_to_sim_pos(d: dict, cls):
    """Restore a SimPosition from a dict."""
    kw = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        if f.type == "Optional[datetime]" and isinstance(v, str):
            v = datetime.fromisoformat(v)
        kw[f.name] = v
    return cls(**kw)


def _closed_trade_to_dict(t) -> dict:
    """Serialise a ClosedTrade to a JSON-safe dict."""
    d = {}
    for f in fields(t):
        v = getattr(t, f.name)
        if isinstance(v, datetime):
            d[f.name] = v.isoformat()
        elif v is None:
            d[f.name] = None
        else:
            d[f.name] = v
    return d


def _dict_to_closed_trade(d: dict, cls):
    """Restore a ClosedTrade from a dict."""
    kw = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        # Restore datetime fields
        if "dt" in f.name and isinstance(v, str):
            v = datetime.fromisoformat(v)
        kw[f.name] = v
    return cls(**kw)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint data structure
# ─────────────────────────────────────────────────────────────────────────────

class BacktestCheckpoint:
    """Captures the full mutable state of a single-symbol backtest loop."""

    def __init__(
        self,
        # Identity
        symbol: str,
        initial_equity: float,
        # Progress
        step: int,                     # current bar index
        n_bars: int,                   # total bars
        sig_bar: int = 0,              # first signal bar (for resume offset)
        # Core state
        equity: float = 0.0,
        equity_curve: Optional[List[float]] = None,
        equity_dates: Optional[List[str]] = None,       # ISO strings
        open_positions: Optional[List[dict]] = None,
        closed_trades: Optional[List[dict]] = None,
        # Circuit-breaker / guard state
        day_open_equity: float = 0.0,
        current_day: Optional[int] = None,
        day_halted: bool = False,
        day_trade_count: int = 0,
        day_sl_count: Optional[Dict[str, int]] = None,
        wk_open_equity: float = 0.0,
        current_week: Optional[int] = None,
        wk_halted: bool = False,
        weekend_blocked: bool = False,
        # Entry guards
        last_entry: Optional[Dict[str, str]] = None,     # symbol → ISO datetime
        last_order_attempt: Optional[Dict[str, str]] = None,
        # Streak guards
        streak_history: Optional[Dict[str, list]] = None,
        streak_block_until: Optional[Dict[str, int]] = None,
        streak_block_dir: Optional[Dict[str, str]] = None,
        streak_block_count: Optional[Dict[str, int]] = None,
        # Margin
        margin_used: float = 0.0,
        # Optional identity (not always needed)
        profile: str = "",
        start_date: str = "",
        end_date: str = "",
        mid_tf: str = "",
        # Meta
        checkpoint_time: str = "",
        version: int = 1,
    ):
        self.symbol = symbol
        self.profile = profile
        self.start_date = start_date
        self.end_date = end_date
        self.mid_tf = mid_tf
        self.initial_equity = initial_equity
        self.step = step
        self.n_bars = n_bars
        self.sig_bar = sig_bar
        self.equity = equity
        self.equity_curve = equity_curve or []
        self.equity_dates = equity_dates or []
        self.open_positions = open_positions or []
        self.closed_trades = closed_trades or []
        self.day_open_equity = day_open_equity
        self.current_day = current_day
        self.day_halted = day_halted
        self.day_trade_count = day_trade_count
        self.day_sl_count = day_sl_count or {}
        self.wk_open_equity = wk_open_equity
        self.current_week = current_week
        self.wk_halted = wk_halted
        self.weekend_blocked = weekend_blocked
        self.last_entry = last_entry or {}
        self.last_order_attempt = last_order_attempt or {}
        self.streak_history = streak_history or {}
        self.streak_block_until = streak_block_until or {}
        self.streak_block_dir = streak_block_dir or {}
        self.streak_block_count = streak_block_count or {}
        self.margin_used = margin_used
        self.checkpoint_time = checkpoint_time or datetime.now(timezone.utc).isoformat()
        self.version = version

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "BacktestCheckpoint":
        # Filter to known fields only
        import inspect
        sig = inspect.signature(cls.__init__)
        valid = {k for k in sig.parameters if k != "self"}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio checkpoint (multi-symbol)
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioCheckpoint:
    """Captures state for a multi-symbol portfolio backtest."""

    def __init__(
        self,
        symbols: List[str],
        profile: str,
        start_date: str,
        end_date: str,
        mid_tf: str,
        initial_equity: float,
        # Progress
        global_step: int,             # unified timeline index
        n_steps: int,
        # Shared state
        equity: float,
        equity_curve: List[float],
        equity_dates: List[str],
        # Per-symbol
        per_symbol: Dict[str, dict],  # sym → BacktestCheckpoint.to_dict()
        # All closed trades across symbols
        all_closed_trades: List[dict],
        # Shared guards
        day_open_equity: float,
        current_day: Optional[int],
        day_halted: bool,
        wk_open_equity: float,
        current_week: Optional[int],
        wk_halted: bool,
        # Per-symbol realized PnL
        sym_realized: Dict[str, float],
        sym_equity_curves: Dict[str, List[float]],
        # Meta
        checkpoint_time: str = "",
        version: int = 1,
    ):
        self.__dict__.update(locals())
        del self.__dict__["self"]
        if not self.checkpoint_time:
            self.checkpoint_time = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "PortfolioCheckpoint":
        import inspect
        sig = inspect.signature(cls.__init__)
        valid = {k for k in sig.parameters if k != "self"}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(ckpt, path: str) -> None:
    """Atomically write checkpoint to disk (write-then-rename for crash safety)."""
    tmp = path + ".tmp"
    data = ckpt.to_dict() if hasattr(ckpt, "to_dict") else ckpt
    with open(tmp, "w") as f:
        json.dump(data, f, default=str)
    os.replace(tmp, path)
    logger.info(f"Checkpoint saved → {path}  (step {data.get('step', data.get('global_step', '?'))})")


def load_checkpoint(path: str) -> Optional[dict]:
    """Load checkpoint from disk. Returns raw dict, or None if not found."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    step = data.get("step", data.get("global_step", "?"))
    n = data.get("n_bars", data.get("n_steps", "?"))
    logger.info(f"Checkpoint loaded ← {path}  (step {step}/{n})")
    return data


def remove_checkpoint(path: str) -> None:
    """Remove checkpoint file after successful completion."""
    try:
        os.remove(path)
        logger.info(f"Checkpoint removed: {path}")
    except FileNotFoundError:
        pass


def checkpoint_path_for_output(output_path: str) -> str:
    """Derive checkpoint file path from the output report path."""
    p = Path(output_path)
    return str(p.with_suffix(".checkpoint.json"))
