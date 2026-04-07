"""Shared signal ranking & slot-filling logic.

Used identically by the backtesting engine and the MT5 live runner so that
both environments apply the exact same pre-filters, sort order, and
fill-time guards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Set


@dataclass
class RankCandidate:
    """One candidate signal ready to be ranked.

    *payload* is opaque — the caller stashes whatever it needs for
    post-selection execution (e.g. an order dict or a graph state dict).
    """

    symbol: str
    ev: float                                        # expected value — sort key
    confidence: float
    sl_distance: float                               # |entry − SL|; ≈0 → invalid
    day_sl_count: int = 0                            # per-symbol daily SL count
    streak_blocked: bool = False                     # True → skip
    last_entry_elapsed_min: Optional[float] = None   # None = never traded
    payload: Any = None


@dataclass
class RankConfig:
    """Guards applied during the ranked-fill phase."""

    max_concurrent: int = 0           # 0 = unlimited
    open_position_count: int = 0      # current non-pending open count
    max_daily: int = 0                # 0 = unlimited
    daily_trade_count: int = 0        # trades placed today so far
    cooldown_minutes: float = 0.0     # per-symbol cooldown
    max_daily_sl: int = 0             # per-symbol daily SL cap; 0 = unlimited


def rank_and_select(
    candidates: List[RankCandidate],
    cfg: RankConfig,
) -> List[RankCandidate]:
    """Shared ranking pipeline.

    1. **Pre-filter** – drop candidates with zero SL distance,
       daily-SL-capped symbols, and streak-blocked entries.
    2. **Sort** – remaining candidates by expected value (descending).
    3. **Fill** – from best to worst, stopping when global caps
       (max_concurrent, max_daily) are reached and skipping per-symbol
       cooldown violations.

    Returns the ordered list of accepted candidates.
    """

    # ── Pre-filter ──────────────────────────────────────────────────────
    valid: List[RankCandidate] = []
    for c in candidates:
        if c.sl_distance < 1e-7:
            continue
        if cfg.max_daily_sl > 0 and c.day_sl_count >= cfg.max_daily_sl:
            continue
        if c.streak_blocked:
            continue
        valid.append(c)

    # ── Sort by EV descending ───────────────────────────────────────────
    valid.sort(key=lambda c: c.ev, reverse=True)

    # ── Fill slots ──────────────────────────────────────────────────────
    selected: List[RankCandidate] = []
    _open = cfg.open_position_count
    _daily = cfg.daily_trade_count
    _filled_syms: Set[str] = set()

    for c in valid:
        if cfg.max_concurrent > 0 and _open >= cfg.max_concurrent:
            break
        if cfg.max_daily > 0 and _daily >= cfg.max_daily:
            break
        # If we already filled a candidate for this symbol in this batch,
        # skip (mirrors the per-symbol cooldown re-check).
        if c.symbol in _filled_syms:
            continue
        if cfg.cooldown_minutes > 0 and c.last_entry_elapsed_min is not None:
            if c.last_entry_elapsed_min < cfg.cooldown_minutes:
                continue

        selected.append(c)
        _open += 1
        _daily += 1
        _filled_syms.add(c.symbol)

    return selected
