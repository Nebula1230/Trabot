"""
Kelly Criterion position-size tracker.

Reads per-symbol closed-trade history from MT5, computes empirical win-rate
and average reward/risk ratio, then returns a position-size multiplier via
the half-Kelly formula.

Formula:
    f* = (p × R − q) / R       (Kelly fraction)
    half_kelly = 0.5 × f*       (conservative version)
    multiplier = 1.0 + half_kelly

where:
    p  = historical win-rate
    q  = 1 − p
    R  = average-win / average-loss  (reward/risk in account-currency units)

A multiplier of 1.0 means "use the configured base risk as-is."
With a 60% win rate and 2:1 R:R the formula yields:
    f* = (0.6×2 − 0.4) / 2 = 0.40  → half = 0.20  → mult = 1.20 (+20%)

With a 40% win rate and 1.2:1 R:R (edge below break-even at half-Kelly):
    f* = (0.4×1.2 − 0.6) / 1.2 = 0.0 → mult = 1.0  (no adjustment)

The multiplier is clamped to [min_mult, max_mult] for safety.
"""

import logging
import numpy as np
from typing import Dict


class KellyTracker:
    """
    Adaptive position-size multiplier based on per-symbol track record.

    Requires a minimum number of closed trades before adjusting size.
    Falls back to 1.0× (baseline) while building history.
    """

    def __init__(self, executor, config: Dict = None):
        self.executor     = executor
        self.config       = config or {}
        self.logger       = logging.getLogger("KellyTracker")

        kelly_cfg = self.config.get("kelly", {})
        self.min_trades    = int(kelly_cfg.get("min_trades",    20))
        self.half_kelly    = bool(kelly_cfg.get("half_kelly",   True))
        self.max_mult      = float(kelly_cfg.get("max_mult",     1.75))
        self.min_mult      = float(kelly_cfg.get("min_mult",     0.60))
        self.lookback_days = int(kelly_cfg.get("lookback_days",  60))

        # Simple in-process cache: {symbol: (timestamp, multiplier)}
        self._cache: Dict[str, tuple] = {}
        self._CACHE_TTL = 3600.0   # refresh at most every hour

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_multiplier(self, symbol: str) -> float:
        """
        Return a position-size multiplier for *symbol* in [min_mult, max_mult].

        1.0  = baseline risk (not enough history yet, or break-even edge).
        >1.0 = proven positive expectancy — bet proportionally more.
        <1.0 = negative expectancy — reduce risk until record improves.
        """
        import time as _t
        now = _t.time()

        # Check cache
        if symbol in self._cache:
            ts, mult = self._cache[symbol]
            if (now - ts) < self._CACHE_TTL:
                return mult

        mult = self._compute_multiplier(symbol)
        self._cache[symbol] = (now, mult)
        return mult

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_multiplier(self, symbol: str) -> float:
        stats = self._symbol_stats(symbol)
        n     = stats["n"]

        if n < self.min_trades:
            self.logger.debug(
                f"Kelly [{symbol}]: only {n}/{self.min_trades} trades — using 1.0×"
            )
            return 1.0

        p   = stats["win_rate"]
        rr  = stats["avg_rr"]
        q   = 1.0 - p

        if rr <= 0 or p <= 0:
            return self.min_mult

        # Kelly fraction
        kelly_f = (p * rr - q) / rr
        if self.half_kelly:
            kelly_f *= 0.5

        # Map to multiplier: kelly_f=0 → 1.0; positive → above 1.0
        mult = float(np.clip(1.0 + kelly_f, self.min_mult, self.max_mult))

        self.logger.info(
            f"Kelly [{symbol}]: n={n} win={p:.1%} rr={rr:.2f} "
            f"f={kelly_f:.3f} → mult={mult:.2f}"
        )
        return mult

    def _symbol_stats(self, symbol: str) -> dict:
        """Compute win-rate and avg R:R from MT5 closed-deal history."""
        try:
            closed = self.executor.get_closed_trades(days=self.lookback_days)
        except Exception as exc:
            self.logger.warning(f"KellyTracker: MT5 history unavailable ({exc})")
            return {"n": 0, "win_rate": 0.5, "avg_rr": 2.0}

        if closed is None:
            closed = []
        trades = [t for t in closed if t.get("symbol") == symbol]
        n      = len(trades)
        if n == 0:
            return {"n": 0, "win_rate": 0.5, "avg_rr": 2.0}

        wins   = [t["profit"] for t in trades if t["profit"] >  0]
        losses = [t["profit"] for t in trades if t["profit"] <= 0]

        win_rate = len(wins) / n
        avg_win  = float(np.mean(wins))         if wins   else 0.0
        avg_loss = float(np.mean(np.abs(losses))) if losses else 1.0
        avg_rr   = avg_win / max(avg_loss, 1e-9)

        return {
            "n":        n,
            "win_rate": win_rate,
            "avg_win":  avg_win,
            "avg_loss": avg_loss,
            "avg_rr":   avg_rr,
        }
