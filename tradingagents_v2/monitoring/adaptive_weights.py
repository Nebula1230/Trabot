"""
Adaptive Agent Weight Manager

Reads per-agent hit-rate statistics from ``AgentCalibrationTracker`` and
periodically updates ``agent.weight`` on each registered agent so that
consistently accurate agents earn more influence in the fusion step, while
noisy or contrarian agents are down-weighted.

Design
──────
The adjustment formula is a shrinkage-regularised hit-rate edge:

    hit_edge   = hit_rate − 0.50          # excess accuracy above chance
    shrinkage  = n / (n + shrink_n)       # Bayesian pull-to-prior (0→1)
    adjusted   = base_weight
                  × (1 + sensitivity × hit_edge × shrinkage)

Clamped to [base_weight × min_mult, base_weight × max_mult] so we never
mute an agent entirely or let a lucky streak explode its influence.

Key properties
──────────────
• Below ``min_trades`` the base weight is kept unchanged (no speculation on
  thin history).
• ``shrink_n`` is the "prior strength": at n=shrink_n the correction is 50%
  of what it would be at full confidence.  Default 60 means we need ~120
  trades before trusting the hit-rate fully.
• ``sensitivity=2.0`` means a +10pp edge (60% hit rate) produces a +10%
  weight boost, rising to +20% at +20pp.  This is deliberately conservative.
• Thread-safe: the update runs in the main async loop; ``agent.weight`` is a
  plain float attribute, so no locking is required for single-writer patterns.
• ``update()`` is idempotent — calling it more often than the interval is a
  no-op (TTL guard).

Integration
───────────
The runner calls ``update()`` once per slow cycle.  The TTL (update_interval_hours)
limits actual recomputation to at most every N hours, so the CPU overhead is
negligible.
"""

import logging
import time
from typing import Dict, Optional

from ..core.agent_base import AgentRegistry
from .agent_tracker import AgentCalibrationTracker


class AdaptiveWeightManager:
    """
    Periodic agent-weight updater driven by closed-trade hit-rate statistics.

    Parameters
    ----------
    registry : AgentRegistry
        The live registry whose ``agent.weight`` attributes will be updated.
    cal_tracker : AgentCalibrationTracker
        The calibration tracker that accumulates per-agent vote outcomes.
    config : dict, optional
        The ``adaptive_weights`` sub-dict from the main config::

            adaptive_weights:
              enabled:               true
              sensitivity:           2.0
              min_trades:            30
              max_mult:              2.50
              min_mult:              0.40
              shrink_n:              60
              update_interval_hours: 4.0
              log_updates:           true
    """

    def __init__(
        self,
        registry: AgentRegistry,
        cal_tracker: AgentCalibrationTracker,
        config: Optional[Dict] = None,
    ):
        self._registry    = registry
        self._tracker     = cal_tracker
        self.logger       = logging.getLogger("AdaptiveWeightManager")

        cfg = config or {}
        self._enabled      = bool(cfg.get("enabled", False))
        self._sensitivity  = float(cfg.get("sensitivity", 2.0))
        self._min_trades   = int(cfg.get("min_trades", 30))
        self._max_mult     = float(cfg.get("max_mult", 2.50))
        self._min_mult     = float(cfg.get("min_mult", 0.40))
        self._shrink_n     = float(cfg.get("shrink_n", 60))
        self._interval_s   = float(cfg.get("update_interval_hours", 4.0)) * 3600.0
        self._log_updates  = bool(cfg.get("log_updates", True))

        # Snapshot the *initial* weights from the registry at construction time.
        # These are the profile-calibrated baselines — adaptive adjustments are
        # always a multiplicative deviation from this anchor, not from the last
        # adjusted value.  This prevents drifting away from the designer's intent
        # across multiple consecutive updates.
        self._base_weights: Dict[str, float] = {
            agent.name: agent.weight
            for agent in self._registry.get_all_agents()
        }
        self._last_update: float = 0.0   # epoch seconds of last actual computation

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def update(self) -> None:
        """
        Recompute adaptive weights from calibration statistics and apply them
        to the registry agents.  No-op when:
          - ``enabled`` is False
          - The TTL since the last update has not elapsed
        """
        if not self._enabled:
            return

        now = time.time()
        if (now - self._last_update) < self._interval_s:
            return   # TTL not elapsed — skip
        self._last_update = now

        stats = self._tracker.get_stats()
        if not stats:
            return

        changes: Dict[str, tuple] = {}   # name → (old_weight, new_weight)

        for agent in self._registry.get_all_agents():
            name = agent.name
            base = self._base_weights.get(name, agent.weight)
            s    = stats.get(name)

            if s is None or s["n_trades"] < self._min_trades:
                # Not enough history — keep current weight at base (no drift)
                if agent.weight != base:
                    agent.weight = base
                continue

            n         = s["n_trades"]
            hit_rate  = s["hit_rate"]

            # Excess accuracy above random (−0.5 → +0.5)
            hit_edge = hit_rate - 0.50

            # Bayesian shrinkage: at n=shrink_n the adjustment is 50 % of full.
            # Prevents over-reacting to a run of luck on thin data.
            shrinkage = n / (n + self._shrink_n)

            # Raw adjusted weight
            raw = base * (1.0 + self._sensitivity * hit_edge * shrinkage)

            # Clamp to [base×min_mult, base×max_mult]
            lo  = base * self._min_mult
            hi  = base * self._max_mult
            new = max(lo, min(hi, raw))

            old = agent.weight
            agent.weight = round(new, 4)
            if abs(new - old) > 1e-6:
                changes[name] = (round(old, 4), round(new, 4))

        if changes and self._log_updates:
            lines = [
                f"  {name:<32}  {old:.4f} → {new:.4f}"
                for name, (old, new) in sorted(changes.items())
            ]
            self.logger.info(
                "[AdaptiveWeights] Updated %d agent weight(s):\n%s",
                len(changes),
                "\n".join(lines),
            )
        elif self._log_updates:
            self.logger.debug("[AdaptiveWeights] No weight changes this cycle")

    def get_status(self) -> Dict:
        """
        Return a snapshot of current adaptive weights and calibration stats
        for observability / CLI display.

        Returns
        -------
        dict
            ``{agent_name: {base, current, hit_rate, n_trades, adj_factor}}``
        """
        stats  = self._tracker.get_stats()
        result = {}
        for agent in self._registry.get_all_agents():
            name = agent.name
            base = self._base_weights.get(name, agent.weight)
            s    = stats.get(name, {})
            n    = s.get("n_trades", 0)
            hr   = s.get("hit_rate", 0.0)
            result[name] = {
                "base":       round(base, 4),
                "current":    round(agent.weight, 4),
                "hit_rate":   hr,
                "n_trades":   n,
                "adj_factor": round(agent.weight / base, 4) if base > 0 else 1.0,
                "calibrated": n >= self._min_trades,
            }
        return result

    def print_status(self) -> str:
        """Return a formatted ASCII table of adaptive weight status."""
        status = self.get_status()
        if not status:
            return "No agents registered."
        header = (
            f"{'Agent':<32} {'Base':>6} {'Current':>8} "
            f"{'Factor':>7} {'HitRate':>8} {'N':>6} {'Calibrated':>12}"
        )
        sep = "─" * 92
        lines = [header, sep]
        for name, s in sorted(status.items()):
            cal_tag = "✓" if s["calibrated"] else "pending"
            lines.append(
                f"{name:<32} {s['base']:>6.3f} {s['current']:>8.4f} "
                f"{s['adj_factor']:>7.4f} {s['hit_rate']:>8.1%} "
                f"{s['n_trades']:>6} {cal_tag:>12}"
            )
        return "\n".join(lines)
