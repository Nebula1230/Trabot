"""
Agent calibration tracker — records per-agent vote outcomes over closed trades.

For each executed trade the runner records which way each agent voted
(via ``record_trade_votes``).  When the trade closes, ``score_closed_trade``
marks each vote as correct/incorrect and accumulates running statistics.

The result is a persistent JSON file (``logs/agent_calibration.json``) that
survives restarts and enables you to see which agents are adding alpha vs noise.

Scoring logic
─────────────
A vote is considered *correct* when:
  • The agent voted in the same direction as the trade AND the trade won, OR
  • The agent voted against the trade AND the trade lost (correct dissent).

An |dir_score| < 0.05 is counted as "abstain" and not scored.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional


class AgentCalibrationTracker:
    """
    Per-agent hit-rate accumulator.

    Internal state::

        _vote_record: {ticket_str: {symbol, direction, votes: {name: dir_score}, ts}}
        _agent_stats: {agent_name: {n, correct, abstain, total_contribution, last_updated}}
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.log_dir / "agent_calibration.json"
        self.logger = logging.getLogger("AgentCalibrationTracker")
        self._vote_record: Dict[str, Dict] = {}
        self._agent_stats: Dict[str, Dict] = {}
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def record_trade_votes(
        self,
        symbol: str,
        ticket: int,
        direction: str,              # "long" or "short"
        agent_outputs: Dict[str, Any],  # AgentOutput instances or dicts
    ) -> None:
        """
        Record each agent's directional vote at the moment a trade is placed.
        Call this immediately after a trade is executed.
        """
        votes: Dict[str, float] = {}
        for name, out in agent_outputs.items():
            if hasattr(out, "dir_score"):
                votes[name] = float(out.dir_score)
            elif isinstance(out, dict):
                votes[name] = float(out.get("dir_score", 0.0))

        self._vote_record[str(ticket)] = {
            "symbol":    symbol,
            "direction": direction,
            "votes":     votes,
            "ts":        time.time(),
        }

        # Ensure all agents exist in the stats dict
        for name in votes:
            if name not in self._agent_stats:
                self._agent_stats[name] = {
                    "n": 0, "correct": 0, "abstain": 0,
                    "total_contribution": 0.0, "last_updated": None,
                }
        self._save()

    def score_closed_trade(self, ticket: int, outcome: float) -> None:
        """
        Score all agents that voted on this ticket.

        ``outcome`` is the trade's realised P&L in account currency.
        Positive = win, negative = loss.

        Call this when a trade is closed by any means (SL, TP, signal, time-stop).
        """
        key = str(ticket)
        record = self._vote_record.pop(key, None)
        if record is None:
            return

        direction = record["direction"]
        votes     = record["votes"]
        is_win    = outcome > 0
        dir_sign  = 1.0 if direction == "long" else -1.0

        for name, dir_score in votes.items():
            if name not in self._agent_stats:
                self._agent_stats[name] = {
                    "n": 0, "correct": 0, "abstain": 0,
                    "total_contribution": 0.0, "last_updated": None,
                }
            stats = self._agent_stats[name]

            if abs(dir_score) < 0.05:
                stats["abstain"] = stats.get("abstain", 0) + 1
                continue

            stats["n"] += 1
            agent_agreed = (dir_score * dir_sign) > 0

            # Correct when: (agreed AND won) OR (disagreed AND lost)
            if (agent_agreed and is_win) or (not agent_agreed and not is_win):
                stats["correct"] += 1
                stats["total_contribution"] += abs(dir_score)

            stats["last_updated"] = time.strftime("%Y-%m-%d %H:%M")

        self._save()

    def get_stats(self) -> Dict[str, Dict]:
        """
        Return per-agent calibration statistics.

        Keys per agent:
            n_trades, hit_rate, abstain_count, avg_contribution,
            total_contribution, last_updated.
        """
        result: Dict[str, Dict] = {}
        for name, s in self._agent_stats.items():
            n = s.get("n", 0)
            result[name] = {
                "n_trades":           n,
                "hit_rate":           round(s.get("correct", 0) / max(n, 1), 4),
                "abstain_count":      s.get("abstain", 0),
                "total_contribution": round(s.get("total_contribution", 0.0), 3),
                "avg_contribution":   round(s.get("total_contribution", 0.0) / max(n, 1), 3),
                "last_updated":       s.get("last_updated"),
            }
        return result

    def get_stats_table(self) -> str:
        """Return a formatted text table suitable for terminal display."""
        stats = self.get_stats()
        if not stats:
            return "No agent calibration data yet."

        header = (
            f"{'Agent':<32} {'N':>6} {'Hit%':>7} "
            f"{'Abstain':>8} {'AvgScore':>10} {'Last Updated':<22}"
        )
        sep = "─" * 90
        lines = [header, sep]
        for name, s in sorted(stats.items(), key=lambda x: -x[1]["hit_rate"]):
            lines.append(
                f"{name:<32} {s['n_trades']:>6} {s['hit_rate']:>7.1%} "
                f"{s['abstain_count']:>8} {s['avg_contribution']:>10.3f} "
                f"{s['last_updated'] or 'never':<22}"
            )
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._vote_record = raw.get("vote_record", {})
            self._agent_stats = raw.get("agent_stats", {})
        except FileNotFoundError:
            pass
        except Exception as exc:
            self.logger.warning(f"Could not load calibration data: {exc}")

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(
                    {"vote_record": self._vote_record, "agent_stats": self._agent_stats},
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            self.logger.warning(f"Could not save calibration data: {exc}")
