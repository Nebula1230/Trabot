"""
Trailing stop manager — staged lock-in strategy to secure gains.

Stages (based on initial SL distance = 1R):
  Stage 1  — price moves +1R  → SL to break-even (entry price)
  Stage 2  — price moves +1.5R → SL to entry + 0.5R  (locks 50% of target)
  Stage 3  — price moves +2R+  → ATR trail (keeps riding the trend)

Rules:
  - SL is never moved against the trade (only ratchets in profit direction)
  - Uses the original SL distance (|entry - sl|) as 1R, not ATR
  - ATR trail in stage 3 uses atr_multiplier (default 1.5)
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..execution.mt5_executor import MT5Executor
    from ..data.loader import DataLoader


class TrailingStopManager:
    """
    Staged break-even + lock-in trailing stop manager.

    Parameters
    ----------
    executor : MT5Executor
    atr_multiplier : float
        ATR trail distance multiplier used in stage 3. Default 1.5.
    min_profit_atr : float
        (Unused legacy param, kept for API compatibility.)
    """

    def __init__(self, executor: "MT5Executor",
                 atr_multiplier: float = 1.5,
                 min_profit_atr: float = 1.0,
                 tp_extend_enabled: bool = True,
                 tp_extend_atr_mult: float = 2.0,
                 partial_tp_enabled: bool = False,
                 partial_tp_fraction: float = 0.50,
                 partial_tp2_enabled: bool = False,
                 partial_tp2_r_mult: float = 2.0,
                 partial_tp2_fraction: float = 0.50,
                 windfall_exit_enabled: bool = False,
                 windfall_r_mult: float = 3.0,
                 early_be_r: float = 0.0,
                 be_buffer_r: float = 0.0,
                 stale_sl_hours: float = 0.0,
                 stale_sl_r_mult: float = 0.75,
                 profile: str = ""):
        self.executor = executor
        self.profile = profile
        self.atr_multiplier = atr_multiplier
        self.tp_extend_enabled = tp_extend_enabled
        self.tp_extend_atr_mult = tp_extend_atr_mult
        # ── Early breakeven: move SL to entry at +early_be_r (e.g. +0.5R) ───
        # Fires BEFORE stage 1 (+1R). Protects trades that briefly run +0.3–0.9R
        # then reverse to full -1R SL hit.  0 = disabled.
        self.early_be_r = early_be_r
        # ── Breakeven buffer: room beyond exact entry for BE SL moves ─────
        # Covers spread cost and noise; avoids ~0R stops that waste partial gains.
        self.be_buffer_r = be_buffer_r
        # ── Stale SL: tighten SL after N hours if trade still in loss ──────
        # Caps max loss at -stale_sl_r_mult × 1R to reduce late SL hits.
        self.stale_sl_hours  = stale_sl_hours
        self.stale_sl_r_mult = stale_sl_r_mult
        # ── Partial TP1: close a fraction at +1R (break-even stage) ──────────
        # Guard: SL still below entry — restart-safe, no state file needed.
        self.partial_tp_enabled  = partial_tp_enabled
        self.partial_tp_fraction = partial_tp_fraction
        # ── Partial TP2: close another fraction when trade extends to +2R ────
        # Guard: SL still at stage-2 level (entry+0.5R for BUY).
        # After firing, SL is bumped to entry+0.7R so the guard can't re-fire.
        self.partial_tp2_enabled  = partial_tp2_enabled
        self.partial_tp2_r_mult   = partial_tp2_r_mult
        self.partial_tp2_fraction = partial_tp2_fraction
        # ── Windfall exit: close ALL when profit hits an exceptional multiple ─
        # Captures spike moves before a reversal can give them back.
        # Restart-safe: no in-memory guard — position gone from MT5 after close.
        self.windfall_exit_enabled = windfall_exit_enabled
        self.windfall_r_mult       = windfall_r_mult
        # Entry ATR snapshot per ticket — for ATR-adaptive stale_sl.
        # On restart, first tick captures current ATR (close enough proxy).
        self._entry_atr: dict[int, float] = {}
        # Peak profit (R) per ticket — for stale winner protection.
        self._peak_profit_r: dict[int, float] = {}
        self.logger = logging.getLogger("TrailingStopManager")

    def update_trails(self, data_loader: "DataLoader"):
        """Iterate all open bot positions and apply staged SL management."""
        positions = self.executor.get_open_positions()
        our_positions = [p for p in positions
                         if p.get("magic") == self.executor.magic_number]
        for pos in our_positions:
            try:
                self._manage_one(pos, data_loader)
            except Exception as e:
                self.logger.error(f"Error managing SL for #{pos.get('ticket')}: {e}")

    def update_structural_sl_tp(self, data_loader: "DataLoader"):
        """
        Recalculate pivot-based SL/TP for every open position and ratchet them.

        Logic per position:
          BUY  -> new_sl = nearest_support - 0.25xATR  (ratchet: only raise SL)
                  new_tp = next resistance above entry   (ratchet: only raise TP)
          SELL -> new_sl = nearest_resist  + 0.25xATR  (ratchet: only lower SL)
                  new_tp = next support below entry      (ratchet: only lower TP)

        Safety rules:
          - SL never widens (never moves further from entry than current SL).
          - SL never crosses entry (won't flip from profit protection to a loss).
          - TP never contracts against us once we are in profit.
          - Minimum SL distance = 1.5xATR (prevents stop-hunting micro-stops).
        """
        positions = self.executor.get_open_positions()
        our_positions = [p for p in positions
                         if p.get("magic") == self.executor.magic_number]
        for pos in our_positions:
            try:
                self._update_structural_one(pos, data_loader)
            except Exception as e:
                self.logger.error(
                    f"Structural SL/TP update error #{pos.get('ticket')}: {e}"
                )

    def _update_structural_one(self, pos: dict, data_loader: "DataLoader"):
        symbol     = pos["symbol"]
        ticket     = pos["ticket"]
        pos_type   = pos["type"].upper()   # "BUY" or "SELL"
        entry      = pos["price_open"]
        price      = pos["price_current"]
        current_sl = pos.get("sl", 0.0)
        current_tp = pos.get("tp", 0.0)

        if current_sl == 0.0:
            return  # unmanaged position -- skip

        features = data_loader.get_features(symbol)
        if features is None:
            return

        atr         = features.atr_14
        buffer      = 0.25 * atr
        min_sl_dist = 1.5 * atr

        prices = self.executor.get_current_price(symbol)
        bid, ask = prices if prices else (price, price)
        live_price = ask if pos_type == "BUY" else bid

        # get_structural_levels() incorporates D1 swing nodes alongside weekly pivots
        pivots = data_loader.get_structural_levels(symbol, live_price, atr)

        if pos_type == "BUY":
            # SL: pull nearest support up, ratchet only
            structural_sl = pivots["nearest_support"] - buffer
            structural_sl = min(structural_sl, live_price - min_sl_dist)
            new_sl = max(current_sl, structural_sl)  # ratchet only — SL never widens

            # TP: push to next resistance, only raise
            resist_candidates = sorted(
                l for l in (pivots["r1"], pivots["r2"]) if l > entry + min_sl_dist
            )
            structural_tp = resist_candidates[0] if resist_candidates else current_tp
            new_tp = max(current_tp, structural_tp) if current_tp > 0 else structural_tp

            # Use 1e-4 (sub-pip) as minimum meaningful change to avoid firing
            # MT5 requests for deltas that will round to zero at the tick level.
            sl_changed = new_sl > current_sl + 1e-4
            tp_changed = new_tp > current_tp + 1e-4

        else:  # SELL
            # SL: pull nearest resistance down, ratchet only
            structural_sl = pivots["nearest_resist"] + buffer
            structural_sl = max(structural_sl, live_price + min_sl_dist)
            new_sl = min(current_sl, structural_sl) if current_sl > 0 else structural_sl  # ratchet only

            # TP: push to next support, only lower
            in_profit = live_price < entry
            support_candidates = sorted(
                (l for l in (pivots["s1"], pivots["s2"]) if l < entry - min_sl_dist),
                reverse=True,
            )
            structural_tp = support_candidates[0] if support_candidates else current_tp
            new_tp = (min(current_tp, structural_tp)
                      if current_tp > 0 else structural_tp) if in_profit else current_tp

            # Use 1e-4 (sub-pip) as minimum meaningful change — same rationale
            # as the BUY branch above.
            sl_changed = (current_sl == 0.0 or new_sl < current_sl - 1e-4)
            tp_changed = (current_tp == 0.0 or new_tp < current_tp - 1e-4)

        if sl_changed or tp_changed:
            self.logger.info(
                f"Structural update {pos_type} {symbol} #{ticket}: "
                f"SL {current_sl:.5f}->{new_sl:.5f}  "
                f"TP {current_tp:.5f}->{new_tp:.5f}  "
                f"(live={live_price:.5f} atr={atr:.5f})"
            )
            final_sl = new_sl if sl_changed else current_sl
            final_tp = new_tp if tp_changed else current_tp
            self.executor.modify_sl_tp(ticket, final_sl, final_tp)

    # ------------------------------------------------------------------

    def _manage_one(self, pos: dict, data_loader: "DataLoader"):
        symbol        = pos["symbol"]
        ticket        = pos["ticket"]
        pos_type      = pos["type"].upper()   # "BUY" or "SELL"
        entry         = pos["price_open"]
        price         = pos["price_current"]
        current_sl    = pos.get("sl", 0.0)
        tp            = pos.get("tp", 0.0)

        # 1R = original SL distance
        if current_sl == 0.0:
            return  # no SL set — nothing to manage
        one_r = abs(entry - current_sl)
        if one_r < 1e-9:
            return

        features = data_loader.get_features(symbol)
        atr = features.atr_14 if features else one_r  # fallback

        # Snapshot entry ATR (first time we see this ticket)
        if ticket not in self._entry_atr:
            self._entry_atr[ticket] = atr

        _be_buf = self.be_buffer_r * one_r

        # ── Progressive stale-SL tightening: trade open > N hours while losing →
        # SL tightens from -stale_r × 1R and progressively squeezes to
        # 0.60 × stale_r × 1R over the next 2× stale_sl_hours.
        if self.stale_sl_hours > 0:
            open_time = pos.get("open_time", 0)
            if open_time > 0:
                import time as _time
                _hours_held = (_time.time() - open_time) / 3600.0
            else:
                _hours_held = 0.0
            _profit_raw = price - entry if pos_type == "BUY" else entry - price
            if _hours_held >= self.stale_sl_hours and _profit_raw < 0:
                # Progressive factor: 1.0 at threshold → 0.60 at threshold + 2×stale_sl_hours
                _overtime_frac = min(1.0, (_hours_held - self.stale_sl_hours)
                                          / (2.0 * self.stale_sl_hours))
                # ATR-adaptive: tighten proportionally when vol compresses
                _e_atr = self._entry_atr.get(ticket, atr)
                _atr_ratio = min(1.0, atr / _e_atr) if _e_atr > 1e-9 else 1.0
                _prog_r = self.stale_sl_r_mult * _atr_ratio * (1.0 - 0.40 * _overtime_frac)
                if pos_type == "BUY":
                    _tight_sl = entry - _prog_r * one_r
                    if current_sl < _tight_sl - 1e-9:
                        self.logger.info(
                            f"[STALE-SL] BUY {symbol} #{ticket}: "
                            f"SL {current_sl:.5f}→{_tight_sl:.5f} "
                            f"(held {_hours_held:.1f}h, cap={_prog_r:.2f}R, "
                            f"progress={_overtime_frac:.0%})"
                        )
                        self.executor.modify_stop_loss(ticket, _tight_sl)
                        current_sl = _tight_sl
                    # Market-close when squeeze is maxed and still losing
                    if _overtime_frac >= 1.0:
                        ok = self.executor.close_position(ticket)
                        if ok:
                            self.logger.info(
                                f"[STALE-CLOSE] BUY {symbol} #{ticket}: "
                                f"squeeze maxed ({_hours_held:.1f}h) — market close"
                            )
                        return
                else:
                    _tight_sl = entry + _prog_r * one_r
                    if current_sl > _tight_sl + 1e-9:
                        self.logger.info(
                            f"[STALE-SL] SELL {symbol} #{ticket}: "
                            f"SL {current_sl:.5f}→{_tight_sl:.5f} "
                            f"(held {_hours_held:.1f}h, cap={_prog_r:.2f}R, "
                            f"progress={_overtime_frac:.0%})"
                        )
                        self.executor.modify_stop_loss(ticket, _tight_sl)
                        current_sl = _tight_sl
                    # Market-close when squeeze is maxed and still losing
                    if _overtime_frac >= 1.0:
                        ok = self.executor.close_position(ticket)
                        if ok:
                            self.logger.info(
                                f"[STALE-CLOSE] SELL {symbol} #{ticket}: "
                                f"squeeze maxed ({_hours_held:.1f}h) — market close"
                            )
                        return

        # ── Track peak profit and protect stalling winners ─────────────────
        _profit_raw_all = price - entry if pos_type == "BUY" else entry - price
        _profit_r_all = _profit_raw_all / one_r
        _prev_peak = self._peak_profit_r.get(ticket, 0.0)
        if _profit_r_all > _prev_peak:
            self._peak_profit_r[ticket] = _profit_r_all
            _prev_peak = _profit_r_all

        if (self.stale_sl_hours > 0
                and _profit_r_all > 0 and _profit_r_all < 1.0
                and _prev_peak - _profit_r_all >= 0.3):
            open_time = pos.get("open_time", 0)
            if open_time > 0:
                import time as _time
                _hours_sw = (_time.time() - open_time) / 3600.0
            else:
                _hours_sw = 0.0
            if _hours_sw >= self.stale_sl_hours:
                if pos_type == "BUY":
                    _be_floor = entry - _be_buf
                    if current_sl < _be_floor - 1e-9:
                        self.executor.modify_stop_loss(ticket, _be_floor)
                        self.logger.info(
                            f"[STALE-WIN] BUY {symbol} #{ticket}: "
                            f"profit stalling (now {_profit_r_all:+.2f}R, "
                            f"peak {_prev_peak:+.2f}R) — SL→BE {current_sl:.5f}→{_be_floor:.5f}"
                        )
                        current_sl = _be_floor
                else:
                    _be_floor = entry + _be_buf
                    if current_sl == 0.0 or current_sl > _be_floor + 1e-9:
                        self.executor.modify_stop_loss(ticket, _be_floor)
                        self.logger.info(
                            f"[STALE-WIN] SELL {symbol} #{ticket}: "
                            f"profit stalling (now {_profit_r_all:+.2f}R, "
                            f"peak {_prev_peak:+.2f}R) — SL→BE {current_sl:.5f}→{_be_floor:.5f}"
                        )
                        current_sl = _be_floor

        if pos_type == "BUY":
            profit = price - entry          # positive = in profit
            new_sl = current_sl             # default: no change
            _be_sl = entry - _be_buf        # BE SL with buffer (below entry for BUY)

            if profit >= 2.0 * one_r:
                # Stage 3: ATR trail — keeps running
                atr_trail = price - atr * self.atr_multiplier
                new_sl = max(current_sl, atr_trail, entry + 0.5 * one_r)
                stage = 3
            elif profit >= 1.5 * one_r:
                # Stage 2: lock in 0.5R above entry
                new_sl = max(current_sl, entry + 0.5 * one_r)
                stage = 2
            elif profit >= 1.0 * one_r:
                # Stage 1: break-even (with buffer)
                new_sl = max(current_sl, _be_sl)
                stage = 1
            elif self.early_be_r > 0 and profit >= self.early_be_r * one_r:
                # Stage 0: early break-even (with buffer)
                new_sl = max(current_sl, _be_sl)
                stage = 0
            else:
                return  # still inside early_be threshold — don't touch

            _pfx = f"[{self.profile.upper()}] " if self.profile else ""

            # ── Windfall exit: close ALL at exceptional R multiple ────────────
            # Captures spike moves before a reversal gives them back.
            # Restart-safe: no in-memory guard needed — close_position is
            # idempotent and the position disappears from MT5 after close.
            if (self.windfall_exit_enabled
                    and profit >= self.windfall_r_mult * one_r):
                ok = self.executor.close_position(ticket)
                if ok:
                    self.logger.info(
                        f"{_pfx}WINDFALL EXIT BUY {symbol} #{ticket}: "
                        f"ALL closed at +{profit/one_r:.1f}R "
                        f"(entry={entry:.5f} price={price:.5f})"
                    )
                return   # position closed — no further SL management

            # ── Partial TP1: fire once at +1R while SL still below entry ─────
            # Guard: current_sl < entry — restart-safe (SL moves to entry after)
            if (self.partial_tp_enabled
                    and stage == 1
                    and current_sl < entry - 1e-5):
                ok = self.executor.close_partial_position(
                    ticket, self.partial_tp_fraction
                )
                if ok:
                    # Move SL to BE + buffer (parity with engine.py partial_tp1)
                    _be_new = entry - _be_buf
                    if _be_new > current_sl + 1e-9:
                        self.executor.modify_stop_loss(ticket, _be_new)
                        self.logger.info(
                            f"{_pfx}PARTIAL TP BUY {symbol} #{ticket}: "
                            f"{self.partial_tp_fraction:.0%} locked in at +1R "
                            f"SL→{_be_new:.5f} (BE+buf) "
                            f"(entry={entry:.5f} price={price:.5f})"
                        )
                        current_sl = _be_new
                    else:
                        self.logger.info(
                            f"{_pfx}PARTIAL TP BUY {symbol} #{ticket}: "
                            f"{self.partial_tp_fraction:.0%} locked in at +1R "
                            f"(entry={entry:.5f} price={price:.5f})"
                        )

            # ── Partial TP2: fire once at +2R while SL still at stage-2 level ─
            # Guard: SL at/near entry+0.5R (set in stage 2, not yet moved by ATR).
            # After firing, new_sl is bumped to entry+0.7R so the guard can't
            # re-fire on the next surveillance tick (restart-safe).
            if (self.partial_tp2_enabled
                    and stage == 3
                    and profit >= self.partial_tp2_r_mult * one_r
                    and current_sl <= entry + 0.52 * one_r):
                ok = self.executor.close_partial_position(
                    ticket, self.partial_tp2_fraction
                )
                if ok:
                    self.logger.info(
                        f"{_pfx}PARTIAL TP2 BUY {symbol} #{ticket}: "
                        f"{self.partial_tp2_fraction:.0%} locked in at "
                        f"+{profit/one_r:.1f}R "
                        f"(entry={entry:.5f} price={price:.5f})"
                    )
                    # Push SL above stage-2 guard threshold to prevent re-fire
                    new_sl = max(new_sl, entry + 0.7 * one_r)

            if new_sl > current_sl:
                self.logger.info(
                    f"SL lock BUY  {symbol} #{ticket} [stage {stage}]: "
                    f"{current_sl:.5f} → {new_sl:.5f}  "
                    f"(entry={entry:.5f} price={price:.5f} 1R={one_r:.5f})"
                )
                # Stage 3: also push TP forward if price is closing in on it
                if (stage == 3 and self.tp_extend_enabled
                        and tp > 0 and (tp - price) < atr):
                    new_tp = price + self.tp_extend_atr_mult * atr
                    self.logger.info(
                        f"TP extend BUY {symbol} #{ticket}: "
                        f"{tp:.5f} → {new_tp:.5f}  (price={price:.5f} ATR={atr:.5f})"
                    )
                    self.executor.modify_sl_tp(ticket, new_sl, new_tp)
                else:
                    self.executor.modify_stop_loss(ticket, new_sl)

        elif pos_type == "SELL":
            profit = entry - price          # positive = in profit
            new_sl = current_sl
            _be_sl = entry + _be_buf        # BE SL with buffer (above entry for SELL)

            if profit >= 2.0 * one_r:
                # Stage 3: ATR trail
                atr_trail = price + atr * self.atr_multiplier
                new_sl = min(current_sl, atr_trail, entry - 0.5 * one_r)
                stage = 3
            elif profit >= 1.5 * one_r:
                # Stage 2: lock in 0.5R below entry
                new_sl = min(current_sl, entry - 0.5 * one_r)
                stage = 2
            elif profit >= 1.0 * one_r:
                # Stage 1: break-even (with buffer)
                new_sl = min(current_sl, _be_sl)
                stage = 1
            elif self.early_be_r > 0 and profit >= self.early_be_r * one_r:
                # Stage 0: early break-even (with buffer)
                new_sl = min(current_sl, _be_sl)
                stage = 0
            else:
                return

            _pfx = f"[{self.profile.upper()}] " if self.profile else ""

            # ── Windfall exit: close ALL at exceptional R multiple ────────────
            # Restart-safe: no in-memory guard needed (position gone after close).
            if (self.windfall_exit_enabled
                    and profit >= self.windfall_r_mult * one_r):
                ok = self.executor.close_position(ticket)
                if ok:
                    self.logger.info(
                        f"{_pfx}WINDFALL EXIT SELL {symbol} #{ticket}: "
                        f"ALL closed at +{profit/one_r:.1f}R "
                        f"(entry={entry:.5f} price={price:.5f})"
                    )
                return   # position closed — no further SL management

            # ── Partial TP1: fire once at +1R while SL still above entry ─────
            # Guard: current_sl > entry — restart-safe (SL moves to entry after)
            if (self.partial_tp_enabled
                    and stage == 1
                    and (current_sl == 0.0 or current_sl > entry + 1e-5)):
                ok = self.executor.close_partial_position(
                    ticket, self.partial_tp_fraction
                )
                if ok:
                    # Move SL to BE + buffer (parity with engine.py partial_tp1)
                    _be_new = entry + _be_buf
                    if current_sl == 0.0 or _be_new < current_sl - 1e-9:
                        self.executor.modify_stop_loss(ticket, _be_new)
                        self.logger.info(
                            f"{_pfx}PARTIAL TP SELL {symbol} #{ticket}: "
                            f"{self.partial_tp_fraction:.0%} locked in at +1R "
                            f"SL→{_be_new:.5f} (BE+buf) "
                            f"(entry={entry:.5f} price={price:.5f})"
                        )
                        current_sl = _be_new
                    else:
                        self.logger.info(
                            f"{_pfx}PARTIAL TP SELL {symbol} #{ticket}: "
                            f"{self.partial_tp_fraction:.0%} locked in at +1R "
                            f"(entry={entry:.5f} price={price:.5f})"
                        )

            # ── Partial TP2: fire once at +2R while SL still at stage-2 level ─
            # Guard: SL at/near entry-0.5R (set in stage 2, not yet moved by ATR).
            # After firing, new_sl is bumped to entry-0.7R so the guard can't
            # re-fire on the next surveillance tick (restart-safe).
            if (self.partial_tp2_enabled
                    and stage == 3
                    and profit >= self.partial_tp2_r_mult * one_r
                    and (current_sl == 0.0 or current_sl >= entry - 0.52 * one_r)):
                ok = self.executor.close_partial_position(
                    ticket, self.partial_tp2_fraction
                )
                if ok:
                    self.logger.info(
                        f"{_pfx}PARTIAL TP2 SELL {symbol} #{ticket}: "
                        f"{self.partial_tp2_fraction:.0%} locked in at "
                        f"+{profit/one_r:.1f}R "
                        f"(entry={entry:.5f} price={price:.5f})"
                    )
                    # Push SL below stage-2 guard threshold to prevent re-fire
                    new_sl = min(new_sl, entry - 0.7 * one_r)

            if current_sl == 0.0 or new_sl < current_sl:
                self.logger.info(
                    f"SL lock SELL {symbol} #{ticket} [stage {stage}]: "
                    f"{current_sl:.5f} → {new_sl:.5f}  "
                    f"(entry={entry:.5f} price={price:.5f} 1R={one_r:.5f})"
                )
                # Stage 3: also push TP forward if price is closing in on it
                if (stage == 3 and self.tp_extend_enabled
                        and tp > 0 and (price - tp) < atr):
                    new_tp = price - self.tp_extend_atr_mult * atr
                    self.logger.info(
                        f"TP extend SELL {symbol} #{ticket}: "
                        f"{tp:.5f} → {new_tp:.5f}  (price={price:.5f} ATR={atr:.5f})"
                    )
                    self.executor.modify_sl_tp(ticket, new_sl, new_tp)
                else:
                    self.executor.modify_stop_loss(ticket, new_sl)
