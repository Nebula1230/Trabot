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
                 tp_extend_atr_mult: float = 2.0):
        self.executor = executor
        self.atr_multiplier = atr_multiplier
        self.tp_extend_enabled = tp_extend_enabled
        self.tp_extend_atr_mult = tp_extend_atr_mult
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
            new_sl = max(current_sl, structural_sl)
            new_sl = min(new_sl, entry - buffer)   # never above entry

            # TP: push to next resistance, only raise
            resist_candidates = sorted(
                l for l in (pivots["r1"], pivots["r2"]) if l > entry + min_sl_dist
            )
            structural_tp = resist_candidates[0] if resist_candidates else current_tp
            new_tp = max(current_tp, structural_tp) if current_tp > 0 else structural_tp

            sl_changed = new_sl > current_sl + 1e-9
            tp_changed = new_tp > current_tp + 1e-9

        else:  # SELL
            # SL: pull nearest resistance down, ratchet only
            structural_sl = pivots["nearest_resist"] + buffer
            structural_sl = max(structural_sl, live_price + min_sl_dist)
            new_sl = min(current_sl, structural_sl) if current_sl > 0 else structural_sl
            new_sl = max(new_sl, entry + buffer)   # never below entry

            # TP: push to next support, only lower
            in_profit = live_price < entry
            support_candidates = sorted(
                (l for l in (pivots["s1"], pivots["s2"]) if l < entry - min_sl_dist),
                reverse=True,
            )
            structural_tp = support_candidates[0] if support_candidates else current_tp
            new_tp = (min(current_tp, structural_tp)
                      if current_tp > 0 else structural_tp) if in_profit else current_tp

            sl_changed = (current_sl == 0.0 or new_sl < current_sl - 1e-9)
            tp_changed = (current_tp == 0.0 or new_tp < current_tp - 1e-9)

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

        if pos_type == "BUY":
            profit = price - entry          # positive = in profit
            new_sl = current_sl             # default: no change

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
                # Stage 1: break-even
                new_sl = max(current_sl, entry)
                stage = 1
            else:
                return  # still inside 1R — don't touch

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
                # Stage 1: break-even
                new_sl = min(current_sl, entry)
                stage = 1
            else:
                return

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
