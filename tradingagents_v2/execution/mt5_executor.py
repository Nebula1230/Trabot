"""
MetaTrader 5 execution module for trade execution.

Supports three backends (tried in order):
  1. pymt5linux  — Linux-friendly, connects to MT5 running in Docker via TCP
  2. MetaTrader5 — Official package (Windows only)
  3. Simulation   — No real broker; all orders are mocked
"""

# ── Backend detection ──────────────────────────────────────────────────────
# Priority:
#  1. mt5linux  — pip install mt5linux  (Python 3.11+, connects to MT5-in-Docker via rpyc/TCP)
#     https://github.com/lucas-campagna/mt5linux  (pymt5linux is a fork that requires Python 3.13)
#  2. MetaTrader5 — official Windows-only package
#  3. Simulation  — no broker

try:
    from mt5linux import MetaTrader5 as _PyMT5LinuxClass
    _PYMT5LINUX_AVAILABLE = True
except ImportError:
    _PYMT5LINUX_AVAILABLE = False
    _PyMT5LinuxClass = None

try:
    import MetaTrader5 as _mt5_official
    _MT5_OFFICIAL_AVAILABLE = True
except ImportError:
    _MT5_OFFICIAL_AVAILABLE = False
    _mt5_official = None

MT5_AVAILABLE = _PYMT5LINUX_AVAILABLE or _MT5_OFFICIAL_AVAILABLE
# ──────────────────────────────────────────────────────────────────────────

from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import time

from ..core.types import TradePlan, Direction


class MT5Executor:
    """
    MetaTrader 5 execution handler.

    Requires MetaTrader 5 terminal running & logged in.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("MT5Executor")
        self.initialized = False
        self.simulation_mode = True   # safe default — overridden below if broker connects
        self.magic_number = self.config.get("magic_number", 424242)
        # Comment stamped on every order — visible in MT5 History/Trades tab.
        # Set to "bot/<profile>" by TradingRunner; fallback for direct construction.
        self._order_comment = self.config.get("order_comment", "bot")
        self._mt5 = None

        # Allow callers to force simulation mode (e.g. tests, dry-run)
        if self.config.get("simulation", False):
            self.logger.info("MT5Executor forced into simulation mode via config")
            self.simulation_mode = True
            self.initialized = True
        elif _PYMT5LINUX_AVAILABLE:
            host = self.config.get("mt5_host", self.config.get("host", "localhost"))
            port = int(self.config.get("mt5_port", self.config.get("port", 18812)))
            self.logger.info(f"Using mt5linux backend → {host}:{port}")
            try:
                self._mt5 = _PyMT5LinuxClass(host=host, port=port)
                self.simulation_mode = False
                self._initialize_mt5()
            except Exception as e:
                self.logger.error(
                    f"mt5linux connection to {host}:{port} failed: {e}\n"
                    "  Is the Docker container running?  docker compose up -d\n"
                    "  Is MT5 fully started inside it?  http://localhost:6082/vnc.html\n"
                    "  Falling back to SIMULATION mode."
                )
                self._mt5 = None
                self.simulation_mode = True
                self.initialized = True
        elif _MT5_OFFICIAL_AVAILABLE:
            self.logger.info("Using official MetaTrader5 package")
            self._mt5 = _mt5_official
            self.simulation_mode = False
            self._initialize_mt5()
        else:
            self.logger.warning(
                "No MT5 client available (install mt5linux for Linux/Docker). "
                "Running in simulation mode."
            )
            self.simulation_mode = True
            self.initialized = True

    def _initialize_mt5(self) -> bool:
        """Initialize MetaTrader 5 connection."""
        if self.simulation_mode:
            self.logger.info("MT5Executor in simulation mode")
            self.initialized = True
            return True
        try:
            login    = self.config.get("login",    0)
            password = self.config.get("password", "")
            server   = self.config.get("server",   "")

            init_kwargs = {}
            if login:    init_kwargs["login"]    = int(login)
            if password: init_kwargs["password"] = str(password)
            if server:   init_kwargs["server"]   = str(server)

            if not self._mt5.initialize(**init_kwargs):
                error = self._mt5.last_error()
                self.logger.error(f"MT5 initialization failed: {error}")
                return False

            account_info = self._mt5.account_info()
            if not account_info:
                self.logger.error("Not logged into MT5 account")
                return False

            self.logger.info(
                f"Connected to MT5: {account_info.login} on {account_info.server} "
                f"| balance={account_info.balance} | trade_allowed={getattr(account_info,'trade_allowed',True)}"
            )
            self.initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Error initializing MT5: {e}")
            return False

    def get_symbol_info(self, symbol: str) -> Optional[Any]:
        """Get symbol information and ensure it's visible."""
        if self.simulation_mode:
            # Return mock symbol info
            return type('MockSymbolInfo', (), {
                'visible': True,
                'name': symbol,
                'point': 0.01,
                'digits': 2
            })()

        try:
            info = self._mt5.symbol_info(symbol)
            if info is None:
                self.logger.error(f"Symbol {symbol} not found")
                return None

            if not info.visible:
                if not self._mt5.symbol_select(symbol, True):
                    self.logger.error(f"Failed to select symbol {symbol}")
                    return None
                info = self._mt5.symbol_info(symbol)

            return info

        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[Tuple[float, float]]:
        """Get current bid/ask prices for a symbol."""
        if self.simulation_mode:
            # Return mock prices
            return 100.0, 100.05

        try:
            tick = self._mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            return tick.bid, tick.ask
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None

    def place_bracket_order(self, trade_plan: TradePlan,
                           entry_price: float = None,
                           stop_loss: float = None,
                           take_profit: float = None) -> Optional[Dict[str, Any]]:
        """
        Place a bracket order (entry + stop loss + take profit).

        Args:
            trade_plan: Trade plan with execution details
            entry_price: Entry price (if None, uses market price)
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Order result dictionary or None if failed
        """
        try:
            if not self.initialized:
                self.logger.error("MT5 not initialized")
                return None

            symbol = trade_plan.symbol
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return None

            # ── Tick freshness check ────────────────────────────────────────────────────
            # Reject if the last tick is older than max_tick_age_ms ms.
            # A stale tick means the market connection is broken or the symbol
            # is outside its trading session — any order would fill at an
            # outdated price.
            max_tick_age_ms = int(self.config.get("execution", {}).get("max_tick_age_ms", 0))
            if not self.simulation_mode and max_tick_age_ms > 0:
                _tick = self._mt5.symbol_info_tick(symbol)
                if _tick is not None and hasattr(_tick, "time_msc"):
                    import time as _t
                    tick_age_ms = _t.time() * 1000 - _tick.time_msc
                    if tick_age_ms > max_tick_age_ms:
                        self.logger.warning(
                            f"{symbol}: last tick is {tick_age_ms:.0f}ms old "
                            f"(limit={max_tick_age_ms}ms) — skipping entry"
                        )
                        return None
            trade_mode = getattr(symbol_info, 'trade_mode', 4)
            if trade_mode == 0:
                self.logger.warning(
                    f"{symbol}: trading disabled on this symbol "
                    f"(trade_mode=0 — may be outside market hours or broker restriction). Skipping."
                )
                return None

            # ── Spread guard ──────────────────────────────────────────────────
            # Reject entry if spread > configured fraction of the planned stop distance.
            # Using a fraction makes this instrument-agnostic: EURUSD spread=3 pts (~0.03 pip)
            # and US30 spread=210 pts (~2.1 index pts) have very different raw values but
            # similar fractions of their respective ATR stops.
            max_spread_fraction = float(self.config.get("max_spread_fraction", 0.20))
            if not self.simulation_mode and max_spread_fraction > 0:
                tick = self._mt5.symbol_info_tick(symbol)
                if tick and symbol_info:
                    tick_size_val = getattr(symbol_info, "trade_tick_size", 0.0001)
                    spread_pts    = getattr(symbol_info, "spread", 0)
                    spread_price  = spread_pts * tick_size_val
                    is_long  = trade_plan.recipe.direction == Direction.LONG
                    entry_est = tick.ask if is_long else tick.bid
                    sl_ref    = stop_loss if stop_loss is not None else trade_plan.stop_loss
                    stop_dist = abs(entry_est - sl_ref) if sl_ref else 0.0
                    if stop_dist > 1e-10 and spread_price / stop_dist > max_spread_fraction:
                        self.logger.warning(
                            f"{symbol}: spread {spread_price:.5f} is "
                            f"{spread_price/stop_dist:.0%} of stop distance {stop_dist:.5f} "
                            f"(> {max_spread_fraction:.0%} limit) — skipping entry"
                        )
                        return None

            # Get current prices if not provided
            if entry_price is None:
                current_prices = self.get_current_price(symbol)
                if not current_prices:
                    return None
                bid, ask = current_prices
                entry_price = ask if trade_plan.recipe.direction == Direction.LONG else bid

            if stop_loss is None:
                stop_loss = trade_plan.stop_loss
            if take_profit is None:
                take_profit = trade_plan.take_profit

            if self.simulation_mode:
                # Simulate order placement
                self.logger.info(f"SIMULATION: Placing bracket order for {symbol}")
                return {
                    "order_ticket": 12345,
                    "symbol": symbol,
                    "volume": trade_plan.quantity,
                    "price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "direction": trade_plan.recipe.direction,
                    "magic": self.magic_number,
                    "simulated": True
                }

            # Clamp to broker limits (min/max lot)
            volume = trade_plan.quantity
            vol_min  = getattr(symbol_info, 'volume_min',  0.01)
            vol_max  = getattr(symbol_info, 'volume_max',  1000.0)
            vol_step = getattr(symbol_info, 'volume_step', 0.01)
            if vol_step > 0:
                volume = round(round(volume / vol_step) * vol_step, 8)
            volume = max(vol_min, min(vol_max, volume))
            self.logger.info(f"{symbol}: lots={volume} (risk=${trade_plan.risk_amount:.2f})")

            # Determine filling mode supported by broker
            filling_type = self._mt5.ORDER_FILLING_FOK
            info_flags = getattr(symbol_info, 'filling_mode', 0)
            if info_flags & 0x1:   # FILLING_FOK supported
                filling_type = self._mt5.ORDER_FILLING_FOK
            elif info_flags & 0x2: # FILLING_IOC supported
                filling_type = self._mt5.ORDER_FILLING_IOC

            # Use market order (TRADE_ACTION_DEAL) for immediate execution
            order_type = (self._mt5.ORDER_TYPE_BUY
                          if trade_plan.recipe.direction == Direction.LONG
                          else self._mt5.ORDER_TYPE_SELL)

            request = {
                "action":       self._mt5.TRADE_ACTION_DEAL,
                "symbol":       symbol,
                "volume":       volume,
                "type":         order_type,
                "price":        entry_price,
                "sl":           stop_loss,
                "tp":           take_profit,
                "deviation":    self.config.get("slippage", 10),
                "magic":        self.magic_number,
                "comment":      (
                    f"{self._order_comment}_"
                    + ("DLO" if trade_plan.recipe.direction.value == "long" else
                       "DSH" if trade_plan.recipe.direction.value == "short" else "DNE")
                ),
                "type_filling": filling_type,
            }
            self.logger.debug(f"order_send request: {request}")

            result = self._mt5.order_send(request)
            if result is None:
                err = self._mt5.last_error()
                self.logger.error(f"order_send returned None — last_error: {err}")
                return None
            if result.retcode != self._mt5.TRADE_RETCODE_DONE:
                self.logger.error(
                    f"Order rejected: retcode={result.retcode} comment={result.comment} "
                    f"vol={volume} price={entry_price} sl={stop_loss} tp={take_profit}"
                )
                return None

            self.logger.info(f"Market order filled: ticket={result.order} vol={volume}")

            return {
                "order_ticket": result.order,
                "symbol": symbol,
                "volume": trade_plan.quantity,
                "price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "direction": trade_plan.recipe.direction,
                "magic": self.magic_number
            }

        except Exception as e:
            self.logger.error(f"Error placing bracket order: {e}")
            return None

    def modify_stop_loss(self, position_ticket: int, new_stop_loss: float) -> bool:
        """Modify stop loss for an open position."""
        if self.simulation_mode:
            self.logger.info(f"SIMULATION: Modified SL for position {position_ticket}: {new_stop_loss}")
            return True

        try:
            positions = self._mt5.positions_get(ticket=position_ticket)
            if not positions:
                self.logger.error(f"Position {position_ticket} not found")
                return False

            pos = positions[0]
            request = {
                "action":   self._mt5.TRADE_ACTION_SLTP,
                "symbol":   pos.symbol,
                "position": position_ticket,
                "sl":       new_stop_loss,
                "tp":       pos.tp,
                "magic":    pos.magic,
            }
            result = self._mt5.order_send(request)
            if result is None:
                self.logger.error(f"SL modification failed: no result")
                return False
            # 10025 = TRADE_RETCODE_NO_CHANGES: new value rounds to the same tick
            # as the current SL — position is already where we want it.
            if result.retcode == 10025:
                self.logger.debug(
                    f"SL already at tick for position {position_ticket} — no MT5 change needed"
                )
                return True
            if result.retcode != self._mt5.TRADE_RETCODE_DONE:
                self.logger.warning(f"SL modification failed: retcode={result.retcode} — original SL still active")
                return False

            self.logger.info(f"SL modified for position {position_ticket}: {new_stop_loss:.5f}")
            return True

        except Exception as e:
            self.logger.error(f"Error modifying SL: {e}")
            return False

    def modify_take_profit(self, position_ticket: int, new_take_profit: float) -> bool:
        """Modify take profit for an open position (SL is preserved unchanged)."""
        if self.simulation_mode:
            self.logger.info(
                f"SIMULATION: Modified TP for position {position_ticket}: {new_take_profit:.5f}"
            )
            return True

        try:
            positions = self._mt5.positions_get(ticket=position_ticket)
            if not positions:
                self.logger.error(f"Position {position_ticket} not found")
                return False

            pos = positions[0]
            request = {
                "action":   self._mt5.TRADE_ACTION_SLTP,
                "symbol":   pos.symbol,
                "position": position_ticket,
                "sl":       pos.sl,          # unchanged
                "tp":       new_take_profit,
                "magic":    pos.magic,
            }
            result = self._mt5.order_send(request)
            if result is None:
                self.logger.error(f"TP modification failed: no result")
                return False
            # 10025 = TRADE_RETCODE_NO_CHANGES: new TP rounds to the same tick.
            if result.retcode == 10025:
                self.logger.debug(
                    f"TP already at tick for position {position_ticket} — no MT5 change needed"
                )
                return True
            if result.retcode != self._mt5.TRADE_RETCODE_DONE:
                self.logger.warning(f"TP modification failed: retcode={result.retcode} — original TP still active")
                return False

            self.logger.info(f"TP modified for position {position_ticket}: {new_take_profit:.5f}")
            return True

        except Exception as e:
            self.logger.error(f"Error modifying TP: {e}")
            return False

    def modify_sl_tp(self, position_ticket: int,
                     new_sl: float, new_tp: float) -> bool:
        """Modify both stop-loss and take-profit atomically in a single MT5 request."""
        if self.simulation_mode:
            self.logger.info(
                f"SIMULATION: Modified SL/TP for position {position_ticket}: "
                f"SL={new_sl:.5f}  TP={new_tp:.5f}"
            )
            return True

        try:
            positions = self._mt5.positions_get(ticket=position_ticket)
            if not positions:
                self.logger.error(f"Position {position_ticket} not found")
                return False

            pos = positions[0]
            request = {
                "action":   self._mt5.TRADE_ACTION_SLTP,
                "symbol":   pos.symbol,
                "position": position_ticket,
                "sl":       new_sl,
                "tp":       new_tp,
                "magic":    pos.magic,
            }
            result = self._mt5.order_send(request)
            if result is None:
                self.logger.error(f"SL/TP modification failed: no result")
                return False
            # 10025 = TRADE_RETCODE_NO_CHANGES: both values already at their
            # tick-rounded positions — not an error, just a no-op.
            if result.retcode == 10025:
                self.logger.debug(
                    f"SL/TP already at tick for position {position_ticket} — no MT5 change needed"
                )
                return True
            if result.retcode != self._mt5.TRADE_RETCODE_DONE:
                self.logger.warning(f"SL/TP modification failed: retcode={result.retcode} — original SL/TP still active")
                return False

            self.logger.info(
                f"SL/TP modified for position {position_ticket}: "
                f"SL={new_sl:.5f}  TP={new_tp:.5f}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error modifying SL/TP: {e}")
            return False

    def close_position(self, position_ticket: int) -> bool:
        """Close an open position."""
        if self.simulation_mode:
            self.logger.info(f"SIMULATION: Closed position {position_ticket}")
            return True

        try:
            position = self._mt5.positions_get(ticket=position_ticket)
            if not position:
                self.logger.error(f"Position {position_ticket} not found")
                return False

            pos = position[0]
            if pos.type == self._mt5.POSITION_TYPE_BUY:
                close_type = self._mt5.ORDER_TYPE_SELL
                price = self._mt5.symbol_info_tick(pos.symbol).bid
            else:
                close_type = self._mt5.ORDER_TYPE_BUY
                price = self._mt5.symbol_info_tick(pos.symbol).ask

            # Auto-detect supported filling mode (same logic as entry orders)
            sym_info = self._mt5.symbol_info(pos.symbol)
            fill_mode = self._mt5.ORDER_FILLING_RETURN
            if sym_info:
                fm = getattr(sym_info, "filling_mode", 0)
                if fm & 0x1:
                    fill_mode = self._mt5.ORDER_FILLING_FOK
                elif fm & 0x2:
                    fill_mode = self._mt5.ORDER_FILLING_IOC

            request = {
                "action":       self._mt5.TRADE_ACTION_DEAL,
                "symbol":       pos.symbol,
                "volume":       pos.volume,
                "type":         close_type,
                "position":     position_ticket,
                "price":        price,
                "deviation":    self.config.get("slippage", 10),
                "magic":        self.magic_number,
                "comment":      f"{self._order_comment}_WFX",
                "type_time":    self._mt5.ORDER_TIME_GTC,
                "type_filling": fill_mode,
            }
            result = self._mt5.order_send(request)
            if result is None or result.retcode != self._mt5.TRADE_RETCODE_DONE:
                retcode = result.retcode if result else "None (connection lost)"
                self.logger.error(f"Position close failed: {retcode}")
                return False

            self.logger.info(f"Position {position_ticket} closed")
            return True

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False

    def close_partial_position(self, position_ticket: int, fraction: float) -> bool:
        """
        Close a fraction of an open position in-place.

        Parameters
        ----------
        position_ticket : int
            MT5 ticket of the position to partially close.
        fraction : float
            Fraction of current volume to close (0 < fraction < 1).
            Rounded down to the symbol's volume_step; skipped if the
            resulting close or remaining volume would be below volume_min.

        Returns True if the partial close was sent and accepted by the broker.
        """
        if self.simulation_mode:
            self.logger.info(
                f"SIMULATION: Partial close {fraction:.0%} of #{position_ticket}"
            )
            return True

        try:
            position = self._mt5.positions_get(ticket=position_ticket)
            if not position:
                self.logger.error(f"Partial close: position #{position_ticket} not found")
                return False

            pos      = position[0]
            sym_info = self._mt5.symbol_info(pos.symbol)
            vol_min  = getattr(sym_info, "volume_min",  0.01)
            vol_step = getattr(sym_info, "volume_step", 0.01)

            # Calculate close volume, snapped to vol_step grid
            close_vol = pos.volume * fraction
            close_vol = round(round(close_vol / vol_step) * vol_step, 8)
            remaining = round(pos.volume - close_vol, 8)

            if close_vol < vol_min or remaining < vol_min:
                self.logger.warning(
                    f"#{position_ticket}: partial close {fraction:.0%} skipped "
                    f"(close={close_vol:.3f} remaining={remaining:.3f} "
                    f"vol_min={vol_min:.3f})"
                )
                return False

            if pos.type == self._mt5.POSITION_TYPE_BUY:
                close_type = self._mt5.ORDER_TYPE_SELL
                price      = self._mt5.symbol_info_tick(pos.symbol).bid
            else:
                close_type = self._mt5.ORDER_TYPE_BUY
                price      = self._mt5.symbol_info_tick(pos.symbol).ask

            fill_mode = self._mt5.ORDER_FILLING_RETURN
            if sym_info:
                fm = getattr(sym_info, "filling_mode", 0)
                if fm & 0x1:
                    fill_mode = self._mt5.ORDER_FILLING_FOK
                elif fm & 0x2:
                    fill_mode = self._mt5.ORDER_FILLING_IOC

            request = {
                "action":       self._mt5.TRADE_ACTION_DEAL,
                "symbol":       pos.symbol,
                "volume":       close_vol,
                "type":         close_type,
                "position":     position_ticket,
                "price":        price,
                "deviation":    self.config.get("slippage", 10),
                "magic":        self.magic_number,
                "comment":      f"{self._order_comment}_PTP",
                "type_time":    self._mt5.ORDER_TIME_GTC,
                "type_filling": fill_mode,
            }
            result = self._mt5.order_send(request)
            if result is None or result.retcode != self._mt5.TRADE_RETCODE_DONE:
                retcode = result.retcode if result else "None (connection lost)"
                self.logger.error(f"Partial close #{position_ticket} failed: {retcode}")
                return False

            self.logger.info(
                f"Partial close #{position_ticket}: {close_vol:.3f} lots "
                f"({fraction:.0%}) locked in — {remaining:.3f} lots remaining"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error in partial close #{position_ticket}: {e}")
            return False

    def get_open_positions(self) -> list:
        """Get all open positions."""
        if self.simulation_mode:
            # Return mock positions
            return [
                {
                    "ticket": 12345,
                    "symbol": "NASDAQ:NVDA",
                    "type": "BUY",
                    "volume": 100,
                    "price_open": 150.0,
                    "price_current": 152.0,
                    "sl": 148.0,
                    "tp": 155.0,
                    "profit": 200.0,
                    "magic": self.magic_number
                }
            ]

        try:
            positions = self._mt5.positions_get()
            if positions is None:
                return []
            return [
                {
                    "ticket":        pos.ticket,
                    "symbol":        pos.symbol,
                    "type":          "BUY" if pos.type == self._mt5.POSITION_TYPE_BUY else "SELL",
                    "volume":        pos.volume,
                    "price_open":    pos.price_open,
                    "price_current": pos.price_current,
                    "sl":            pos.sl,
                    "tp":            pos.tp,
                    "profit":        pos.profit,
                    "magic":         pos.magic,
                    "open_time":     pos.time,   # UNIX timestamp of position open
                }
                for pos in positions
                if pos.magic == self.magic_number
            ]

        except Exception as e:
            self.logger.error(f"Error getting open positions: {e}")
            return []

    def get_closed_trades(self, days: int = 1) -> list:
        """
        Return closed deals for the last *days* days that belong to this bot
        (matched by magic number).
        """
        if self.simulation_mode:
            return []   # simulation never actually closes trades

        if not self.initialized:
            return []

        try:
            from datetime import timedelta
            date_from = datetime.now() - timedelta(days=days)
            deals = self._mt5.history_deals_get(date_from, datetime.now())
            if deals is None:
                return []
            return [
                {
                    "ticket":     d.ticket,
                    "symbol":     d.symbol,
                    "type":       "BUY" if d.type == self._mt5.DEAL_TYPE_BUY else "SELL",
                    "volume":     d.volume,
                    "price":      d.price,
                    "profit":     d.profit,
                    "commission": d.commission,
                    "swap":       d.swap,
                    "time":       datetime.fromtimestamp(d.time).isoformat(),
                    "magic":      d.magic,
                }
                for d in deals
                if d.magic == self.magic_number and d.profit != 0
            ]
        except Exception as e:
            self.logger.error(f"Error getting closed trades: {e}")
            return []

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information."""
        if self.simulation_mode:
            # Return mock account info
            return {
                "login": 12345,
                "server": "Simulation",
                "balance": 100000.0,
                "equity": 100200.0,
                "margin": 20000.0,
                "free_margin": 80200.0,
                "profit": 200.0,
                "leverage": 1.0
            }

        try:
            account = self._mt5.account_info()
            if not account:
                return None
            return {
                "login":       account.login,
                "server":      account.server,
                "balance":     account.balance,
                "equity":      account.equity,
                "margin":      account.margin,
                "free_margin": account.margin_free,
                "profit":      account.profit,
                "leverage":    account.leverage,
            }
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None

    def get_tf_map(self) -> Dict[str, Any]:
        """Return a timeframe-string → MT5-constant mapping for use by DataLoader."""
        if self.simulation_mode or self._mt5 is None:
            return {}
        try:
            return {
                "1m":  self._mt5.TIMEFRAME_M1,
                "5m":  self._mt5.TIMEFRAME_M5,
                "15m": self._mt5.TIMEFRAME_M15,
                "30m": self._mt5.TIMEFRAME_M30,
                "1H":  self._mt5.TIMEFRAME_H1,
                "4H":  self._mt5.TIMEFRAME_H4,
                "1D":  self._mt5.TIMEFRAME_D1,
                "1W":  self._mt5.TIMEFRAME_W1,
            }
        except AttributeError:
            self.logger.warning("Could not build TF map from MT5 object")
            return {}

    def copy_rates(self, symbol: str, timeframe: str, n: int):
        """Fetch OHLCV bars from MT5 for *symbol* on *timeframe*."""
        if self.simulation_mode or self._mt5 is None:
            return None
        tf_map = self.get_tf_map()
        tf = tf_map.get(timeframe)
        if tf is None:
            self.logger.error(f"Unknown timeframe: {timeframe}")
            return None
        # Ensure symbol is subscribed so MT5 has/downloads its history
        try:
            if not self._mt5.symbol_info(symbol):
                self._mt5.symbol_select(symbol, True)
        except Exception:
            pass
        return self._mt5.copy_rates_from_pos(symbol, tf, 0, n)

    def shutdown(self):
        """Shutdown MT5 connection."""
        if self.initialized and not self.simulation_mode and self._mt5 is not None:
            self._mt5.shutdown()
            self.initialized = False
            self.logger.info("MT5 connection closed")
        elif self.simulation_mode:
            self.logger.info("MT5Executor simulation mode shutdown")

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown() 