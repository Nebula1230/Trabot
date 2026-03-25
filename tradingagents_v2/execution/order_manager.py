"""
Order management for trade execution.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

from ..core.types import TradePlan, Direction


class OrderManager:
    """
    Manages order lifecycle and tracking.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("OrderManager")
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []

    def create_order(self, trade_plan: TradePlan, order_type: str = "market",
                     mt5_ticket: Optional[int] = None) -> Dict[str, Any]:
        """
        Create a new order from a trade plan.

        Args:
            trade_plan: Trade plan with execution details
            order_type: Type of order (market, limit, stop)
            mt5_ticket: The MT5 order ticket returned by the broker, used to
                        correlate this shadow record with the live position.

        Returns:
            Order details dictionary
        """
        order_id = f"order_{len(self.active_orders) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        order = {
            "order_id": order_id,
            "mt5_ticket": mt5_ticket,        # live broker ticket (None in simulation)
            "symbol": trade_plan.symbol,
            "direction": trade_plan.recipe.direction,
            "quantity": trade_plan.quantity,
            "entry_price": trade_plan.entry_price,
            "stop_loss": trade_plan.stop_loss,
            "take_profit": trade_plan.take_profit,
            "order_type": order_type,
            "status": "pending",
            "created_at": datetime.now(),
            "trade_plan": trade_plan,
        }

        self.active_orders[order_id] = order
        self.logger.info(f"Created order {order_id} for {trade_plan.symbol}")

        return order

    def update_order_status(self, order_id: str, status: str, **kwargs) -> bool:
        """
        Update the status of an order.

        Args:
            order_id: Order identifier
            status: New status
            **kwargs: Additional fields to update

        Returns:
            True if successful, False otherwise
        """
        if order_id not in self.active_orders:
            self.logger.error(f"Order {order_id} not found")
            return False

        order = self.active_orders[order_id]
        order["status"] = status
        order["updated_at"] = datetime.now()

        # Update additional fields
        for key, value in kwargs.items():
            order[key] = value

        self.logger.info(f"Updated order {order_id} status to {status}")

        # Move to history if order is completed
        if status in ["filled", "cancelled", "rejected"]:
            self._move_to_history(order_id)

        return True

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order details by ID.

        Args:
            order_id: Order identifier

        Returns:
            Order details or None if not found
        """
        return self.active_orders.get(order_id)

    def get_active_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Get all active orders, optionally filtered by symbol.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of active orders
        """
        if symbol:
            return [order for order in self.active_orders.values() if order["symbol"] == symbol]
        return list(self.active_orders.values())

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.

        Args:
            order_id: Order identifier

        Returns:
            True if successful, False otherwise
        """
        if order_id not in self.active_orders:
            self.logger.error(f"Order {order_id} not found")
            return False

        order = self.active_orders[order_id]
        if order["status"] not in ["pending", "submitted"]:
            self.logger.warning(f"Cannot cancel order {order_id} with status {order['status']}")
            return False

        order["status"] = "cancelled"
        order["cancelled_at"] = datetime.now()
        self.logger.info(f"Cancelled order {order_id}")

        self._move_to_history(order_id)
        return True

    def _move_to_history(self, order_id: str):
        """Move completed order to history."""
        if order_id in self.active_orders:
            order = self.active_orders.pop(order_id)
            self.order_history.append(order)

    def get_order_history(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get order history, optionally filtered by symbol.

        Args:
            symbol: Optional symbol filter
            limit: Maximum number of orders to return

        Returns:
            List of historical orders
        """
        history = self.order_history
        if symbol:
            history = [order for order in history if order["symbol"] == symbol]
        
        # Sort by creation time (newest first) and limit results
        history.sort(key=lambda x: x["created_at"], reverse=True)
        return history[:limit]

    def get_order_statistics(self, symbol: str = None) -> Dict[str, Any]:
        """
        Get statistics about orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            Dictionary with order statistics
        """
        all_orders = self.get_active_orders(symbol) + self.get_order_history(symbol)
        
        if not all_orders:
            return {
                "total_orders": 0,
                "filled_orders": 0,
                "cancelled_orders": 0,
                "rejected_orders": 0,
                "success_rate": 0.0
            }

        total = len(all_orders)
        filled = len([o for o in all_orders if o["status"] == "filled"])
        cancelled = len([o for o in all_orders if o["status"] == "cancelled"])
        rejected = len([o for o in all_orders if o["status"] == "rejected"])

        return {
            "total_orders": total,
            "filled_orders": filled,
            "cancelled_orders": cancelled,
            "rejected_orders": rejected,
            "success_rate": filled / total if total > 0 else 0.0
        }

    def cleanup_old_orders(self, days: int = 30):
        """
        Clean up old orders from history.

        Args:
            days: Number of days to keep in history
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        self.order_history = [
            order for order in self.order_history 
            if order["created_at"] > cutoff_date
        ]
        self.logger.info(f"Cleaned up orders older than {days} days")
