"""
TradingAgents v2 - Advanced Multi-Agent LLM Financial Trading Framework

A comprehensive trading system featuring:
- Multi-timeframe technical analysis agents
- Probability-calibrated decision making
- MT5 execution integration
- Risk management and monitoring
- Backtesting capabilities
"""

__version__ = "0.2.0"
__author__ = "TradingAgents Team"
__email__ = "yijia.xiao@cs.ucla.edu"

# Core components
from . import core
from . import agents
from . import data
from . import execution
from . import risk
from . import monitoring
from . import backtesting
from . import config

# Main classes for easy access
from .core.graph import TradingGraph
from .core.agent_base import BaseAgent, AgentRegistry
from .core.types import AgentOutput, Timeframe, Direction, TradePlan
from .runner import TradingRunner

__all__ = [
    "core",
    "agents", 
    "data",
    "execution",
    "risk",
    "monitoring",
    "backtesting",
    "config",
    "TradingGraph",
    "TradingRunner",
    "BaseAgent",
    "AgentRegistry",
    "AgentOutput",
    "Timeframe",
    "Direction",
    "TradePlan",
]
