"""
TradingAgents - Multi-Agents LLM Financial Trading Framework
"""

__version__ = "0.1.0"
__author__ = "TradingAgents Team"
__email__ = "yijia.xiao@cs.ucla.edu"

# Import main components
from . import agents
from . import dataflows
from . import graph
from . import default_config

__all__ = ["agents", "dataflows", "graph", "default_config"] 