"""
Base agent class for all trading agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel
import asyncio
import logging

from .types import AgentOutput, TechnicalFeatures, Timeframe


class BaseAgent(ABC, BaseModel):
    """Base class for all trading agents."""
    
    name: str
    timeframe: Timeframe
    enabled: bool = True
    weight: float = 1.0

    model_config = {"arbitrary_types_allowed": True}

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(f"agent.{self.name}")
    
    @abstractmethod
    async def analyze(self, features: TechnicalFeatures, context: Dict[str, Any] = None) -> AgentOutput:
        """
        Analyze market data and return agent output.
        
        Args:
            features: Technical analysis features
            context: Additional context (regime, market conditions, etc.)
            
        Returns:
            AgentOutput with directional score, confidence, and rationale
        """
        pass
    
    def validate_features(self, features: TechnicalFeatures) -> bool:
        """Validate that required features are available."""
        required_fields = self.get_required_features()
        for field in required_fields:
            if not hasattr(features, field) or getattr(features, field) is None:
                self.logger.warning(f"Missing required feature: {field}")
                return False
        return True
    
    def get_required_features(self) -> list:
        """Return list of required feature names."""
        return []
    
    def calculate_confidence(self, features: TechnicalFeatures, context: Dict[str, Any] = None) -> float:
        """
        Calculate confidence score based on feature quality and historical performance.
        Override in subclasses for custom confidence logic.
        """
        # Default confidence calculation
        base_confidence = 0.5
        
        # Adjust based on feature quality
        if hasattr(features, 'atr_14') and features.atr_14 > 0:
            # Higher confidence in more volatile conditions
            volatility_factor = min(features.atr_14 / 100, 1.0)
            base_confidence += volatility_factor * 0.2
        
        # Adjust based on trend strength
        if hasattr(features, 'adx_14'):
            trend_factor = min(features.adx_14 / 50, 1.0)
            base_confidence += trend_factor * 0.3
        
        return min(max(base_confidence, 0.0), 1.0)
    
    async def run(self, features: TechnicalFeatures, context: Dict[str, Any] = None) -> Optional[AgentOutput]:
        """
        Main entry point for running the agent.
        
        Args:
            features: Technical analysis features
            context: Additional context
            
        Returns:
            AgentOutput if successful, None if failed
        """
        try:
            if not self.enabled:
                return None
                
            if not self.validate_features(features):
                self.logger.error(f"Feature validation failed for {self.name}")
                return None
            
            output = await self.analyze(features, context)

            # NOTE: agent weight is intentionally NOT applied here.
            # It is applied exactly once in TradingGraph._fuse_timeframes via
            # _agent_weights so that the raw dir_score stays in [-1, 1] and
            # the weight does not get squared during fusion.
            self.logger.debug(f"{self.name} output: {output}")
            return output
            
        except Exception as e:
            self.logger.error(f"Error in {self.name}: {e}")
            return None
    
    def __str__(self):
        return f"{self.name}({self.timeframe})"
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', timeframe={self.timeframe})"


class AgentRegistry:
    """Registry for managing all agents."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
    
    def register(self, agent: BaseAgent):
        """Register an agent."""
        self.agents[agent.name] = agent
        logging.info(f"Registered agent: {agent.name}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name."""
        return self.agents.get(name)
    
    def get_agents_by_timeframe(self, timeframe: Timeframe) -> list:
        """Get all agents for a specific timeframe."""
        return [agent for agent in self.agents.values() if agent.timeframe == timeframe]
    
    def get_all_agents(self) -> list:
        """Get all registered agents."""
        return list(self.agents.values())
    
    def enable_agent(self, name: str):
        """Enable an agent."""
        if name in self.agents:
            self.agents[name].enabled = True
            logging.info(f"Enabled agent: {name}")
    
    def disable_agent(self, name: str):
        """Disable an agent."""
        if name in self.agents:
            self.agents[name].enabled = False
            logging.info(f"Disabled agent: {name}")
    
    def set_agent_weight(self, name: str, weight: float):
        """Set agent weight."""
        if name in self.agents:
            self.agents[name].weight = weight
            logging.info(f"Set weight for {name}: {weight}") 