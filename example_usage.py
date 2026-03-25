#!/usr/bin/env python3
"""
Example usage of TradingAgents v2 system.

This script demonstrates how to:
1. Set up the agent registry
2. Configure the trading graph
3. Run analysis on a symbol
4. Handle the results
"""

import asyncio
import logging
from typing import Dict, Any

# Import TradingAgents v2 components
from tradingagents_v2 import TradingGraph, AgentRegistry
from tradingagents_v2.agents import (
    RegimeAgent, TrendAgent, MomentumAgent, MeanReversionAgent,
    VolatilityAgent, BreadthAgent, PatternAgent
)
from tradingagents_v2.core.types import (
    TechnicalFeatures, PortfolioState, RiskLimits, Timeframe
)
from tradingagents_v2.config import TradingConfig


def setup_logging():
    """Configure logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('tradingagents_example.log')
        ]
    )


def create_sample_features() -> TechnicalFeatures:
    """Create sample technical features for demonstration."""
    return TechnicalFeatures(
        # Structure
        swing_highs=[150.0, 155.0, 160.0],
        swing_lows=[140.0, 145.0, 148.0],
        hh_hl_count=2,
        lh_ll_count=1,
        last_break="high",
        
        # Moving Averages
        ema20=152.0,
        ema50=150.0,
        ema200=145.0,
        ema20_slope=0.02,
        ema50_slope=0.01,
        ema200_slope=0.005,
        
        # Momentum
        rsi_14=65.0,
        rsi_4=70.0,
        macd_hist=0.5,
        macd_hist_delta=0.1,
        roc_10=0.03,
        bb_percent_b=0.7,
        
        # Volatility
        atr_14=2.5,
        atr_5=2.0,
        realized_vol=0.025,
        bb_width=0.08,
        keltner_width=0.06,
        
        # Regime
        adx_14=28.0,
        rvi=0.3,
        hurst_exponent=0.6,
        atr_price_ratio=0.016,
        vwap_distance=0.02,
        
        # Breadth
        index_trend=0.3,
        advance_decline=0.6,
        above_50ema_pct=0.65,
        above_200ema_pct=0.55,
        sector_direction=0.4
    )


def create_sample_portfolio() -> PortfolioState:
    """Create sample portfolio state for demonstration."""
    return PortfolioState(
        equity=100000.0,
        margin_used=20000.0,
        free_margin=80000.0,
        daily_pnl=1500.0,
        daily_drawdown=-0.5,
        open_positions=["NASDAQ:AAPL"],
        max_daily_drawdown=-2.0,
        leverage_used=1.2
    )


def create_sample_config() -> TradingConfig:
    """Create sample configuration for demonstration."""
    return TradingConfig(
        symbols=["NASDAQ:NVDA", "NASDAQ:MSFT"],
        risk=RiskLimits(
            base_risk_pct=0.25,
            max_daily_drawdown_pct=2.0,
            max_concurrent_trades=3
        ),
        log_level="INFO"
    )


async def run_trading_analysis():
    """Run the complete trading analysis workflow."""
    
    # Setup
    setup_logging()
    logger = logging.getLogger("Example")
    
    logger.info("🚀 Starting TradingAgents v2 Example")
    
    # Create configuration
    config = create_sample_config()
    logger.info(f"Configuration loaded: {config.symbols}")
    
    # Create agent registry
    registry = AgentRegistry()
    
    # Register all agents
    agents = [
        RegimeAgent(),
        TrendAgent(),
        MomentumAgent(),
        MeanReversionAgent(),
        VolatilityAgent(),
        BreadthAgent(),
        PatternAgent()
    ]
    
    for agent in agents:
        registry.register(agent)
        logger.info(f"Registered agent: {agent.name}")
    
    # Create trading graph
    graph = TradingGraph(registry, config)
    logger.info("Trading graph created")
    
    # Create sample data
    features = create_sample_features()
    portfolio = create_sample_portfolio()
    risk_limits = config.risk
    
    logger.info("Sample data created")
    
    # Run analysis
    symbol = "NASDAQ:NVDA"
    logger.info(f"Running analysis for {symbol}")
    
    try:
        result = await graph.run(symbol, features, portfolio, risk_limits)
        
        # Display results
        logger.info("📊 Analysis Results:")
        logger.info(f"Symbol: {result.symbol}")
        logger.info(f"Decision: {result.decision}")
        
        if result.agent_outputs:
            logger.info(f"Agent outputs: {len(result.agent_outputs)} agents completed")
            for agent_name, output in result.agent_outputs.items():
                logger.info(f"  {agent_name}: {output.dir_score:.3f} (conf: {output.conf:.3f})")
        
        if result.timeframe_fusion:
            fusion = result.timeframe_fusion
            logger.info(f"Timeframe fusion:")
            logger.info(f"  Long: {fusion.dir_long:.3f} (conf: {fusion.conf_long:.3f})")
            logger.info(f"  Mid: {fusion.dir_mid:.3f} (conf: {fusion.conf_mid:.3f})")
            logger.info(f"  Short: {fusion.dir_short:.3f} (conf: {fusion.conf_short:.3f})")
            logger.info(f"  Regime trendiness: {fusion.regime_trendiness:.3f}")
            logger.info(f"  Breadth score: {fusion.breadth_score:.3f}")
        
        if result.trade_recipe:
            recipe = result.trade_recipe
            logger.info(f"Trade recipe generated:")
            logger.info(f"  Name: {recipe.name}")
            logger.info(f"  Direction: {recipe.direction}")
            logger.info(f"  Win probability: {recipe.win_probability:.3f}")
            logger.info(f"  Expected value: {recipe.expected_value:.3f}")
            logger.info(f"  Risk/reward: {recipe.risk_reward_ratio:.2f}")
        
        if result.trade_plan:
            plan = result.trade_plan
            logger.info(f"Trade plan created:")
            logger.info(f"  Entry price: ${plan.entry_price:.2f}")
            logger.info(f"  Stop loss: ${plan.stop_loss:.2f}")
            logger.info(f"  Take profit: ${plan.take_profit:.2f}")
            logger.info(f"  Quantity: {plan.quantity:.2f}")
            logger.info(f"  Risk amount: ${plan.risk_amount:.2f}")
        
        if result.errors:
            logger.warning(f"Errors encountered: {len(result.errors)}")
            for error in result.errors:
                logger.warning(f"  {error}")
        
        if result.metadata:
            logger.info(f"Metadata: {result.metadata}")
        
        # Summary
        if result.decision == "continue" and result.trade_plan:
            logger.info("✅ Trade approved and plan created!")
        elif result.decision == "stop":
            logger.info("⛔ Trade rejected - conditions not met")
        else:
            logger.info("❓ Analysis completed with no clear decision")
            
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        raise


async def run_multiple_symbols():
    """Run analysis on multiple symbols."""
    logger = logging.getLogger("MultiSymbol")
    
    # This would be the same setup as above
    config = create_sample_config()
    registry = AgentRegistry()
    
    # Register agents
    agents = [RegimeAgent(), TrendAgent(), MomentumAgent()]
    for agent in agents:
        registry.register(agent)
    
    graph = TradingGraph(registry, config)
    
    # Sample symbols
    symbols = ["NASDAQ:NVDA", "NASDAQ:MSFT", "NASDAQ:AAPL"]
    
    for symbol in symbols:
        logger.info(f"Analyzing {symbol}...")
        try:
            features = create_sample_features()  # In real usage, get actual data
            portfolio = create_sample_portfolio()
            
            result = await graph.run(symbol, features, portfolio, config.risk)
            
            # Quick summary
            if result.trade_plan:
                logger.info(f"  {symbol}: ✅ Trade plan created")
            else:
                logger.info(f"  {symbol}: ⛔ No trade plan")
                
        except Exception as e:
            logger.error(f"  {symbol}: Error - {e}")


def main():
    """Main function to run the example."""
    print("TradingAgents v2 Example")
    print("=" * 50)
    
    # Run single symbol analysis
    print("\n1. Single Symbol Analysis")
    print("-" * 30)
    asyncio.run(run_trading_analysis())
    
    # Run multiple symbols
    print("\n2. Multiple Symbol Analysis")
    print("-" * 30)
    asyncio.run(run_multiple_symbols())
    
    print("\nExample completed!")


if __name__ == "__main__":
    main() 