<p align="center">
  <img src="assets/TauricResearch.png" style="width: 60%; height: auto;">
</p>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2412.20138" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.20138-B31B1B?logo=arxiv"/></a>
  <a href="https://discord.com/invite/hk9PGKShPK" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-TradingResearch-7289da?logo=discord&logoColor=white&color=7289da"/></a>
  <a href="./assets/wechat.png" target="_blank"><img alt="WeChat" src="https://img.shields.io/badge/WeChat-TauricResearch-brightgreen?logo=wechat&logoColor=white"/></a>
  <a href="https://x.com/TauricResearch" target="_blank"><img alt="X Follow" src="https://img.shields.io/badge/X-TauricResearch-white?logo=x&logoColor=white"/></a>
  <br>
  <a href="https://github.com/TauricResearch/" target="_blank"><img alt="Community" src="https://img.shields.io/badge/Join_GitHub_Community-TauricResearch-14C290?logo=discourse"/></a>
</div>

<div align="center">
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=de">Deutsch</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=es">Español</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=fr">français</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ja">日本語</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ko">한국어</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=pt">Português</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ru">Русский</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=zh">中文</a>
</div>

---

# TradingAgents v2 - Advanced Multi-Agent LLM Financial Trading Framework

A comprehensive, technically-driven multi-agent trading system designed for price action + technical analysis, aligned across long/mid/short horizons with probability estimates, tight entries, and MT5 execution.

## 🚀 System Overview

### Architecture
```
Data Layer → Signal Agents → Timeframe Fusion → Decision Engine → Risk Manager → MT5 Executor
    ↓              ↓              ↓              ↓              ↓           ↓
Market Data   Parallel Agents  Multi-TF      Trade Recipe   Portfolio   Order Mgmt
+ Features    + Indicators    Alignment     + Probability   Limits      + Monitoring
```

### Core Components

1. **Data Layer**: Market data loader (multi-timeframe bars), features/indicators, index & breadth feeds
2. **Signal Agents**: 7 specialized agents running in parallel
3. **Timeframe Fusion**: Combines signals across Long (W/D), Mid (4H/1H), Short (15m/5m)
4. **Decision Engine**: Gates on alignment, sizes risk, proposes entry/stop/targets
5. **Risk Manager**: Portfolio & leverage limits, news blackout, slippage cushions
6. **MT5 Executor**: Orders, bracket management, trailing, partials
7. **Monitor/Logger**: PnL, slippage, rule breaks, attribution

## 🧠 Signal Agents

### A) Regime Agent
- **Goal**: Identify trending vs choppy + volatility state
- **Features**: ADX(14), RVI, Hurst exponent, ATR/price, realized vol, VWAP distance
- **Output**: trendiness (0..1), vol_state (low/normal/high)

### B) Trend Agent (Structure-First)
- **Goal**: Direction via market structure + filtered MAs
- **Features**: HH/HL or LH/LL detection, EMA ribbon (20/50/200), slope & stack
- **Signal**: Directional score from structure + MA alignment

### C) Momentum Agent
- **Goal**: Confirm impulse in direction of trend; avoid exhausted moves
- **Features**: RSI(14/4) regime logic, MACD histogram delta, ROC(10), %B/Bollinger
- **Signal**: + if impulse with room to run, − if late stage

### D) Mean-Reversion Guard
- **Goal**: Block chasing extended moves
- **Features**: Z-score vs 20-day mean, distance from 20EMA/VWAP, Keltner position
- **Signal**: Negative when extension > k·ATR

### E) Volatility Agent
- **Goal**: Make entries that respect risk
- **Features**: ATR(14) & intraday realized vol, spread & book depth
- **Signal**: Minimum stop distance, expected slippage risk score

### F) Breadth/Index Agent
- **Goal**: Align with market/sector wind
- **Features**: Index trend, advance/decline, % above 50/200EMA, sector ETF direction
- **Signal**: Penalize longs if index down across long+mid TFs

### G) Pattern Agent
- **Goal**: Price patterns that improve timing
- **Features**: Breakout bases, flags, pullback to 20EMA, failed-break traps
- **Signal**: Binary/graded pattern quality + entry recipe

## 🔄 Alignment Logic

### Per-Timeframe Aggregation
```python
dir_t = weighted_avg(agent.dir_score * agent.conf)
conf_t = mean(agent.conf) * regime.trendiness
```

### Directional Agreement Gate
- **Go-Long**: `dir_long > +0.4`, `dir_mid > +0.3`, `dir_short > +0.2`, `breadth ≥ 0`
- **Go-Short**: Analogous conditions
- **Blocked by**: Mean-Reversion Guard if `dir_score < -0.5`

### Probability Estimation
- Rolling performance tables per entry recipe
- Beta-Binomial priors: `p ~ Beta(a,b)` updated with wins/losses
- Final `win_prob = E[p] * conf_factors * regime * breadth`
- Trade only if `win_prob ≥ p_min` and expected value positive

## 📊 Entry, Stop, Targets

### Entry Strategy
- **Trigger**: Break of pullback high/low, inside-bar break + buffer, retest & go
- **Micro-structure**: Focus on 5-15m timeframe for precision

### Position Sizing
```python
risk_per_trade = min(base_risk_pct * equity, daily_var_cap_remaining, volatility_cap)
qty = floor(risk_per_trade / stop_distance_price)
```

### Profit Taking
- **Scale-out**: 50% at +1R, trail rest with ATR(14) or last swing
- **Two-target**: T1 at 1R, T2 at 2-3R; move stop to BE at 1R
- **Time stop**: Exit if no follow-through after N bars

### Leverage Rules
- Hard caps on per-symbol exposure, sector exposure, aggregate margin
- Lockout after max_daily_drawdown hit

## 🏗️ Installation & Setup

### Prerequisites
- Python 3.10+
- MetaTrader 5 terminal running & logged in
- Redis (optional, for caching)

### Install Dependencies
```bash
# Create virtual environment
python -m venv tradingagents_env
source tradingagents_env/bin/activate  # Linux/Mac
# or
tradingagents_env\Scripts\activate     # Windows

# Install package
pip install -e .

# Install optional dependencies
pip install -e ".[dev,backtest]"
```

### Configuration
1. Copy `config.yaml` to your working directory
2. Modify settings according to your risk tolerance and trading style
3. Ensure MT5 is running and logged in
4. Set environment variables if needed (see `.env.example`)

## 🚀 Quick Start

### Basic Usage
```python
from tradingagents_v2 import TradingGraph, AgentRegistry
from tradingagents_v2.agents import RegimeAgent, TrendAgent
from tradingagents_v2.config import TradingConfig

# Load configuration
config = TradingConfig()

# Create agent registry
registry = AgentRegistry()
registry.register(RegimeAgent())
registry.register(TrendAgent())

# Create trading graph
graph = TradingGraph(registry, config)

# Run analysis
result = await graph.run("NASDAQ:NVDA", features, portfolio_state)
```

### CLI Usage
```bash
# Run trading analysis
tradingagents analyze --symbol NASDAQ:NVDA --config config.yaml

# Run backtest
tradingagents backtest --config config.yaml --start-date 2024-01-01

# Monitor portfolio
tradingagents monitor --config config.yaml
```

## 📈 Backtesting

The same agent graph runs in backtesting mode:
- Swap `LiveDataLoader` → `HistoricalDataLoader`
- Identical node sequence and decision logic
- Record: Expectancy (R), Hit-rate, Avg hold time, Max DD, Sharpe, Ulcer index
- Save per-recipe stats for calibration

## 🔧 Configuration

### Key Settings
```yaml
# Risk Management
risk:
  base_risk_pct: 0.25          # % equity per trade
  max_daily_drawdown_pct: 2.0  # Circuit breaker
  max_concurrent_trades: 3     # Position limit

# Execution
execution:
  slippage_bp: 2               # Slippage tolerance
  use_bracket_orders: true     # Entry + exit orders
  partial_take_profits: true   # Scale out

# Agents
agents:
  enabled_agents:               # Which agents to use
    - "RegimeAgent"
    - "TrendAgent"
  agent_weights:                # Agent importance
    RegimeAgent: 1.2
```

## 🛡️ Risk Management

### Portfolio Limits
- **Daily Drawdown**: Circuit breaker at configurable threshold
- **Concurrent Trades**: Maximum open positions
- **Leverage Caps**: Per-symbol and portfolio limits
- **News Blackout**: Avoid trading during scheduled events

### Trade-Level Risk
- **Stop Distance**: Minimum ATR-based, maximum risk-based
- **Position Size**: Fixed-fraction risk model
- **Slippage Protection**: Spread guards and tick age checks

## 📊 Monitoring & Attribution

### Performance Metrics
- **Trade Attribution**: Which signals contributed to success/failure
- **Recipe Performance**: Win rate, expectancy per entry type
- **Agent Performance**: Individual agent accuracy and contribution
- **Risk Metrics**: VaR, max drawdown, Sharpe ratio

### Real-Time Monitoring
- Portfolio P&L and risk exposure
- Open positions and order status
- Agent signal alignment
- Market regime changes

## 🔌 MT5 Integration

### Features
- **Bracket Orders**: Entry + stop loss + take profit in single request
- **Order Management**: Modify stops, close positions, partial fills
- **Account Info**: Balance, equity, margin, leverage
- **Symbol Data**: Real-time prices, spreads, volume

### Setup
1. Install MetaTrader 5 terminal
2. Log into your trading account
3. Ensure symbols are visible in Market Watch
4. Set magic number in config for order identification

## 🧪 Development

### Adding New Agents
```python
from tradingagents_v2.core.agent_base import BaseAgent

class CustomAgent(BaseAgent):
    name = "CustomAgent"
    timeframe = Timeframe.MID
    
    async def analyze(self, features, context=None):
        # Your analysis logic here
        return AgentOutput(...)
```

### Extending the Graph
```python
# Add custom nodes to the trading graph
workflow.add_node("custom_analysis", self._custom_analysis)
workflow.add_edge("run_agents", "custom_analysis")
```

### Testing
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=tradingagents_v2

# Run specific test
pytest tests/test_agents.py::test_regime_agent
```

## 📚 API Reference

### Core Classes
- `TradingGraph`: Main orchestrator
- `BaseAgent`: Agent base class
- `AgentRegistry`: Agent management
- `MT5Executor`: Trade execution
- `TradingConfig`: Configuration management

### Data Types
- `AgentOutput`: Standard agent response
- `TimeframeFusion`: Multi-timeframe signals
- `TradeRecipe`: Entry conditions and probability
- `TradePlan`: Complete execution plan

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

Apache License 2.0 - see LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results.

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: [Wiki](https://github.com/TauricResearch/TradingAgents/wiki)
- **Discussions**: GitHub Discussions
- **Email**: yijia.xiao@cs.ucla.edu

---

**TradingAgents v2** - Where AI meets Technical Analysis for Systematic Trading Excellence.
