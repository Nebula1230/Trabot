# TradingAgents v2

Multi-agent algorithmic trading system for forex and index CFDs. 15 specialised technical agents vote on direction across three timeframes; a fusion engine blends their signals, gates on alignment, sizes risk, and executes bracket orders on MetaTrader 5.

Runs live (MT5 via Docker on Linux, native on Windows) or in backtest with realistic fills, dynamic spreads, and commission/swap modelling.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Signal Agents](#signal-agents)
3. [Multi-Timeframe Fusion](#multi-timeframe-fusion)
4. [Alignment & Entry Gates](#alignment--entry-gates)
5. [Trade Execution](#trade-execution)
6. [Trailing Stop & Profit Taking](#trailing-stop--profit-taking)
7. [Risk Management](#risk-management)
8. [Profiles](#profiles)
9. [Backtesting Engine](#backtesting-engine)
10. [Monitoring & Adaptation](#monitoring--adaptation)
11. [Setup & Usage](#setup--usage)
12. [Project Structure](#project-structure)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        TradingRunner (runner.py)                     │
│                                                                      │
│  ┌─────────────┐   ┌──────────────────────────────────────────────┐  │
│  │  Data Layer  │   │            TradingGraph (LangGraph)          │  │
│  │             │   │                                              │  │
│  │ • MT5 bars  │──▶│  run_agents ──▶ fuse_timeframes              │  │
│  │   D1/1H/15m │   │                    │                         │  │
│  │ • Features  │   │              check_alignment                 │  │
│  │   60+ indic │   │                    │                         │  │
│  │ • Macro     │   │              generate_recipe                 │  │
│  │   DXY/VIX/  │   │                    │                         │  │
│  │   Crude/Yld │   │              risk_check                      │  │
│  │ • Calendar  │   │                    │                         │  │
│  │   ForexFact │   │              create_plan ──▶ execute_trade   │  │
│  └─────────────┘   └──────────────────────────────────────────────┘  │
│                                                                      │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────────┐  │
│  │  Trailing   │   │   Risk       │   │   Monitoring             │  │
│  │  Stop Mgr   │   │   Gates      │   │   • Journal (jsonl/csv)  │  │
│  │  3-stage    │   │   • Kelly    │   │   • Agent calibration    │  │
│  │  lock-in    │   │   • DD halt  │   │   • Adaptive weights     │  │
│  │  + partials │   │   • VIX scl  │   │   • PnL snapshots        │  │
│  └─────────────┘   └──────────────┘   └──────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────────┐│
│  │                  MT5 Executor                                    ││
│  │  mt5linux (Docker/Linux) │ MetaTrader5 (Windows) │ Simulation   ││
│  └──────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────┘
```

**Two concurrent loops** run in the runner:

| Loop | Frequency | Purpose |
|------|-----------|---------|
| **Signal cycle** | Every N seconds (300s balanced, 60s scalp) | Analyse all symbols, generate and execute new trades |
| **Surveillance** | Every 10–60s | Trail open positions, check exit signals, update SL/TP |

---

## Signal Agents

15 agents, each producing a directional score (−1 to +1) and confidence (0 to 1). Organised into **swing/context agents** (active on all profiles) and **scalp agents** (dominant in the scalp profile, suppressed elsewhere).

### Swing / Context Agents

| Agent | Tier | What It Does |
|-------|------|-------------|
| **TrendAgent** | LONG | Market structure (HH/HL, LH/LL) + EMA 20/50/200 slope & stack + ADX confirmation. The D1 directional backbone. |
| **MomentumAgent** | MID | RSI(14/4) regime logic, MACD histogram delta, ROC(10), Bollinger %B. Confirms impulse with room to run vs. exhaustion. |
| **MeanReversionAgent** | SHORT | Bollinger %B extremes, Keltner position, RSI extremes, VWAP distance. Blocks chasing extended moves. |
| **VolatilityAgent** | SHORT | ATR(14), realised vol, BB/Keltner width. Outputs volatility regime (low/normal/high) and minimum safe stop distance. |
| **RegimeAgent** | MID | ADX(14), RVI, Hurst exponent. Outputs trendiness (0–1) and vol_state. Trendiness controls fusion weighting of all other agents. |
| **PatternAgent** | SHORT | Swing patterns, breakout bases, flags, consolidation, support/resistance. Graded pattern quality score. |
| **IntermarketAgent** | LONG | Pre-weighted macro signals: DXY (0.75), VIX (0.65), crude (0.35), yield (0.55). Per-symbol sensitivity mapping. |
| **BreadthAgent** | LONG | Index trend, advance/decline, % above 50/200 EMA, sector direction. Used as a filter gate, not scored in fusion. |
| **SessionBreakoutAgent** | MID | London/NY breakout of the Asian session range. Only fires on 1H bars during session opens. |
| **DivergenceAgent** | MID | Regular & hidden RSI divergences with MACD confirmation. |
| **CorrelationAgent** | LONG | DXY divergence (0.45w), peer-pair divergence (0.30w), risk-appetite divergence (0.25w). Cross-symbol directional pressure. |

### Scalp / Microstructure Agents

| Agent | What It Does |
|-------|-------------|
| **ScalpingAgent** | Micro-RSI (RSI4 vs RSI14), MACD histogram, range breakout, spread guard, intrabar volume. 1m momentum snap. |
| **VwapScalpAgent** | VWAP-distance regimes: reversion (>2 ATR), breakout (0.5–2 ATR), consolidation (<0.5 ATR). RSI exhaustion filter. |
| **SqueezeBreakoutAgent** | BB inside Keltner = squeeze. Detects squeeze release using MACD + ROC + ADX. State machine prevents re-fire. |
| **OrderFlowAgent** | Candle body commitment (BB%B), ROC velocity, realised vol expansion, VWAP confirmation for institutional sponsorship. |

### Experimental

| Agent | What It Does |
|-------|-------------|
| **LLMSentimentAgent** | Optional bridge to upstream TradingAgents LLM pipeline (market/social/news/fundamentals analysts). Thread-pool dispatch, async timeout, circuit breaker, per-symbol throttle cache (4h TTL). Score mapping: BUY→+0.70, HOLD→0, SELL→−0.70. |

---

## Multi-Timeframe Fusion

The fusion engine (`_fuse_timeframes` in `graph.py`) blends agent outputs into three tier scores—`dir_long`, `dir_mid`, `dir_short`—each in the range [−1, +1].

### 1. Regime-Dynamic Weighting

RegimeAgent's trendiness score (0–1) redistributes agent influence:

| Market State | Amplified | Suppressed |
|-------------|-----------|------------|
| **Trending** (trendiness > 0.55) | TrendAgent 2.0×, MomentumAgent 1.6×, IntermarketAgent 1.5× | MeanReversionAgent 0.1×, DivergenceAgent 0.2× |
| **Ranging** (trendiness < 0.40) | MeanReversionAgent 1.8×, DivergenceAgent 1.6× | TrendAgent 0.4× |
| **Neutral** | All agents 1.0× | — |

### 2. Trend-Aware Dampening

In trending markets, counter-trend agents (MeanReversion, Divergence) are progressively muted:

```
dampening = max(0, 1.0 − (trendiness − 0.55) / 0.30)
```

At trendiness = 0.70 → halved. At 0.85 → fully muted. Prevents overbought RSI and bearish divergences from dragging fusion away from the true trend.

### 3. Correlation Auto-Weighting

Overlapping agent pairs (e.g., TrendAgent + MomentumAgent) are penalised when both vote the same direction, preventing double-counting of similar signals.

### 4. Per-Tier Weighted Average

```
dir_score = Σ(agent.dir × agent.conf × weight × regime_mult × corr_adj)
            ÷ Σ(agent.conf × weight × regime_mult × corr_adj)
```

With **consensus dampening**: if agents within a tier disagree, the tier score is scaled down. Strong unanimous signals pass through at full strength.

### 5. Trendiness Floor by Tier

| Tier | Floor | Effect |
|------|-------|--------|
| LONG | 0.85 | Multi-day trend valid regardless of short-term noise |
| MID | 0.65 | Moderate adjustment |
| SHORT | 0.50 | Most sensitive to regime changes |

---

## Alignment & Entry Gates

After fusion, `check_alignment` runs a multi-gate filter. A trade is only generated if all gates pass.

| Gate | Description |
|------|-------------|
| **Score thresholds** | `dir_long > L`, `dir_mid > M`, `dir_short > S`. Profile-dependent (balanced: 0.15/0.12/0.08). |
| **Dead zone scaling** | 22:00–06:00 UTC: all thresholds × dead_zone_factor (1.4×). Prevents false entries during low liquidity. |
| **Open zone scaling** | Optional London/NY open threshold boost during spread spikes. |
| **Hard-block hours** | Optionally skip all entries during specific UTC hours. |
| **Breadth filter** | BreadthAgent score must be above breadth_min (default −0.50). |
| **ADX trend gate** | Optional: require D1 ADX above minimum to avoid choppy markets. |
| **Pullback entry path** | When D1 + MID agree but SHORT weakly opposes (retracing a strong trend), allow entry with `short_min − pullback_tolerance`. |
| **Counter-trend scalp** | Optional: when MID + SHORT both strongly oppose D1, allow a quick intraday scalp with reduced R:R and a win-prob penalty. |

**Recipe generation** follows: the system computes `win_probability` from historical closed trades, `expected_value`, and `risk_reward_ratio`.

**Risk check** then validates portfolio constraints (see [Risk Management](#risk-management)).

---

## Trade Execution

### MT5 Integration

Three backends in priority order:

| Backend | Platform | Connection |
|---------|----------|------------|
| **mt5linux** | Linux / Docker | TCP to MT5 Docker container (`docker compose up -d`) |
| **MetaTrader5** | Windows | Official Python package, native terminal |
| **Simulation** | Any | All orders mocked (fallback for testing) |

### Order Flow

1. Signal approved → `create_plan` sizes the position:
   ```
   risk_per_trade = equity × base_risk_pct × kelly_multiplier × vix_scaling
   quantity = risk_per_trade / stop_distance_price
   ```
2. `execute_trade` places a bracket order: entry + SL + TP
3. Entry fills at next bar open + half spread (pending → fill on next tick)
4. SL/TP managed by the trailing stop manager

### Position Sizing Inputs

| Factor | Description |
|--------|-------------|
| **base_risk_pct** | Fixed % of equity per trade (profile-set: 0.05–0.20%) |
| **Kelly multiplier** | Half-Kelly adaptive sizing from per-symbol win-rate history (0.60–1.75×) |
| **VIX scaling** | Reduces size when VIX elevated (floor 0.35–0.50× of base) |
| **Symbol cap** | Per-symbol risk multiplier (e.g., gold capped at 0.30× on risky) |

---

## Trailing Stop & Profit Taking

Three-stage lock-in strategy based on initial risk distance (1R = |entry − original SL|):

| Stage | Trigger | SL Action |
|-------|---------|-----------|
| **0** | Price reaches +early_be_r × 1R (e.g., +0.5R) | Move SL to entry ± buffer |
| **1** | Price reaches +1R | Move SL to entry ± buffer (breakeven) |
| **2** | Price reaches +1.5R | Lock SL at entry + 0.5R |
| **3** | Price reaches +2R+ | ATR trail: SL = price − atr_multiplier × ATR |

**Breakeven buffer** (`be_buffer_r`): instead of moving SL to exact entry (which loses the spread on every BE stop), the SL is placed `be_buffer_r × 1R` back from entry. Default 0.10 (base), 0.15 (risky).

### Partial Take-Profits

| Event | Action |
|-------|--------|
| **Partial TP1** | At +1R: close 40–60% of position, move SL to breakeven (with buffer) |
| **Partial TP2** | At +2–2.5R: close 40–50% of remainder, bump SL to entry + 0.7R |
| **Windfall exit** | At +3–4R: close ALL (capture spike before reversal) |
| **TP extension** | Stage 3: if price closes within 1 ATR of TP, push TP forward (let winners run) |

### Time-Based Exits

| Exit | Description |
|------|-------------|
| **Time-stop** | Close if open > N hours (safe: 8h, balanced: 10h, risky: 12h, scalp: 2h) |
| **Stale SL tightening** | After N hours losing, cap SL at −0.75R instead of full −1R |
| **D1 flip exit** | If D1 score convincingly reverses past flip threshold, close position |
| **Conviction fade** | Optional: tighten SL + TP when signal conviction weakens |

---

## Risk Management

### Portfolio-Level Guards

| Guard | Description |
|-------|-------------|
| **Daily drawdown halt** | All entries blocked if daily P&L ≤ −max_daily_dd% (0.5–2.0%) |
| **Weekly drawdown cap** | Same, weekly scope (1.5–4%) |
| **Max concurrent trades** | Hard cap on total open positions (3–8) |
| **Correlated positions** | Max N positions in same currency direction (DXY-based grouping) |
| **Portfolio leverage cap** | Total margin / equity ≤ cap |
| **Per-symbol leverage cap** | Per-symbol exposure ≤ equity × cap |
| **Entry cooldown** | Min time between entries on same symbol (60–90 min) |
| **Daily trade cap** | Hard limit on total orders per day (4–20) |
| **Spread guard** | Reject entry if spread > 20% of stop distance |
| **Pivot buffer** | Reject if entry within 0.3–0.5 × ATR of weekly support/resistance |
| **News blackout** | Block entries ± N min around high-impact ForexFactory events |

### Kelly Criterion Tracker

Per-symbol adaptive position sizing:

```
f* = (win_rate × avg_win/avg_loss − (1 − win_rate)) / (avg_win/avg_loss)
half_kelly = 0.5 × f*
multiplier = clamp(1.0 + half_kelly, 0.60, 1.75)
```

Requires minimum 20 closed trades before adjusting. Cached per symbol, refreshed hourly.

### VIX Risk Scaling

When VIX is elevated, position size is reduced automatically:
- **Threshold**: 0.30–0.40 (profile-dependent)
- **Floor**: 0.35–0.50× of base risk
- Prevents large positions during macro fear spikes

---

## Profiles

Four risk profiles applied as deep-merge config patches. Key differences:

| Parameter | Safe | Balanced | Risky | Scalp |
|-----------|------|----------|-------|-------|
| **Risk per trade** | 0.05% | 0.10% | 0.20% | 0.05% |
| **Daily DD halt** | 0.50% | 1.00% | 2.00% | 1.00% |
| **Max trades/day** | 10 | 6 | 4 | 40 |
| **Entry cooldown** | 60 min | 60 min | 90 min | 2 min |
| **Long threshold** | 0.25 | 0.15 | 0.20 | −1.0 (bypassed) |
| **Mid threshold** | 0.18 | 0.12 | 0.15 | −1.0 (bypassed) |
| **Short threshold** | 0.12 | 0.08 | 0.08 | 0.05 |
| **Partial TP1** | 60% at +1R | 50% at +1R | 40% at +1R | 50% at +1R |
| **Time-stop** | 8h | 10h | 12h | 2h |
| **Scale-in** | Disabled | 2 max, 50% risk | 2 max, 50% risk | Disabled |
| **Scalp agents weight** | 0.10× | 0.15× | 0.20× | 2.0–3.0× |
| **Counter-trend** | Disabled | Enabled | Disabled | N/A |
| **Cycle interval** | 300s | 300s | 300s | 60s |
| **Primary symbols** | Forex + indices | Forex + indices | Forex + indices | USTEC, US30, DAX |

---

## Backtesting Engine

Full-fidelity simulation in `backtesting/engine.py`. The same agent graph, fusion, and risk logic run on historical bars.

### Realistic Fill Model

| Aspect | Model |
|--------|-------|
| **Entry** | Fills at next bar open + half dynamic spread (no look-ahead) |
| **Dynamic spread** | `base_spread × (bar_range / ATR14)`, capped at 5× base. Models broker widening during volatile bars. |
| **Same-bar SL/TP** | Intra-bar path inferred from OHLC shape: bullish bar → O→L→H→C (dip first), bearish → O→H→L→C. Whichever level hit first wins. |
| **Slippage** | ±0.3 pip random noise on every fill |
| **Spread guard** | Rejects entry when spread > 20% of stop distance |
| **Commission** | Round-trip per-lot: $7 forex, $5 gold, $3 indices |
| **Swap** | Direction-aware overnight financing per symbol, Wed 3× for weekend roll |
| **Lot rounding** | Rounds to 0.01 step, clamps to broker min/max |

### Walk-Forward Analysis

`backtesting/walk_forward.py` splits data into overlapping train/test windows, re-optimises on train, tests on unseen out-of-sample, then rolls forward.

### Metrics

Win rate, profit factor, Sharpe ratio, Calmar ratio, max drawdown, expectancy in R-units, per-symbol breakdown. HTML report generation included.

### Debug Tracer

`backtesting/debug_tracer.py` records agent votes, fusion scores, alignment decisions, and recipe selection at each bar for post-hoc playback.

---

## Monitoring & Adaptation

### Trade Journal

Persists to `logs/`:

| File | Content |
|------|---------|
| `decisions_YYYY-MM-DD.jsonl` | One JSON per symbol per cycle (decision, agent outputs, features) |
| `trades_YYYY-MM-DD.csv` | One row per order (entry, SL, TP, risk, ticket, confidence) |
| `pnl_YYYY-MM-DD.jsonl` | Periodic P&L snapshots (unrealised, realised, total) |
| `summary.json` | Running stats (total trades, win rate, profit factor, equity curve) |

### Agent Calibration Tracker

Records which agents voted which way on each trade. When a trade closes, scores each vote as correct/incorrect. Accumulates per-agent win rate and contribution. Persists to `logs/agent_calibration.json`.

### Adaptive Weight Manager

Recomputes agent fusion weights every 4 hours based on tracked hit-rate:

```
hit_edge   = hit_rate − 0.50
shrinkage  = n / (n + shrink_n)          # Bayesian pull-to-prior
new_weight = base_weight × (1 + sensitivity × hit_edge × shrinkage)
clamped to [base × min_mult, base × max_mult]
```

Agents that consistently vote correctly earn more fusion influence. Regularisation prevents overreaction on thin data (min 30 trades before adjusting).

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- MetaTrader 5 terminal (demo or live account)
- Docker (Linux) or native MT5 (Windows)

### Installation

```bash
git clone <repo>
cd TradingAgents-v2

python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -e .
```

### MT5 on Linux (Docker)

```bash
# Start the MT5 Docker container
docker compose up -d

# Open VNC and log into your broker's DEMO account (one-time)
# http://localhost:6081/vnc.html

# Install the mt5linux client
pip install mt5linux
```

### Configuration

Edit `config.demo.yaml`:

```yaml
mt5:
  host:     "localhost"
  port:     18812
  login:    12345678         # ← your MT5 demo account
  password: "your_password"  # ← your MT5 demo password
  server:   "YourBroker-Demo"

symbols:
  - EURUSD
  - GBPUSD
  - USDJPY
  - US30
  - US500

interval_seconds: 300   # signal cycle every 5 min
```

### Running

```bash
# Live trading (connects to MT5)
python run_demo.py --live --profile balanced

# Single cycle then exit (debug)
python run_demo.py --live --once

# Override profile
python run_demo.py --live --profile risky
python run_demo.py --live --profile scalp

# Override config file
python run_demo.py --live --config my_config.yaml
```

### Backtesting

```bash
python trading_test.py --profile risky --symbol EURUSD --start 2026-01-01 --end 2026-03-20
```

### Graceful Shutdown

Press `Ctrl+C` — the runner finishes the current cycle, flushes journals, releases the profile lock, and exits cleanly. A per-profile lock file (`/tmp/tradingbot_{profile}.lock`) prevents duplicate instances.

---

## Project Structure

```
TradingAgents-v2/
├── run_demo.py                     # CLI entry point (--live, --profile, --once)
├── trading_test.py                 # Backtest entry point
├── config.demo.yaml                # Base configuration (symbols, MT5, thresholds)
├── docker-compose.yml              # MT5 Docker container
├── requirements.txt
├── pyproject.toml
│
└── tradingagents_v2/
    ├── runner.py                   # TradingRunner — signal loop + surveillance loop
    │
    ├── core/
    │   ├── types.py                # AgentOutput, TechnicalFeatures, TradeRecipe, etc.
    │   ├── agent_base.py           # BaseAgent abstract class + AgentRegistry
    │   └── graph.py                # TradingGraph — LangGraph 7-step pipeline
    │
    ├── agents/                     # 15 signal agents
    │   ├── trend_agent.py          # TrendAgent (LONG)
    │   ├── momentum_agent.py       # MomentumAgent (MID)
    │   ├── mean_reversion_agent.py # MeanReversionAgent (SHORT)
    │   ├── volatility_agent.py     # VolatilityAgent (SHORT)
    │   ├── regime_agent.py         # RegimeAgent (MID) — controls fusion weights
    │   ├── pattern_agent.py        # PatternAgent (SHORT)
    │   ├── breadth_agent.py        # BreadthAgent (LONG) — filter only
    │   ├── intermarket_agent.py    # IntermarketAgent (LONG)
    │   ├── session_agent.py        # SessionBreakoutAgent (MID)
    │   ├── divergence_agent.py     # DivergenceAgent (MID)
    │   ├── correlation_agent.py    # CorrelationAgent (LONG)
    │   ├── scalping_agent.py       # ScalpingAgent (SHORT)
    │   ├── vwap_agent.py           # VwapScalpAgent (SHORT)
    │   ├── squeeze_agent.py        # SqueezeBreakoutAgent (SHORT)
    │   ├── orderflow_agent.py      # OrderFlowAgent (SHORT)
    │   └── llm_sentiment_agent.py  # LLMSentimentAgent (LONG, optional)
    │
    ├── config/
    │   ├── settings.py             # TradingConfig Pydantic model
    │   └── yaml_config.py          # YAML loading + profile deep-merge patches
    │
    ├── data/
    │   └── loader.py               # DataLoader — MT5 bars, 60+ indicators, macro signals
    │
    ├── execution/
    │   ├── mt5_executor.py         # MT5Executor — bracket orders, SL/TP, account info
    │   └── trailing_stop.py        # TrailingStopManager — 3-stage + partials + windfall
    │
    ├── risk/
    │   └── kelly_tracker.py        # KellyTracker — per-symbol half-Kelly adaptive sizing
    │
    ├── backtesting/
    │   ├── engine.py               # BacktestEngine — full sim with dynamic spread, swaps
    │   ├── walk_forward.py         # Walk-forward optimisation framework
    │   ├── metrics.py              # Win rate, Sharpe, drawdown, per-recipe stats
    │   └── debug_tracer.py         # Bar-level decision recording for playback
    │
    └── monitoring/
        ├── journal.py              # TradeJournal — decisions, trades, PnL to logs/
        ├── agent_tracker.py        # AgentCalibrationTracker — per-agent vote scoring
        └── adaptive_weights.py     # AdaptiveWeightManager — Bayesian weight updates
```

---

## Disclaimer

This software is for educational and research purposes. Trading involves substantial risk of loss. Past performance does not guarantee future results.
