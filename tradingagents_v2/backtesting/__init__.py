"""
Backtesting package.

Quick start::

    from tradingagents_v2.backtesting import BacktestEngine, WalkForwardValidator
    from tradingagents_v2.config.yaml_config import load_config

    cfg    = load_config("config.yaml", profile="balanced")
    engine = BacktestEngine(cfg)
    result = engine.run("EURUSD", "2025-01-01", "2026-01-01")

CLI::

    python -m tradingagents_v2.backtesting \\
        --profile balanced --symbol EURUSD \\
        --start 2025-01-01 --end 2026-01-01 \\
        --output report.html
"""

from .engine import BacktestEngine, BacktestResult, ClosedTrade, SimPosition
from .metrics import compute_metrics, compute_drawdown_series
from .walk_forward import WalkForwardValidator, WalkForwardResult, WFWindow
from .report import generate_report, generate_json_report

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "ClosedTrade",
    "SimPosition",
    "compute_metrics",
    "compute_drawdown_series",
    "WalkForwardValidator",
    "WalkForwardResult",
    "WFWindow",
    "generate_report",
    "generate_json_report",
]


def main() -> None:
    """CLI entry point: ``python -m tradingagents_v2.backtesting``."""
    import argparse
    import sys
    from pathlib import Path

    _ALL_AGENTS = [
        "RegimeAgent", "TrendAgent", "MomentumAgent", "MeanReversionAgent",
        "VolatilityAgent", "BreadthAgent", "PatternAgent", "IntermarketAgent",
        "SessionBreakoutAgent", "DivergenceAgent", "ScalpingAgent",
        "VwapScalpAgent", "SqueezeBreakoutAgent", "OrderFlowAgent",
        "CorrelationAgent", "LLMSentimentAgent",
    ]

    parser = argparse.ArgumentParser(
        prog="python -m tradingagents_v2.backtesting",
        description="Run a backtest and generate an HTML performance report.",
    )
    parser.add_argument("--profile",  default="balanced",
                        choices=["safe", "balanced", "risky", "scalp", "hft"],
                        help="Trading profile to backtest (default: balanced)")
    parser.add_argument("--symbol",   nargs="+", default=None,
                        help="One or more MT5 symbols (default: read from config symbols list)")
    parser.add_argument("--symbols",  nargs="+", default=None,
                        help="Alias for --symbol (for backwards compatibility)")
    parser.add_argument("--start",    required=True,
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end",      required=True,
                        help="End date   YYYY-MM-DD")
    parser.add_argument("--equity",   type=float, default=100_000.0,
                        help="Initial equity in account currency (default: 100000)")
    parser.add_argument("--output",   default="backtest_report.html",
                        help="Output HTML path (default: backtest_report.html)")
    parser.add_argument("--wf",       action="store_true",
                        help="Run walk-forward analysis on the first symbol")
    parser.add_argument("--is-months",  type=int, default=6,
                        help="Walk-forward in-sample months (default: 6)")
    parser.add_argument("--oos-months", type=int, default=2,
                        help="Walk-forward out-of-sample months (default: 2)")
    parser.add_argument("--config",   default="config.demo.yaml",
                        help="Path to YAML config (default: config.demo.yaml)")
    parser.add_argument("--mid-tf",   default="1m",
                        dest="mid_tf",
                        choices=["1m", "5m", "15m", "30m", "1H", "4H", "1D"],
                        help="Override the mid-tier timeframe used as bar granularity "
                             "(e.g. --mid-tf 1H for hourly bars). "
                             "Default: 1m (1-minute bars for all profiles).")
    parser.add_argument("--agents",   nargs="+", default=None,
                        metavar="AGENT",
                        help=(
                            "Run ONLY these agents (space-separated). "
                            "Available: " + ", ".join(_ALL_AGENTS) + ". "
                            "Mutually exclusive with --disable-agents."
                        ))
    parser.add_argument("--disable-agents", nargs="+", default=None,
                        dest="disable_agents",
                        metavar="AGENT",
                        help=(
                            "Disable these agents; all others remain active. "
                            "Mutually exclusive with --agents."
                        ))
    parser.add_argument("--list-agents", action="store_true",
                        dest="list_agents",
                        help="Print all available agent names and exit.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug tracing: prints a structured decision trace "
                             "and SL/TP sizing detail at the end of the run.")

    # Handle --list-agents before argparse enforces required fields
    if "--list-agents" in sys.argv:
        print("Available agents:")
        for a in _ALL_AGENTS:
            print(f"  {a}")
        sys.exit(0)

    args = parser.parse_args()

    if args.agents and args.disable_agents:
        print("[ERROR] --agents and --disable-agents are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    # ── Debug tracer ──
    import logging

    # ── tqdm-compatible logging handler ──────────────────────────────────
    # Standard StreamHandler writes to stderr, which breaks tqdm's \r
    # carriage-return redraws and causes the progress bar to regenerate
    # on every log line.  Route log output through tqdm.write() instead.
    try:
        from tqdm import tqdm as _tqdm_cls
        class _TqdmHandler(logging.StreamHandler):
            """Logging handler that writes through tqdm.write()."""
            def emit(self, record):
                try:
                    msg = self.format(record)
                    _tqdm_cls.write(msg, file=self.stream)
                    self.flush()
                except Exception:
                    self.handleError(record)
        _log_handler = _TqdmHandler(stream=sys.stderr)
    except ImportError:
        _log_handler = logging.StreamHandler(stream=sys.stderr)

    _log_fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _log_handler.setFormatter(_log_fmt)

    # ── Force the tqdm handler onto the root logger ──────────────────────
    # logging.basicConfig() is a NO-OP when the root logger already has
    # handlers (which happens if any import triggered logging before us).
    # Explicitly replace all root handlers with our tqdm-compatible one.
    _root = logging.getLogger()

    if args.debug:
        from .debug_tracer import DebugTracer, set_tracer
        _tracer = DebugTracer(enabled=True)
        set_tracer(_tracer)
        _root.setLevel(logging.DEBUG)
    else:
        _root.setLevel(logging.INFO)
        _tracer = None

    # Silence noisy third-party loggers (both debug and non-debug modes)
    for noisy in ("httpx", "httpcore", "openai", "langchain", "langgraph",
                   "urllib3", "asyncio", "yfinance", "charset_normalizer",
                   "matplotlib", "PIL", "faker", "peewee", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Remove any pre-existing handlers, then install the tqdm-safe one
    for _h in _root.handlers[:]:
        _root.removeHandler(_h)
    _root.addHandler(_log_handler)

    # Load config
    try:
        from tradingagents_v2.config.yaml_config import load_config_from_yaml
        cfg = load_config_from_yaml(args.config, profile=args.profile)
    except Exception as e:
        print(f"[ERROR] Could not load config '{args.config}': {e}", file=sys.stderr)
        sys.exit(1)

    # Apply mid-timeframe override (e.g. --mid-tf 1H for hourly granularity)
    if args.mid_tf:
        cfg.timeframes.mid = [args.mid_tf]

    # Apply agent selection
    if args.agents:
        unknown = [a for a in args.agents if a not in _ALL_AGENTS]
        if unknown:
            print(f"[ERROR] Unknown agent(s): {unknown}. Run --list-agents for valid names.",
                  file=sys.stderr)
            sys.exit(1)
        cfg.agents.enabled_agents = list(args.agents)
        print(f"Agent selection: ONLY {cfg.agents.enabled_agents}")
    elif args.disable_agents:
        unknown = [a for a in args.disable_agents if a not in _ALL_AGENTS]
        if unknown:
            print(f"[ERROR] Unknown agent(s): {unknown}. Run --list-agents for valid names.",
                  file=sys.stderr)
            sys.exit(1)
        cfg.agents.enabled_agents = [a for a in _ALL_AGENTS if a not in args.disable_agents]
        print(f"Agent selection: all except {args.disable_agents} → {cfg.agents.enabled_agents}")

    symbols = args.symbols or args.symbol or getattr(cfg, "symbols", None) or ["EURUSD"]
    engine  = BacktestEngine(cfg)

    _mid_tf_display = cfg.timeframes.mid[0] if cfg.timeframes.mid else "1H"
    print(f"Backtesting {symbols}  {args.start}→{args.end}  "
          f"profile={args.profile}  granularity={_mid_tf_display}")
    results = engine.run_multi(symbols, args.start, args.end, args.equity)

    for r in results:
        m = compute_metrics(r)
        print(
            f"  {r.symbol:8s}  trades={m['total_trades']:4d}  "
            f"win={m['win_rate']:.1%}  PF={m['profit_factor']:.2f}  "
            f"return={m['total_return_pct']:+.2f}%  "
            f"maxDD={m['max_drawdown_pct']:.2f}%  "
            f"Sharpe={m['sharpe']:.2f}"
        )

    wf_result = None
    if args.wf:
        print(
            f"\nWalk-forward [{symbols[0]}] "
            f"IS={args.is_months}m OOS={args.oos_months}m ..."
        )
        validator = WalkForwardValidator(cfg, args.is_months, args.oos_months, args.equity)
        wf_result = validator.run(symbols[0], args.start, args.end)
        print(
            f"  verdict={wf_result.verdict}  "
            f"avg_efficiency={wf_result.avg_efficiency:.2f}"
        )

    output = generate_report(results, args.output, wf_result)
    print(f"\nReport saved to: {Path(output).resolve()}")

    json_path = str(Path(args.output).with_suffix(".json"))
    output_json = generate_json_report(results, json_path, wf_result)
    print(f"JSON report saved to: {Path(output_json).resolve()}")

    # ── Debug trace output ──
    if _tracer is not None:
        _tracer.print_summary()
        _trace_json = str(Path(args.output).with_suffix(".trace.json"))
        _tracer.save_json(_trace_json)
        print(f"Debug trace saved to: {Path(_trace_json).resolve()}", file=sys.stderr)
