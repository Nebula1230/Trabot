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
    from datetime import datetime

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
                        choices=["safe", "balanced", "risky", "risky_equity", "scalp", "hft"],
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
    parser.add_argument("--output",   default=None,
                        help="Output HTML path. If omitted, auto-generates a name "
                             "under reports/ using profile, symbols, mid-tf and date.")
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
    parser.add_argument("--signal-interval", type=int, default=None,
                        dest="signal_interval",
                        help="Signal loop interval in seconds (overrides config interval_seconds). "
                             "Controls how often agents run in the backtest.")
    parser.add_argument("--surveillance-interval", type=int, default=None,
                        dest="surveillance_interval",
                        help="Surveillance loop interval in seconds (overrides config "
                             "realtime.surveillance_interval_seconds). "
                             "Controls exit-check frequency in the backtest.")
    parser.add_argument("--data-dir", default=None,
                        dest="data_dir",
                        help="Directory of CSV/Parquet OHLCV files. "
                             "Files should be named {SYMBOL}_{TF}.csv (e.g. EURUSD_1m.csv). "
                             "Bypasses MT5 and yfinance — enables 1m backtests with "
                             "unlimited history from Dukascopy, HistData, etc.")
    parser.add_argument("--independent", action="store_true",
                        help="Run each symbol independently with its own equity "
                             "(no cross-symbol DD/position sharing). "
                             "Default: portfolio mode (shared equity, shared DD).")
    parser.add_argument("--rank-signals", action="store_true",
                        help="Rank competing signals by expected value and fill "
                             "the best ones first (portfolio mode only).")
    parser.add_argument("--no-rank-signals", action="store_true",
                        dest="no_rank_signals",
                        help="Disable signal ranking even if the config enables it.")
    parser.add_argument("--max-concurrent", type=int, default=None,
                        dest="max_concurrent",
                        help="Override max concurrent open trades (overrides profile). "
                             "Useful with --rank-signals to cap how many of the "
                             "ranked signals actually get filled.")
    parser.add_argument("--compare-live", action="store_true",
                        dest="compare_live",
                        help="Auto-detect live trades in the backtest date range "
                             "and show a side-by-side comparison. "
                             "Reads from --log-dir/closed_trades.jsonl.")
    parser.add_argument("--log-dir", default="logs",
                        dest="log_dir",
                        help="Directory containing closed_trades.jsonl from the live bot "
                             "(default: logs). Used with --compare-live.")
    parser.add_argument("--tune", action="store_true",
                        help="Run per-symbol parameter tuning (grid search over "
                             "confidence_sizing, streak_sizing, ct_mid_flip_threshold) "
                             "before the main backtest. Saves results to <output>.tuned.json "
                             "and applies the best params automatically.")
    parser.add_argument("--tune-tf", default=None, dest="tune_tf",
                        help="Mid-timeframe granularity for tuning (default: 1H). "
                             "Lower = more accurate but slower. E.g. --tune-tf 15m")
    parser.add_argument("--tune-workers", default=3, type=int, dest="tune_workers",
                        help="Number of parallel threads for tuning CT backtests (default: 3). "
                             "Set to 1 for sequential execution.")
    parser.add_argument("--tune-file", default=None,
                        dest="tune_file",
                        help="Load per-symbol tuned params from a JSON file produced "
                             "by a previous --tune run (skips re-tuning).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted backtest from the checkpoint file "
                             "(derived from --output, e.g. report.html → report.checkpoint.json). "
                             "If no checkpoint exists the backtest starts from scratch.")

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
        from tradingagents_v2.config.settings import TradingConfig
        cfg = load_config_from_yaml(args.config, profile=args.profile)
    except Exception as e:
        print(f"[ERROR] Could not load config '{args.config}': {e}", file=sys.stderr)
        sys.exit(1)

    # Apply mid-timeframe override (e.g. --mid-tf 1H for hourly granularity)
    if args.mid_tf:
        cfg.timeframes.mid = [args.mid_tf]

    # Enable debug decision logging alongside the existing DebugTracer
    if args.debug:
        cfg.journal.debug_decisions = True

    # Apply interval overrides to config so the backtest engine picks them up
    if args.signal_interval is not None:
        cfg.interval_seconds = args.signal_interval
    if args.surveillance_interval is not None:
        # Inject into the realtime block so engine._run_from_bars reads it
        _rt = cfg.model_dump().get("realtime", {})
        _rt["surveillance_interval_seconds"] = args.surveillance_interval
        cfg = cfg.model_copy(update={"realtime": _rt})

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
    if args.rank_signals:
        cfg.rank_signals = True
    if args.no_rank_signals:
        cfg.rank_signals = False
    if args.max_concurrent is not None:
        cfg.risk.max_concurrent_trades = args.max_concurrent

    # ── Auto-generate output path if not explicitly provided ─────────────
    # Format: reports/{profile}_{SYM1-SYM2}_{mid_tf}_{YYYYMMDD_HHMM}.html
    if args.output is None:
        _reports_dir = Path("reports")
        _reports_dir.mkdir(exist_ok=True)
        _mid_tf_tag = (cfg.timeframes.mid[0] if cfg.timeframes.mid else "1m").replace(" ", "")
        _sym_tag = "-".join(symbols[:5])  # cap at 5 to avoid absurd filenames
        if len(symbols) > 5:
            _sym_tag += f"-plus{len(symbols) - 5}"
        _now_tag = datetime.now().strftime("%Y%m%d_%H%M")
        args.output = str(_reports_dir / f"{args.profile}_{_sym_tag}_{_mid_tf_tag}_{_now_tag}.html")
        print(f"Report output: {args.output}")

    # ── Per-symbol tuning layer ──────────────────────────────────────────
    _tuner_cached_bars = None  # set by --tune to reuse data for final backtest
    if args.tune_file:
        from .symbol_tuner import SymbolTuner
        overrides = SymbolTuner.load_overrides(args.tune_file)
        cfg_dict = cfg.model_dump()
        cfg_dict["symbol_overrides"] = overrides
        cfg = TradingConfig(**cfg_dict)
        print(f"Loaded tuned params for {list(overrides.keys())} from {args.tune_file}")

    if args.tune:
        from .symbol_tuner import SymbolTuner
        print(f"\n{'═'*60}")
        print(f"  Per-symbol tuning: {symbols}")
        print(f"{'═'*60}")
        tuner = SymbolTuner(cfg, data_dir=args.data_dir, initial_equity=args.equity,
                            max_workers=args.tune_workers)
        if args.tune_tf:
            tuner.TUNE_MID_TF = args.tune_tf
            print(f"  Tuning granularity: {args.tune_tf}")
        tune_results = tuner.tune_symbols(symbols, args.start, args.end)
        tune_json = str(Path(args.output).with_suffix(".tuned.json"))
        SymbolTuner.save_results(tune_results, tune_json)
        print(f"\nTuning results saved to {tune_json}")
        for sym, r in tune_results.items():
            print(f"  {sym}: IS Sharpe={r.is_sharpe:.3f}  OOS Sharpe={r.oos_sharpe:.3f}  "
                  f"trials={r.trials_run}")
        # Apply tuned params to config
        overrides = {sym: r.best_params for sym, r in tune_results.items()}
        cfg_dict = cfg.model_dump()
        cfg_dict["symbol_overrides"] = overrides
        cfg = TradingConfig(**cfg_dict)
        _tuner_cached_bars = tuner.cached_bars  # reuse bars for final backtest
        print(f"{'═'*60}\n")

    if cfg.symbol_overrides:
        print(f"[OVERRIDES] symbol_overrides active: {list(cfg.symbol_overrides.keys())}")
        for _so_sym, _so_params in cfg.symbol_overrides.items():
            print(f"  {_so_sym}: {_so_params}")
    engine  = BacktestEngine(cfg)
    if args.data_dir:
        engine.set_data_dir(args.data_dir)

    # ── Checkpoint / resume setup ────────────────────────────────────────
    from .checkpoint import checkpoint_path_for_output, load_checkpoint
    _ckpt_path = checkpoint_path_for_output(args.output)
    _resume_ckpt = None
    if args.resume:
        _resume_ckpt = load_checkpoint(_ckpt_path)
        if _resume_ckpt is not None:
            print(f"Resuming from checkpoint: step {_resume_ckpt['step']} "
                  f"equity={_resume_ckpt['equity']:.2f} "
                  f"trades={len(_resume_ckpt.get('closed_trades', []))}")
        else:
            print("No checkpoint found — starting from scratch.")

    _mode = "independent" if args.independent else "portfolio"
    _mid_tf_display = cfg.timeframes.mid[0] if cfg.timeframes.mid else "1H"
    _n_syms = len(symbols)
    _max_conc = cfg.risk.max_concurrent_trades
    _is_ranked = bool(getattr(cfg, "rank_signals", False))
    _conc_scale = 1.0 if _is_ranked else (min(1.0, _max_conc / _n_syms) if _n_syms > 1 else 1.0)
    _risk_per_trade = args.equity * cfg.risk.base_risk_pct / 100 * _conc_scale
    print(f"Backtesting {symbols}  {args.start}→{args.end}  "
          f"profile={args.profile}  granularity={_mid_tf_display}  mode={_mode}")
    print(f"  equity=${args.equity:,.0f}  risk/trade=${_risk_per_trade:.2f}  "
          f"({_n_syms} symbol{'s' if _n_syms > 1 else ''}, "
          f"concentration×{_conc_scale:.2f})")
    if args.independent:
        results = engine._run_multi_independent(symbols, args.start, args.end, args.equity)
    elif _tuner_cached_bars and len(symbols) == 1 and symbols[0] in _tuner_cached_bars:
        # Reuse bars from tuning for data consistency
        _sym = symbols[0]
        _bars, _mid_tf, _bar_dates = _tuner_cached_bars[_sym]
        print(f"  (reusing {len(_bar_dates)} cached bars from tuning)")
        results = [engine.run_with_bars(
            _sym, args.start, args.end, _bars, _mid_tf, _bar_dates, args.equity
        )]
    else:
        results = engine.run_multi(symbols, args.start, args.end, args.equity,
                                   checkpoint_path=_ckpt_path,
                                   resume_checkpoint=_resume_ckpt)

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

    # ── Backtest vs Live comparison ──
    comparison = None
    if args.compare_live:
        from .compare import load_live_trades, compare_trades, print_comparison
        live_trades = load_live_trades(args.log_dir, args.start, args.end)
        if live_trades:
            comparison = compare_trades(results, live_trades)
            print_comparison(comparison)
        else:
            print(f"\n[compare-live] No live trades found in {args.log_dir}/closed_trades.jsonl "
                  f"for {args.start} → {args.end}")

    output = generate_report(results, args.output, wf_result, comparison=comparison)
    print(f"\nReport saved to: {Path(output).resolve()}")

    json_path = str(Path(args.output).with_suffix(".json"))
    output_json = generate_json_report(results, json_path, wf_result, comparison=comparison)
    print(f"JSON report saved to: {Path(output_json).resolve()}")

    # ── Debug trace output ──
    if _tracer is not None:
        _tracer.print_summary()
        _trace_json = str(Path(args.output).with_suffix(".trace.json"))
        _tracer.save_json(_trace_json)
        print(f"Debug trace saved to: {Path(_trace_json).resolve()}", file=sys.stderr)
