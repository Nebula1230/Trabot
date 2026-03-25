#!/usr/bin/env python
"""
run_demo.py — Demo trading runner for TradingAgents-v2
=======================================================

Quick start (Linux with Docker)
-------------------------------
1. Start the MT5 Docker container:
       docker compose up -d
   Then open http://localhost:6081/vnc.html and log into your DEMO account.

2. Install the mt5linux client:
       pip install mt5linux

3. Edit config.demo.yaml — fill in mt5.login, mt5.password, mt5.server
   (host/port default to localhost:8001 which matches mt5docker).

4. Run:
       python run_demo.py --live
   or for a single one-shot cycle:
       python run_demo.py --live --once

Logs are written to the  logs/  folder (configurable via config.demo.yaml).
Press Ctrl+C to stop gracefully.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path


def _setup_logging(log_dir: str, level: str = "INFO"):
    """Configure logging to both console and a rotating file."""
    from logging.handlers import RotatingFileHandler
    from datetime import datetime

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"bot_{datetime.now().strftime('%Y-%m-%d')}.log"

    fmt = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5),
    ]
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, datefmt=datefmt, handlers=handlers)

    # Silence noisy third-party libs
    for noisy in ("httpx", "httpcore", "openai", "langchain", "langgraph"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _check_mt5_installed():
    """Return True if mt5linux (Linux/Docker) or official MetaTrader5 (Windows) is available."""
    try:
        import mt5linux  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import MetaTrader5  # noqa: F401
        return True
    except ImportError:
        return False


def _print_startup_banner(config, simulation: bool):
    sep = "═" * 72
    print(f"\n{sep}")
    print("  TradingAgents-v2  •  Demo Trading Bot")
    print(sep)
    print(f"  Mode      : {'SIMULATION (no real orders)' if simulation else '⚡ LIVE DEMO (real MT5 orders)'}")
    print(f"  Profile   : {config.profile.upper()}")
    print(f"  Symbols   : {', '.join(config.symbols)}")
    print(f"  Interval  : every {config.interval_seconds}s  "
          f"({config.interval_seconds // 60} min)")
    print(f"  Risk/trade: {config.risk.base_risk_pct}%  "
          f"| Max DD: {config.risk.max_daily_drawdown_pct}%  "
          f"| Max pos: {config.risk.max_concurrent_trades}")
    print(f"  Log dir   : {config.journal.log_dir}/")
    if not simulation:
        print(f"  MT5 server: {config.mt5.server or '(not set)'}  "
              f"login: {config.mt5.login or '(not set)'}")
    print(sep)
    print("  Press Ctrl+C to stop gracefully.\n")


async def _main(config_path: str, once: bool, simulation: bool, profile: str):
    from tradingagents_v2.config.yaml_config import load_config_from_yaml
    from tradingagents_v2.runner import TradingRunner

    # Load config with active profile
    config = load_config_from_yaml(config_path, profile=profile)

    # Validate
    if not config.validate_config():
        sys.exit(1)

    # Warn if MT5 client not installed
    mt5_available = _check_mt5_installed()
    if not mt5_available and not simulation:
        print("\n⚠  No MT5 Python client found.")
        print("   On Linux (Docker):  pip install mt5linux")
        print("   On Windows:         pip install MetaTrader5")
        print("   Falling back to SIMULATION mode.\n")
        simulation = True

    if not simulation and (not config.mt5.login or not config.mt5.server):
        print("\n⚠  MT5 credentials not set in config (login / server).")
        print("   Edit config.demo.yaml and fill in mt5.login, mt5.password, mt5.server.")
        print("   Falling back to SIMULATION mode.\n")
        simulation = True

    _setup_logging(config.journal.log_dir, config.log_level)
    _print_startup_banner(config, simulation)

    runner = TradingRunner(config=config, simulation=simulation)

    if once:
        results = await runner.run_once()
        account = runner.executor.get_account_info()
        equity = account.get("equity") if account else None
        runner.journal.print_cycle_banner(1, results, equity)
        return

    await runner.run_forever()


def main():
    parser = argparse.ArgumentParser(
        description="TradingAgents-v2 demo trading bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="config.demo.yaml",
        help="Path to YAML config file (default: config.demo.yaml)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single analysis cycle and exit",
    )
    parser.add_argument(
        "--simulation", action="store_true", default=True,
        help="Force simulation mode even if MT5 is available (default: True)",
    )
    parser.add_argument(
        "--live", action="store_true", default=False,
        help="Use live MT5 connection (requires MT5 installed and credentials set)",
    )
    parser.add_argument(
        "--profile", default="balanced",
        choices=["safe", "balanced", "risky", "scalp"],
        help="Risk profile preset: safe | balanced (default) | risky | scalp",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file '{config_path}' not found.")
        print("Copy and edit config.demo.yaml to get started.")
        sys.exit(1)

    simulation = not args.live

    try:
        asyncio.run(_main(str(config_path), once=args.once,
                          simulation=simulation, profile=args.profile))
    except KeyboardInterrupt:
        print("\n\nBot stopped by user.")


if __name__ == "__main__":
    main()
