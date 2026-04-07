"""Generate a backtest-compatible report from live trading logs.

Usage::

    python -m tradingagents_v2.monitoring.live_report \\
        --log-dir logs \\
        --profile risky \\
        --equity 100000 \\
        --output live_report.html

This reads ``closed_trades.jsonl`` from the journal log directory,
reconstructs a ``BacktestResult``, and produces the same HTML + JSON
reports that ``python -m tradingagents_v2.backtesting`` generates.

You can then compare the live report side-by-side with a backtest
report to verify real-world performance matches simulation.
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tradingagents_v2.monitoring.live_report",
        description="Generate a performance report from live trading logs.",
    )
    parser.add_argument(
        "--log-dir", default="logs",
        help="Journal log directory containing closed_trades.jsonl (default: logs)",
    )
    parser.add_argument(
        "--profile", default="live",
        help="Profile label for the report (default: live)",
    )
    parser.add_argument(
        "--equity", type=float, default=100_000.0,
        help="Initial equity used for return calculations (default: 100000)",
    )
    parser.add_argument(
        "--output", default="live_report.html",
        help="Output HTML path (default: live_report.html)",
    )
    parser.add_argument(
        "--config", default=None,
        help="Optional YAML config path to embed in the report metadata",
    )
    args = parser.parse_args()

    journal_path = Path(args.log_dir) / "closed_trades.jsonl"
    if not journal_path.exists():
        print(
            f"[ERROR] No closed_trades.jsonl found in {args.log_dir}/\n"
            f"The live bot writes this file automatically. "
            f"Make sure the bot has run and closed at least one trade.",
            file=sys.stderr,
        )
        sys.exit(1)

    from .journal import TradeJournal
    from ..backtesting.metrics import compute_metrics
    from ..backtesting.report import generate_report, generate_json_report

    # Load optional config for metadata
    config_dict = {}
    if args.config:
        try:
            from ..config.yaml_config import load_config_from_yaml
            cfg = load_config_from_yaml(args.config, profile=args.profile)
            config_dict = cfg.model_dump()
        except Exception as e:
            print(f"[WARN] Could not load config: {e}", file=sys.stderr)

    journal = TradeJournal(log_dir=args.log_dir)
    result = journal.load_live_result(
        profile=args.profile,
        initial_equity=args.equity,
        config=config_dict,
    )

    if not result.trades:
        print("[ERROR] No closed trades found in the journal.", file=sys.stderr)
        sys.exit(1)

    m = compute_metrics(result)
    print(
        f"Live trading report ({args.profile})\n"
        f"  Trades: {m['total_trades']}  "
        f"Win rate: {m['win_rate']:.1%}  "
        f"PF: {m['profit_factor']:.2f}  "
        f"Return: {m['total_return_pct']:+.2f}%  "
        f"Max DD: {m['max_drawdown_pct']:.2f}%  "
        f"Sharpe: {m['sharpe']:.2f}"
    )

    # Per-symbol breakdown
    symbols = sorted({t.symbol for t in result.trades})
    if len(symbols) > 1:
        print(f"\nPer-symbol breakdown ({len(symbols)} symbols):")
        for sym in symbols:
            sym_trades = [t for t in result.trades if t.symbol == sym]
            wins = sum(1 for t in sym_trades if t.pnl > 0)
            total_pnl = sum(t.pnl for t in sym_trades)
            wr = wins / len(sym_trades) if sym_trades else 0
            print(
                f"  {sym:12s}  trades={len(sym_trades):3d}  "
                f"win={wr:.1%}  pnl={total_pnl:+.2f}"
            )

    output = generate_report([result], args.output)
    print(f"\nHTML report: {Path(output).resolve()}")

    json_path = str(Path(args.output).with_suffix(".json"))
    output_json = generate_json_report([result], json_path)
    print(f"JSON report: {Path(output_json).resolve()}")


if __name__ == "__main__":
    main()
