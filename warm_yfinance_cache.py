import argparse
import json
from pathlib import Path

from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.y_finance import warm_yfinance_history_cache


DEFAULT_TICKERS = ["AAPL", "GOOGL", "AMZN", "MSFT", "META", "TSLA", "NVDA"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warm shared yfinance history cache for common TradingAgents tickers."
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=DEFAULT_TICKERS,
        help="Tickers to prewarm. Defaults to MAG7 used by the local harness.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override tradingagents.data_cache_dir for this run.",
    )
    parser.add_argument(
        "--ttl-seconds",
        type=int,
        default=None,
        help="Override cache TTL for this run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of plain text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_update = {}
    if args.cache_dir:
        config_update["data_cache_dir"] = args.cache_dir
    if args.ttl_seconds is not None:
        config_update["data_cache_ttl_seconds"] = args.ttl_seconds
    if config_update:
        set_config(config_update)

    results = warm_yfinance_history_cache(args.tickers)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    if args.cache_dir:
        print(f"[CACHE_DIR] {Path(args.cache_dir).resolve()}")

    for item in results:
        print(
            f"[{item['status'].upper()}] {item['symbol']} rows={item['rows']} "
            f"path={item['path']}"
        )


if __name__ == "__main__":
    main()
