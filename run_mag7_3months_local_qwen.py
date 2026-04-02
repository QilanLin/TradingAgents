import csv
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from langchain_core.runnables import Runnable

from tradingagents.default_config import DEFAULT_CONFIG
import tradingagents.graph.trading_graph as tg
from tradingagents.dataflows.config import get_config, set_config
from tradingagents.dataflows.y_finance import (
    get_balance_sheet as get_yf_balance_sheet,
    get_cashflow as get_yf_cashflow,
    get_fundamentals as get_yf_fundamentals,
    get_income_statement as get_yf_income_statement,
    get_insider_transactions as get_yf_insider_transactions,
)


TICKERS = ["AAPL", "GOOGL", "AMZN", "MSFT", "META", "TSLA", "NVDA"]
MONTHS = {
    "2025-09": ("2025-09-01", "2025-09-30"),
    "2025-10": ("2025-10-01", "2025-10-31"),
    "2026-01": ("2026-01-01", "2026-01-31"),
}
INITIAL_CAPITAL = 1_000_000.0

PRICE_CANDIDATES = [
    "/root/private_data/experiments0202change/data_cache/price/{ticker}_plain_daily_2024-09-30_2025-09-30.csv",
    "/root/private_data/data_cache/price/{ticker}_adjusted_2024-10-31_2025-10-31.csv",
    "/root/private_data/data_cache/price/{ticker}_plain_daily_2025-01-31_2026-01-31.csv",
    "/root/private_data/data_cache/price/{ticker}_adjusted_2025-01-16_2026-01-16.csv",
]


def get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def extract_rating(text: str) -> str:
    upper = (text or "").upper()
    for token in ["OVERWEIGHT", "UNDERWEIGHT", "BUY", "SELL", "HOLD"]:
        if token in upper:
            return token
    return "HOLD"


def official_rating_rank(rating: str) -> int:
    """Bridge official 5-level ratings into an ordinal rank for backtesting only."""
    order = {
        "BUY": 4,
        "OVERWEIGHT": 3,
        "HOLD": 2,
        "UNDERWEIGHT": 1,
        "SELL": 0,
    }
    return order.get(rating.upper(), 2)


def load_price_frame(ticker: str) -> pd.DataFrame:
    """Load and normalize merged OHLCV frames from local experiment caches."""
    frames: List[pd.DataFrame] = []
    for pat in PRICE_CANDIDATES:
        path = pat.format(ticker=ticker)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "date" not in df.columns:
            continue
        close_col = "adjusted_close" if "adjusted_close" in df.columns else "close"
        if close_col not in df.columns:
            continue
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(df["date"], errors="coerce"),
                "Open": pd.to_numeric(df.get("open"), errors="coerce"),
                "High": pd.to_numeric(df.get("high"), errors="coerce"),
                "Low": pd.to_numeric(df.get("low"), errors="coerce"),
                "Close": pd.to_numeric(df.get(close_col), errors="coerce"),
                "Adj Close": pd.to_numeric(df.get(close_col), errors="coerce"),
                "Raw Close": pd.to_numeric(
                    df.get("raw_close", df.get(close_col)), errors="coerce"
                ),
                "Volume": pd.to_numeric(df.get("volume"), errors="coerce").fillna(0),
            }
        )
        frame = frame.dropna(subset=["Date", "Close"])
        frames.append(frame)

    if not frames:
        raise RuntimeError(f"No price files found for {ticker}")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")
    merged["Volume"] = merged["Volume"].astype(int)
    return merged.reset_index(drop=True)


def seed_tradingagents_yfinance_history_cache(tickers: List[str]) -> List[str]:
    """Pre-populate TradingAgents' yfinance history cache from local price snapshots."""
    config = get_config()
    cache_dir = Path(config["data_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    today_date = pd.Timestamp.today()
    start_date = today_date - pd.DateOffset(years=15)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = today_date.strftime("%Y-%m-%d")

    seeded: List[str] = []
    for ticker in tickers:
        frame = load_price_frame(ticker)
        out_path = cache_dir / f"{ticker.upper()}-YFin-data-{start_date_str}-{end_date_str}.csv"
        frame.to_csv(out_path, index=False)
        seeded.append(str(out_path))
    return seeded


def prewarm_yfinance_text_cache(tickers: List[str]) -> List[dict[str, str]]:
    """Pre-populate slow yfinance text endpoints into vendor_cache before graph runs."""
    results: List[dict[str, str]] = []
    for ticker in tickers:
        tasks = [
            ("fundamentals", lambda t=ticker: get_yf_fundamentals(t)),
            ("balance_sheet_q", lambda t=ticker: get_yf_balance_sheet(t, "quarterly")),
            ("cashflow_q", lambda t=ticker: get_yf_cashflow(t, "quarterly")),
            ("income_statement_q", lambda t=ticker: get_yf_income_statement(t, "quarterly")),
            ("insider_transactions", lambda t=ticker: get_yf_insider_transactions(t)),
        ]
        for name, func in tasks:
            payload = func()
            status = "ok"
            if isinstance(payload, str) and payload.startswith("Error "):
                status = "error"
            results.append(
                {
                    "ticker": ticker,
                    "dataset": name,
                    "status": status,
                }
            )
    return results


def load_prices(ticker: str) -> pd.Series:
    frame = load_price_frame(ticker)
    close_series = frame["Raw Close"] if "Raw Close" in frame.columns else frame["Close"]
    s = pd.Series(close_series.values, index=frame["Date"])
    return s


class RunnableLocalQwen(Runnable):
    def __init__(self, inner):
        self.inner = inner

    def bind_tools(self, tools):
        return self

    def invoke(self, input, config=None, **kwargs):
        if hasattr(input, "to_messages"):
            input = input.to_messages()
        return self.inner.invoke(input)


class LocalQwenClientAdapter:
    _cache: Dict[str, RunnableLocalQwen] = {}

    def __init__(self, model: str, base_url=None, **kwargs):
        self.model = model

    def get_llm(self):
        if self.model not in self._cache:
            spec = importlib.util.spec_from_file_location(
                "local_qwen_mod",
                "/root/private_data/tradingagents_tsfm_modified_v5/tradingagents/llms/local_qwen.py",
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            local = mod.LocalQwenChat(
                model_name=self.model,
                temperature=0.0,
                max_tokens=get_env_int("TRADINGAGENTS_LOCAL_QWEN_MAX_TOKENS", 20000),
            )
            self._cache[self.model] = RunnableLocalQwen(local)
        return self._cache[self.model]

    def validate_model(self):
        return True


def patched_create_llm_client(provider: str, model: str, base_url=None, **kwargs):
    return LocalQwenClientAdapter(model=model, base_url=base_url, **kwargs)


@dataclass
class MonthResult:
    month: str
    trading_days: int
    final_value: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float


def compute_max_drawdown(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    running_max = np.maximum.accumulate(arr)
    drawdowns = (arr - running_max) / running_max
    return float(drawdowns.min())


def run_month(month: str, start: str, end: str, ta, price_map: Dict[str, pd.Series], decisions_writer) -> MonthResult:
    all_days = sorted(
        d
        for d in price_map[TICKERS[0]].index
        if start <= d <= end
        and all(d in price_map[t].index for t in TICKERS)
    )
    if not all_days:
        raise RuntimeError(f"No trading days for {month}")

    first_day = all_days[0]
    shares = {
        t: (INITIAL_CAPITAL / len(TICKERS)) / float(price_map[t][first_day])
        for t in TICKERS
    }

    portfolio_values: List[float] = []
    daily_returns: List[float] = []
    prev_value = INITIAL_CAPITAL

    for i, day in enumerate(all_days, start=1):
        prices = {t: float(price_map[t][day]) for t in TICKERS}
        value = float(sum(shares[t] * prices[t] for t in TICKERS))
        portfolio_values.append(value)

        dr = (value - prev_value) / prev_value if prev_value > 0 else 0.0
        daily_returns.append(float(dr))
        prev_value = value

        ratings = {}
        for ticker_idx, t in enumerate(TICKERS, start=1):
            print(
                f"[{month}] day {i}/{len(all_days)} {day} ticker {ticker_idx}/{len(TICKERS)} {t} start",
                flush=True,
            )
            _, signal = ta.propagate(t, day)
            rating = extract_rating(signal)
            ratings[t] = rating
            decisions_writer.writerow([month, day, t, rating, signal])
            print(
                f"[{month}] day {i}/{len(all_days)} {day} ticker {ticker_idx}/{len(TICKERS)} {t} done rating={rating}",
                flush=True,
            )

        raw_weights = {t: float(official_rating_rank(ratings[t])) for t in TICKERS}
        weight_sum = sum(raw_weights.values())
        if weight_sum <= 0:
            target_weights = {t: 1.0 / len(TICKERS) for t in TICKERS}
        else:
            target_weights = {t: raw_weights[t] / weight_sum for t in TICKERS}

        shares = {
            t: (value * target_weights[t]) / prices[t] if prices[t] > 0 else 0.0
            for t in TICKERS
        }

        print(f"[{month}] {i}/{len(all_days)} {day} value={value:.2f} ratings={ratings}", flush=True)

    final_value = portfolio_values[-1]
    total_return = final_value / INITIAL_CAPITAL - 1.0
    annualized = (1.0 + total_return) ** (252.0 / max(len(all_days), 1)) - 1.0

    ret_arr = np.array(daily_returns[1:], dtype=float) if len(daily_returns) > 1 else np.array([], dtype=float)
    if ret_arr.size > 1 and float(np.std(ret_arr)) > 0:
        sharpe = float(math.sqrt(252.0) * np.mean(ret_arr) / np.std(ret_arr))
    else:
        sharpe = 0.0

    mdd = compute_max_drawdown(portfolio_values)
    return MonthResult(
        month=month,
        trading_days=len(all_days),
        final_value=float(final_value),
        total_return=float(total_return),
        annualized_return=float(annualized),
        sharpe_ratio=float(sharpe),
        max_drawdown=float(mdd),
    )


def main() -> None:
    tg.create_llm_client = patched_create_llm_client

    cfg = DEFAULT_CONFIG.copy()
    cfg["llm_provider"] = "openai"
    cfg["deep_think_llm"] = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    cfg["quick_think_llm"] = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    # 恢复为仓库默认轮数。
    # 这两个值之前被手工改成 0 / 0，会让这条 harness 跳过 Bear Researcher、
    # Conservative Analyst、Neutral Analyst，使结果不再代表默认 TradingAgents 流程。
    # DEFAULT_CONFIG 里的默认值是 1 / 1，因此这里显式对齐回默认配置。
    cfg["max_debate_rounds"] = get_env_int(
        "TRADINGAGENTS_MAX_DEBATE_ROUNDS",
        DEFAULT_CONFIG["max_debate_rounds"],
    )
    cfg["max_risk_discuss_rounds"] = get_env_int(
        "TRADINGAGENTS_MAX_RISK_DISCUSS_ROUNDS",
        DEFAULT_CONFIG["max_risk_discuss_rounds"],
    )
    # Keep repository-default vendors here. Forcing every tool through Alpha Vantage
    # makes smoke tests brittle because some endpoints are premium and the free quota
    # is too small for a full multi-agent run.
    set_config(cfg)

    seeded_paths = seed_tradingagents_yfinance_history_cache(TICKERS)
    print(
        f"[CACHE] Seeded TradingAgents yfinance history cache for {len(seeded_paths)} tickers",
        flush=True,
    )
    text_cache_results = prewarm_yfinance_text_cache(TICKERS)
    text_cache_ok = sum(item["status"] == "ok" for item in text_cache_results)
    text_cache_err = sum(item["status"] != "ok" for item in text_cache_results)
    print(
        f"[CACHE] Prewarmed yfinance text cache entries: ok={text_cache_ok} error={text_cache_err}",
        flush=True,
    )

    ta = tg.TradingAgentsGraph(debug=False, config=cfg)
    price_map = {t: load_prices(t) for t in TICKERS}

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"/root/private_data/logs/tradingagents_official_3months_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions_csv = out_dir / "decisions.csv"
    summary_json = out_dir / "summary.json"

    results: List[MonthResult] = []
    with decisions_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["month", "date", "ticker", "rating", "signal_raw"])
        for month, (start, end) in MONTHS.items():
            print(f"=== Running month {month} ({start} to {end}) ===", flush=True)
            r = run_month(month, start, end, ta, price_map, writer)
            results.append(r)
            print(f"=== Done {month}: total={r.total_return:.4%}, sharpe={r.sharpe_ratio:.4f} ===", flush=True)

    payload = {
        "run_id": run_id,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "config": {
            "tickers": TICKERS,
            "months": MONTHS,
            "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "max_tokens": get_env_int("TRADINGAGENTS_LOCAL_QWEN_MAX_TOKENS", 20000),
            "max_debate_rounds": cfg["max_debate_rounds"],
            "max_risk_discuss_rounds": cfg["max_risk_discuss_rounds"],
        },
        "monthly_results": [r.__dict__ for r in results],
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"SUMMARY_JSON={summary_json}", flush=True)
    print(f"DECISIONS_CSV={decisions_csv}", flush=True)


if __name__ == "__main__":
    main()
