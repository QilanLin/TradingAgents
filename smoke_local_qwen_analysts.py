from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.analysts.social_media_analyst import (
    create_social_media_analyst,
)
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.analysts.fundamentals_analyst import (
    create_fundamentals_analyst,
)
import tradingagents.graph.trading_graph as tg
import run_mag7_3months_local_qwen as harness


TICKER = "AAPL"
TRADE_DATE = "2025-09-02"
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def make_state() -> dict:
    return {
        "trade_date": TRADE_DATE,
        "company_of_interest": TICKER,
        "messages": [],
    }


def report_preview(text: str, limit: int = 160) -> str:
    return (text or "").replace("\n", " ")[:limit]


def main() -> None:
    tg.create_llm_client = harness.patched_create_llm_client

    cfg = DEFAULT_CONFIG.copy()
    cfg["llm_provider"] = "openai"
    cfg["deep_think_llm"] = MODEL_NAME
    cfg["quick_think_llm"] = MODEL_NAME
    cfg["max_debate_rounds"] = harness.get_env_int(
        "TRADINGAGENTS_MAX_DEBATE_ROUNDS",
        DEFAULT_CONFIG["max_debate_rounds"],
    )
    cfg["max_risk_discuss_rounds"] = harness.get_env_int(
        "TRADINGAGENTS_MAX_RISK_DISCUSS_ROUNDS",
        DEFAULT_CONFIG["max_risk_discuss_rounds"],
    )

    print("[SMOKE-ANALYSTS] vendors", cfg["data_vendors"], flush=True)
    print(
        "[SMOKE-ANALYSTS] qwen_max_tokens",
        harness.get_env_int("TRADINGAGENTS_LOCAL_QWEN_MAX_TOKENS", 20000),
        flush=True,
    )

    llm = tg.create_llm_client(
        provider=cfg["llm_provider"],
        model=cfg["quick_think_llm"],
    ).get_llm()

    nodes = [
        ("market", "market_report", create_market_analyst(llm)),
        ("social", "sentiment_report", create_social_media_analyst(llm)),
        ("news", "news_report", create_news_analyst(llm)),
        ("fundamentals", "fundamentals_report", create_fundamentals_analyst(llm)),
    ]

    for name, report_key, node in nodes:
        print(f"[SMOKE-ANALYSTS] start {name}", flush=True)
        result = node(make_state())
        report = result.get(report_key, "")
        print(
            f"[SMOKE-ANALYSTS] done {name} len={len(report)} preview={report_preview(report)}",
            flush=True,
        )

    print("[SMOKE-ANALYSTS] done", flush=True)


if __name__ == "__main__":
    main()
