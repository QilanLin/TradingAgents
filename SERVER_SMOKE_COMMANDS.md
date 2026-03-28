# TradingAgents Server Smoke Commands

This repo now has a reliable server-side smoke entry for the local-Qwen path.

## Preferred smoke

Use the analyst-only smoke first. It verifies:

- server can import the repo
- local Qwen can load
- tool execution works
- market / social / news / fundamentals reports are generated

```bash
cd /root/private_data/TradingAgents
export ALPHA_VANTAGE_API_KEY='YOUR_KEY'
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export HF_HOME=/root/private_data/.cache
export HUGGINGFACE_HUB_CACHE=/root/private_data/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRADINGAGENTS_LOCAL_QWEN_MAX_TOKENS=64
export TRADINGAGENTS_MAX_DEBATE_ROUNDS=0
export TRADINGAGENTS_MAX_RISK_DISCUSS_ROUNDS=0
timeout 180 /root/private_data/.venv_timesfm25/bin/python smoke_local_qwen_analysts.py
```

## Why this smoke is preferred

The full graph smoke is much heavier because it includes:

- full multi-node propagation
- repeated LLM generations
- debate / risk paths

That path is valid for real experiments, but it is a poor health check because it is easy
to hit timeouts even when the code is functioning normally.

## Full graph smoke

Only use this if the analyst-only smoke already passes and you explicitly want to test
end-to-end graph propagation.

```bash
cd /root/private_data/TradingAgents
export ALPHA_VANTAGE_API_KEY='YOUR_KEY'
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export HF_HOME=/root/private_data/.cache
export HUGGINGFACE_HUB_CACHE=/root/private_data/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRADINGAGENTS_LOCAL_QWEN_MAX_TOKENS=64
export TRADINGAGENTS_MAX_DEBATE_ROUNDS=0
export TRADINGAGENTS_MAX_RISK_DISCUSS_ROUNDS=0
timeout 180 /root/private_data/.venv_timesfm25/bin/python - <<'PY'
from tradingagents.default_config import DEFAULT_CONFIG
import tradingagents.graph.trading_graph as tg
import run_mag7_3months_local_qwen as harness

tg.create_llm_client = harness.patched_create_llm_client
cfg = DEFAULT_CONFIG.copy()
cfg["llm_provider"] = "openai"
cfg["deep_think_llm"] = "Qwen/Qwen3-30B-A3B-Instruct-2507"
cfg["quick_think_llm"] = "Qwen/Qwen3-30B-A3B-Instruct-2507"
cfg["max_debate_rounds"] = harness.get_env_int(
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS",
    DEFAULT_CONFIG["max_debate_rounds"],
)
cfg["max_risk_discuss_rounds"] = harness.get_env_int(
    "TRADINGAGENTS_MAX_RISK_DISCUSS_ROUNDS",
    DEFAULT_CONFIG["max_risk_discuss_rounds"],
)

ta = tg.TradingAgentsGraph(debug=False, config=cfg)
state, signal = ta.propagate("AAPL", "2025-09-02")
print(signal[:400].replace("\n", " "))
print(len(state.get("market_report", "")))
print(len(state.get("sentiment_report", "")))
print(len(state.get("news_report", "")))
print(len(state.get("fundamentals_report", "")))
PY
```
