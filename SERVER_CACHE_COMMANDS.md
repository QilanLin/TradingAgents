# TradingAgents Server Cache Commands

Use this helper to prewarm shared `yfinance` history cache before smoke tests or
full multi-day runs.

## Warm MAG7 history cache

```bash
cd /root/private_data/TradingAgents
export PYTHONPATH=.
/root/private_data/.venv_timesfm25/bin/python warm_yfinance_cache.py
```

## Warm specific tickers

```bash
cd /root/private_data/TradingAgents
export PYTHONPATH=.
/root/private_data/.venv_timesfm25/bin/python warm_yfinance_cache.py \
  --tickers AAPL MSFT NVDA
```

## Warm a custom shared cache directory

```bash
cd /root/private_data/TradingAgents
export PYTHONPATH=.
/root/private_data/.venv_timesfm25/bin/python warm_yfinance_cache.py \
  --cache-dir /root/private_data/tradingagents_cache \
  --tickers AAPL GOOGL AMZN MSFT META TSLA NVDA
```

## JSON output for scripting

```bash
cd /root/private_data/TradingAgents
export PYTHONPATH=.
/root/private_data/.venv_timesfm25/bin/python warm_yfinance_cache.py --json
```
