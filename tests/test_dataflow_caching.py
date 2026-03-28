import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows import alpha_vantage_common
from tradingagents.dataflows.alpha_vantage_common import _make_api_request
from tradingagents.dataflows.cache_utils import get_or_fetch_cached_text, save_cached_text
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.yfinance_news import get_news_yfinance
from tradingagents.dataflows.y_finance import (
    get_YFin_data_online,
    warm_yfinance_history_cache,
)


class DataflowCachingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_config = DEFAULT_CONFIG.copy()
        self.original_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")

    def tearDown(self) -> None:
        set_config(self.original_config)
        if self.original_api_key is None:
            os.environ.pop("ALPHA_VANTAGE_API_KEY", None)
        else:
            os.environ["ALPHA_VANTAGE_API_KEY"] = self.original_api_key

    def test_get_or_fetch_cached_text_uses_fresh_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_config(
                {
                    "data_cache_dir": tmpdir,
                    "data_cache_ttl_seconds": 3600,
                }
            )
            key = {"ticker": "AAPL"}
            save_cached_text("unit", key, "cached-value")

            result = get_or_fetch_cached_text(
                "unit",
                key,
                lambda: self.fail("fetch should not run when cache is fresh"),
            )

            self.assertEqual(result, "cached-value")

    def test_get_or_fetch_cached_text_uses_stale_cache_on_fetch_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_config(
                {
                    "data_cache_dir": tmpdir,
                    "data_cache_ttl_seconds": 1,
                }
            )
            key = {"ticker": "AAPL"}
            save_cached_text("unit", key, "stale-value")

            cache_file = next((Path(tmpdir) / "vendor_cache" / "unit").iterdir())
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            payload["stored_at"] = time.time() - 10
            cache_file.write_text(json.dumps(payload), encoding="utf-8")

            result = get_or_fetch_cached_text(
                "unit",
                key,
                lambda: (_ for _ in ()).throw(RuntimeError("network down")),
                ttl_seconds=1,
                fallback_exceptions=(RuntimeError,),
            )

            self.assertEqual(result, "stale-value")

    def test_alpha_vantage_request_falls_back_to_stale_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_config(
                {
                    "data_cache_dir": tmpdir,
                    "data_cache_ttl_seconds": 1,
                }
            )
            os.environ["ALPHA_VANTAGE_API_KEY"] = "demo"

            good_response = Mock()
            good_response.raise_for_status.return_value = None
            good_response.text = "timestamp,close\n2025-09-02,100\n"

            with patch.object(alpha_vantage_common.requests, "get", return_value=good_response):
                first = _make_api_request(
                    "TIME_SERIES_DAILY_ADJUSTED",
                    {"symbol": "AAPL", "datatype": "csv"},
                )

            self.assertIn("2025-09-02", first)

            cache_dir = Path(tmpdir) / "vendor_cache" / "alpha_vantage"
            cache_file = next(cache_dir.iterdir())
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            payload["stored_at"] = time.time() - 10
            cache_file.write_text(json.dumps(payload), encoding="utf-8")

            rate_limited = Mock()
            rate_limited.raise_for_status.return_value = None
            rate_limited.text = json.dumps(
                {"Information": "Thank you for using Alpha Vantage! Please retry later due to rate limit."}
            )

            with patch.object(alpha_vantage_common.requests, "get", return_value=rate_limited):
                second = _make_api_request(
                    "TIME_SERIES_DAILY_ADJUSTED",
                    {"symbol": "AAPL", "datatype": "csv"},
                )

            self.assertEqual(second, first)

    def test_yfinance_news_uses_cached_rendered_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_config(
                {
                    "data_cache_dir": tmpdir,
                    "data_cache_ttl_seconds": 3600,
                }
            )

            article = {
                "content": {
                    "title": "Apple launches update",
                    "summary": "Summary text",
                    "provider": {"displayName": "Example News"},
                    "canonicalUrl": {"url": "https://example.com/article"},
                    "pubDate": "2025-09-02T08:00:00Z",
                }
            }

            ticker_obj = Mock()
            ticker_obj.get_news.return_value = [article]

            with patch("tradingagents.dataflows.yfinance_news.yf.Ticker", return_value=ticker_obj):
                first = get_news_yfinance("AAPL", "2025-09-01", "2025-09-02")

            with patch(
                "tradingagents.dataflows.yfinance_news.yf.Ticker",
                side_effect=RuntimeError("should not refetch"),
            ):
                second = get_news_yfinance("AAPL", "2025-09-01", "2025-09-02")

            self.assertEqual(first, second)
            self.assertIn("Apple launches update", second)

    def test_yfinance_history_does_not_cache_empty_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_config(
                {
                    "data_cache_dir": tmpdir,
                    "data_cache_ttl_seconds": 3600,
                }
            )

            empty_history = pd.DataFrame()
            with patch("tradingagents.dataflows.y_finance.yf.download", return_value=empty_history):
                result = get_YFin_data_online("AAPL", "2025-09-01", "2025-09-05")

            self.assertIn("No data found", result)
            self.assertEqual(list(Path(tmpdir).glob("*YFin-data-*.csv")), [])

    def test_warm_yfinance_history_cache_creates_shared_history_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_config(
                {
                    "data_cache_dir": tmpdir,
                    "data_cache_ttl_seconds": 3600,
                }
            )

            sample = pd.DataFrame(
                {
                    "Date": ["2025-09-02", "2025-09-03"],
                    "Open": [100.0, 101.0],
                    "High": [101.0, 102.0],
                    "Low": [99.0, 100.0],
                    "Close": [100.5, 101.5],
                    "Volume": [1000, 1200],
                }
            )

            with patch("tradingagents.dataflows.y_finance.yf.download", return_value=sample):
                first = warm_yfinance_history_cache(["AAPL"])

            self.assertEqual(first[0]["status"], "fetched")
            self.assertEqual(first[0]["rows"], 2)
            self.assertTrue(Path(first[0]["path"]).exists())

            with patch(
                "tradingagents.dataflows.y_finance.yf.download",
                side_effect=RuntimeError("should not redownload"),
            ):
                second = warm_yfinance_history_cache(["AAPL"])

            self.assertEqual(second[0]["status"], "hit")
            self.assertEqual(second[0]["rows"], 2)

    def test_warm_yfinance_history_cache_replaces_invalid_empty_cache_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_config(
                {
                    "data_cache_dir": tmpdir,
                    "data_cache_ttl_seconds": 3600,
                }
            )

            stale_path = (
                Path(tmpdir)
                / "AAPL-YFin-data-2011-03-28-2026-03-28.csv"
            )
            pd.DataFrame(columns=["Date", "Close"]).to_csv(stale_path, index=False)

            sample = pd.DataFrame(
                {
                    "Date": ["2025-09-02", "2025-09-03"],
                    "Open": [100.0, 101.0],
                    "High": [101.0, 102.0],
                    "Low": [99.0, 100.0],
                    "Close": [100.5, 101.5],
                    "Volume": [1000, 1200],
                }
            )

            with patch("tradingagents.dataflows.y_finance.yf.download", return_value=sample):
                result = warm_yfinance_history_cache(["AAPL"])

            self.assertEqual(result[0]["rows"], 2)
            self.assertIn(result[0]["status"], {"fetched", "hit"})
            self.assertTrue(Path(result[0]["path"]).exists())

            reloaded = pd.read_csv(result[0]["path"])
            self.assertEqual(len(reloaded), 2)


if __name__ == "__main__":
    unittest.main()
