import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows import alpha_vantage_common
from tradingagents.dataflows.alpha_vantage_common import _make_api_request
from tradingagents.dataflows.cache_utils import get_or_fetch_cached_text, save_cached_text
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.yfinance_news import get_news_yfinance


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


if __name__ == "__main__":
    unittest.main()
