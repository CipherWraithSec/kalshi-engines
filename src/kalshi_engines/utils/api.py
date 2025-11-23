"""Robust API fetching with retries and exponential backoff."""
import time
import requests
import logging
from functools import lru_cache
from typing import Any, Dict

log = logging.getLogger(__name__)


def fetch_with_retry(url: str, params: Dict | None = None, retries: int = 4) -> Any:
    """
    Fetch JSON from URL with exponential backoff.
    Used for Kalshi, NOAA, EIA, nflfastR, etc.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=12)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            wait = 2 ** attempt
            log.warning(
                f"API failed (attempt {attempt}/{retries}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Permanent failure fetching {url}")


@lru_cache(maxsize=128)
def get_kalshi_market_price(ticker: str) -> float:
    """Cached wrapper — avoids hammering Kalshi API during backtesting."""
    # In real version: call Kalshi public API
    return 0.54  # placeholder
