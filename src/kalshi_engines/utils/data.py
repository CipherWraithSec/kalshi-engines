import requests
import time
import logging

log = logging.getLogger(__name__)

# Utility function to fetch data with retries


def fetch_with_retry(url: str, retries: int = 3, backoff: int = 2):
    for i in range(retries):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning(f"Fetch failed ({i+1}/{retries}): {e}")
            time.sleep(backoff * (2 ** i))
    raise RuntimeError(f"Failed to fetch {url}")
