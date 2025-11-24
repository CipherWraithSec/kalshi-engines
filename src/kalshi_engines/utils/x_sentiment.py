"""
X (Twitter) sentiment booster for Mention RNMM.
Uses local DistilBERT — zero cost, zero non-determinism, runs offline.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List

import numpy as np
from transformers import pipeline

log = logging.getLogger(__name__)

# Load once at import time — fast subsequent calls
_sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,  # CPU only
    truncation=True,
    max_length=512
)


def get_x_sentiment_score(posts: List[str]) -> float:
    """
    Returns average sentiment score in [-1, 1] range.
    Empty list → 0.0 (neutral).
    """
    if not posts:
        return 0.0

    results = _sentiment_pipeline(posts)
    scores = []
    for r in results:
        score = r["score"]
        if r["label"] == "NEGATIVE":
            score = -score
        scores.append(score)

    return float(np.mean(scores))


def mock_x_search(query: str, days: int = 14, limit: int = 80) -> List[str]:
    """
    Placeholder for real X search.
    In production: use your tool call or snscrape/Twitter API v2.
    For submission: returns realistic mock posts.
    """
    # These are real-style X posts from 2024 Trump coverage
    mock_posts = [
        "Trump just crushed that speech. Mention incoming.",
        "No way he says the word tonight. Too risky.",
        "Everyone knows he’ll mention it. Book it.",
        "Media trying to bait him again. He won’t fall for it.",
        "If he doesn’t say it, I’ll be shocked.",
        "He literally can’t help himself lol",
        "Vegas has it at 68 %. Easy money.",
        "CNN already writing the outrage piece just in case"
    ]
    return mock_posts[:limit]
