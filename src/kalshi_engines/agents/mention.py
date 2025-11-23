"""Mention Engine – heavily extended from Notion template."""
from __future__ import annotations
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from kalshi_engines.agents.base import BaseAgent
from kalshi_engines.constants import DATA_SAMPLES_DIR
from kalshi_engines.utils.features import add_ewm_recency
from kalshi_engines.utils.api import get_kalshi_market_price


class MentionAgent(BaseAgent):
    NAME = "mention"

    def train_model(self):
        # In real project: load full 2015–2025 resolved mentions
        df = pd.read_csv(DATA_SAMPLES_DIR /
                         "mention_2024_sample.csv", parse_dates=["date"])

        # === Your high-agency extension: time-series aware recency ===
        df["sentiment_ewm"] = add_ewm_recency(
            df, "speaker", "sentiment", span=14)

        X = df[["hsfm_score", "cim_score", "rnmm_intensity", "sentiment_ewm"]]
        y = df["actual_mention"]

        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        return model

    def fetch_live_features(self) -> pd.Series:
        if self.demo:
            return pd.Series({
                "hsfm_score": 0.505,
                "cim_score": 0.623,
                "rnmm_intensity": 0.712,
                "sentiment_ewm": 0.581
            })
        # Real implementation would fetch last 30 days of speeches + news
        # and compute ewm on-the-fly
        return pd.Series({"hsfm_score": 0.51, "cim_score": 0.63, "rnmm_intensity": 0.72, "sentiment_ewm": 0.59})

    def generate_signal(self, prob: float) -> dict:
        kalshi_price = get_kalshi_market_price("TRUMP-MENTION-2025")
        edge = prob - kalshi_price
        return {
            "engine": "mention_prediction",
            "predicted_probability": round(prob, 4),
            "kalshi_yes_price": kalshi_price,
            "edge": round(edge, 4),
            "recommendation": "STRONG BUY YES" if edge > 0.07 else "BUY YES" if edge > 0.03 else "PASS",
            "confidence": "high" if abs(edge) > 0.08 else "medium"
        }
