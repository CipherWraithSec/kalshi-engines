from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor

from kalshi_engines.agents.base import BaseAgent
from kalshi_engines.config import DATA_SAMPLES_DIR
from kalshi_engines.utils.data import fetch_with_retry


class MentionAgent(BaseAgent):
    def __init__(self, demo: bool = False):
        super().__init__(name="mention", demo=demo)

    def train_model(self):
        # In real version: load full 2015–2025 resolved mentions
        # For demo: use tiny sample
        df = pd.read_csv(DATA_SAMPLES_DIR /
                         "mention_2024_sample.csv", parse_dates=["date"])
        df["sentiment_ewm"] = df.groupby("speaker")["sentiment"].transform(
            lambda x: x.ewm(span=14).mean()
        )
        X = df[["hsfm_score", "cim_score", "rnmm_intensity", "sentiment_ewm"]]
        y = df["actual_mention"]
        model = XGBRegressor(n_estimators=400, max_depth=6, random_state=42)
        model.fit(X, y)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)

    def fetch_current_features(self):
        if self.demo:
            return pd.Series({"hsfm_score": 0.48, "cim_score": 0.61, "rnmm_intensity": 0.72, "sentiment_ewm": 0.55})
        # Real version: fetch last 30 days transcripts + 14 days news
        # Placeholder
        return pd.Series({"hsfm_score": 0.50, "cim_score": 0.62, "rnmm_intensity": 0.70, "sentiment_ewm": 0.58})

    def generate_signal(self, prob: float):
        kalshi_price = 0.54  # In real: API call
        edge = prob - kalshi_price
        return {
            "engine": "mention",
            "predicted_prob": round(prob, 4),
            "kalshi_price": kalshi_price,
            "edge": round(edge, 4),
            "recommendation": "BUY YES" if edge > 0.05 else "PASS",
        }
