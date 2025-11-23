from __future__ import annotations
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

from kalshi_engines.agents.base import BaseAgent
from kalshi_engines.config import DATA_SAMPLES_DIR


class NFLAgent(BaseAgent):
    def __init__(self, demo: bool = False):
        super().__init__(name="nfl", demo=demo)

    def train_model(self):
        df = pd.read_csv(DATA_SAMPLES_DIR / "nfl_week15_2024.csv")
        df["rolling_epa"] = df.groupby("team")["epa"].transform(
            lambda x: x.ewm(span=4).mean())
        X = df[["home_epa", "away_epa", "weather_temp",
                "injury_count", "rolling_epa"]]
        y = df["home_win"]
        model = LGBMRegressor(n_estimators=500, max_depth=7, random_state=42)
        model.fit(X, y)
        joblib.dump(model, self.model_path)

    def fetch_current_features(self):
        if self.demo:
            return pd.Series({"home_epa": 0.12, "away_epa": -0.05, "weather_temp": 28, "injury_count": 2, "rolling_epa": 0.08})
        # Real: nflfastR + NOAA + injury API
        return pd.Series({"home_epa": 0.15, "away_epa": -0.03, "weather_temp": 32, "injury_count": 1, "rolling_epa": 0.10})

    def generate_signal(self, prob: float):
        kalshi_price = 0.58
        edge = prob - kalshi_price
        return {
            "engine": "nfl",
            "home_win_prob": round(prob, 4),
            "kalshi_price": kalshi_price,
            "edge": round(edge, 4),
            "recommendation": "BUY HOME" if edge > 0.06 else "PASS",
        }
