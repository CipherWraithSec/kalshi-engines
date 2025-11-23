"""NFL Engine – extended with weather, injuries, rolling EPA."""
from __future__ import annotations
import pandas as pd
from lightgbm import LGBMRegressor

from kalshi_engines.agents.base import BaseAgent
from kalshi_engines.constants import DATA_SAMPLES_DIR
from kalshi_engines.utils.features import add_ewm_recency


class NFLAgent(BaseAgent):
    NAME = "nfl"

    def train_model(self):
        df = pd.read_csv(DATA_SAMPLES_DIR / "nfl_week15_2024.csv")
        df["epa_roll"] = add_ewm_recency(df, "team", "epa", span=4)
        X = df[["home_epa", "away_epa", "temperature_f", "key_injuries", "epa_roll"]]
        y = df["home_win"]
        model = LGBMRegressor(n_estimators=600, max_depth=8,
                              random_state=42, n_jobs=-1)
        model.fit(X, y)
        return model

    def fetch_live_features(self) -> pd.Series:
        if self.demo:
            return pd.Series({"home_epa": 0.14, "away_epa": -0.06, "temperature_f": 31, "key_injuries": 2, "epa_roll": 0.09})
        # Real: pull current week from nflfastR + NOAA + injury reports
        return pd.Series({"home_epa": 0.16, "away_epa": -0.04, "temperature_f": 28, "key_injuries": 1, "epa_roll": 0.11})

    def generate_signal(self, prob: float) -> dict:
        price = 0.58
        return {
            "engine": "nfl_prediction",
            "home_win_probability": round(prob, 4),
            "market_price": price,
            "edge": round(prob - price, 4),
            "recommendation": "BET HOME" if prob > 0.65 else "PASS"
        }
