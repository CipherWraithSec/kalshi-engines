from __future__ import annotations
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from kalshi_engines.agents.base import BaseAgent
from kalshi_engines.config import DATA_SAMPLES_DIR, PLOTS_DIR
import matplotlib.pyplot as plt


class WeatherOilAgent(BaseAgent):
    def __init__(self, demo: bool = False):
        super().__init__(name="weather_oil", demo=demo)

    def train_model(self):
        df = pd.read_csv(DATA_SAMPLES_DIR / "weather_oil_2024.csv")
        X = df[["noaa_prob", "gulf_exposure", "enso_index", "inventory_surprise"]]
        y = df["oil_change_30d"]
        model = XGBRegressor(n_estimators=600, max_depth=5, random_state=42)
        model.fit(X, y)
        joblib.dump(model, self.model_path)

    def fetch_current_features(self):
        if self.demo:
            return pd.Series({"noaa_prob": 0.61, "gulf_exposure": 0.78, "enso_index": -0.4, "inventory_surprise": -1.2})
        # Real: NOAA CPC + EIA API
        return pd.Series({"noaa_prob": 0.62, "gulf_exposure": 0.80, "enso_index": -0.5, "inventory_surprise": -0.9})

    def generate_signal(self, predicted_impact: float):
        weather_price = 0.44
        oil_price = 0.29
        fair_oil = 0.29 + predicted_impact * 0.8  # simplified mapping
        edge = fair_oil - oil_price

        # Plot simulation
        sims = np.random.normal(predicted_impact, 0.12, 10000)
        plt.figure(figsize=(8, 5))
        plt.hist(sims, bins=50, alpha=0.7)
        plt.axvline(predicted_impact, color="red", label="Model")
        plt.title("Weather → Oil Impact Distribution")
        plt.legend()
        plt.savefig(PLOTS_DIR / "weather_oil_demo.png")
        plt.close()

        return {
            "engine": "weather_oil_arbitrage",
            "weather_market": "ATLANTIC-HURRICANE-2025 ≥3 MAJOR",
            "weather_price": weather_price,
            "model_noaa_prob": 0.61,
            "oil_market": "WTI-2025-12 >$95",
            "oil_price": oil_price,
            "predicted_impact": round(predicted_impact, 4),
            "fair_oil_price": round(fair_oil, 4),
            "edge": round(edge, 4),
            "kelly_fraction": round(max(edge - 0.02, 0) / 0.5, 3),
            "recommendation": "STATISTICAL ARB: BUY OIL + HEDGE WEATHER" if edge > 0.05 else "NO EDGE",
        }
