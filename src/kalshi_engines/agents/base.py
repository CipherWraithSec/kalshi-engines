from __future__ import annotations
import logging
from pathlib import Path
import joblib

from kalshi_engines.config import MODELS_DIR

log = logging.getLogger(__name__)


class BaseAgent:
    def __init__(self, name: str, demo: bool = False):
        self.name = name
        self.demo = demo
        self.model_path = MODELS_DIR / f"{name}_2015_2025.pkl"
        self.model = self._load_or_train()

    def _load_or_train(self):
        if self.model_path.exists():
            log.info(f"Loading {self.name} model from {self.model_path}")
            return joblib.load(self.model_path)
        else:
            log.info(f"Training {self.name} model on full history...")
            self.train_model()
            return joblib.load(self.model_path)

    def train_model(self):
        raise NotImplementedError

    def fetch_current_features(self):
        raise NotImplementedError

    def run(self):
        features = self.fetch_current_features()
        prob = float(self.model.predict(features.values.reshape(1, -1))[0])
        return self.generate_signal(prob)

    def generate_signal(self, prob: float):
        raise NotImplementedError
