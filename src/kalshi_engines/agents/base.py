"""Base class for all engines — enforces consistent interface."""
from __future__ import annotations
import joblib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np

from kalshi_engines.constants import MODELS_DIR

log = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Every engine inherits from this.
    Guarantees: train once, load fast, zero lookahead, clean signal output.
    """
    NAME: str  # Must be defined in child class

    def __init__(self, demo: bool = False):
        self.demo = demo
        self.model_path = MODELS_DIR / f"{self.NAME}_2015_2025.pkl"
        self.model = self._load_or_train_model()

    def _load_or_train_model(self):
        if self.model_path.exists():
            log.info(f"Loaded {self.NAME} model from disk")
            return joblib.load(self.model_path)
        else:
            log.info(f"No model found — training {self.NAME} on full history")
            model = self.train_model()
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, self.model_path)
            log.info(f"Saved {self.NAME} model to {self.model_path}")
            return model

    @abstractmethod
    def train_model(self):
        """Train on full resolved 2015–2025 data."""
        pass

    @abstractmethod
    def fetch_live_features(self) -> pd.Series:
        """Only data available RIGHT NOW — no future leak."""
        pass

    @abstractmethod
    def generate_signal(self, prediction: float) -> dict:
        """Return human + machine readable dict."""
        pass

    def run(self) -> dict:
        """Public method called by orchestrator."""
        features = self.fetch_live_features()
        pred = float(self.model.predict(features.to_frame().T)[0])
        return self.generate_signal(pred)
