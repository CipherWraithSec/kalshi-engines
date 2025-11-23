"""Global constants used across the entire project."""
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_SAMPLES_DIR = BASE_DIR / "data_samples"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Model training years (full resolved history)
TRAIN_YEARS = slice("2015-01-01", "2025-12-31")

# Recency windows
EWM_SPAN_NEWS = 14        # 14-day exponential moving average for news sentiment
EWM_SPAN_EPA = 4           # 4-week rolling EPA (NFL)
