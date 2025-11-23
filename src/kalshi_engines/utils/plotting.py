"""Reusable plotting to avoid code duplication."""
import matplotlib.pyplot as plt
from kalshi_engines.constants import PLOTS_DIR

# Generic function to save distribution plots


def save_distribution(data, title: str, filename: str = "demo.png"):
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=60, alpha=0.75, color="#1f77b4", edgecolor="black")
    plt.axvline(data.mean(), color="red", linestyle="--",
                label=f"Mean = {data.mean():.3f}")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename)
    plt.close()
