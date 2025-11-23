"""All feature engineering in one place — keeps agents clean."""
import pandas as pd

# Feature engineering functions used across multiple agents


def add_ewm_recency(df: pd.DataFrame, group_col: str, value_col: str, span: int) -> pd.Series:
    """Exponential weighted moving feature — used for news & EPA."""
    return df.groupby(group_col)[value_col].transform(lambda x: x.ewm(span=span).mean())
