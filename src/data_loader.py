# -*- coding: utf-8 -*-
"""
Load the Dataset
"""

from pathlib import Path
import pandas as pd


def load_dataframe(data_path: str | Path) -> pd.DataFrame:
    """
    Load the benchmark dataframe.

    Expected columns include:
    - date
    - stock_id
    - feature_* columns
    """

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find dataframe CSV at: {data_path}"
        )

    return pd.read_csv(data_path, parse_dates=["date"])