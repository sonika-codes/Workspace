import pandas as pd
import numpy as np
from google.cloud import bigquery
import matplotlib.pyplot as plt

TABLE_ID = "your-project.your_dataset.unified"  # <-- set this

def load_unified_bq() -> pd.DataFrame:
    client = bigquery.Client()
    q = f"SELECT * FROM `{TABLE_ID}`"
    return client.query(q).result().to_dataframe(create_bqstorage_client=True)

def _pivot_user_hour(df: pd.DataFrame) -> pd.DataFrame:
    # Produces a matrix: index=user, columns=1..168, values=metric_value
    mat = (
        df.pivot_table(index="user", columns="index_value", values="metric_value", aggfunc="first")
          .reindex(columns=range(1,169))  # enforce 1..168 order
          .sort_index()
    )
    return mat

def build_average_matrix(unified: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    metric ∈ {'average total calls','average manual calls','average user status'}
    """
    avg = unified[unified["metric_name"]==metric].copy()
    # If a user/hour has no baseline row (brand new), fill with 0
    mat = _pivot_user_hour(avg).fillna(0.0)
    return mat

def build_deviation_matrix(unified: pd.DataFrame, day: str, metric: str) -> pd.DataFrame:
    """
    day: 'YYYY-MM-DD'
    metric ∈ {'total calls deviation','manual calls deviation','user status deviation'}
    """
    dev = unified[(unified["metric_name"]==metric) & (unified["date"]==pd.to_datetime(day))].copy()
    # If cold-start suppressed some rows, missing = 0 deviation (optional)
    mat = _pivot_user_hour(dev).fillna(0.0)
    return mat

def plot_heatmap(matrix: pd.DataFrame, title: str):
    plt.figure(figsize=(14, max(4, 0.3*len(matrix))))  # grow with users
    plt.imshow(matrix.values, aspect="auto")  # (default colormap, per your constraints)
    plt.title(title)
    plt.xlabel("Hour of week (1–168)")
    plt.ylabel("User")
    plt.yticks(ticks=range(len(matrix.index)), labels=matrix.index)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
