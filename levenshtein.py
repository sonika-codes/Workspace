from rapidfuzz.distance import Levenshtein
from scipy.stats import zscore
import pandas as pd

# --- Levenshtein similarity (normalized) ---
def levenshtein_similarity(a, b):
    max_len = max(len(a), len(b))
    return 1 - (Levenshtein.distance(a, b) / max_len) if max_len else 1.0

# --- Weighted average similarity to historical commands ---
def weighted_avg_similarity(cmd, history_counts):
    sims, weights = [], []
    for hist_cmd, freq in history_counts.items():
        sim = levenshtein_similarity(cmd, hist_cmd)
        sims.append(sim * freq)
        weights.append(freq)
    return sum(sims) / sum(weights) if weights else 0.0

# --- Main function: score anomalies for one user ---
def score_anomalies_for_user(user, user_df, history_counts, z_thresh=-2):
    """
    Inputs:
        user          - string (user ID)
        user_df       - DataFrame with at least 'cleaned_command' column, rows for this user
        history_counts - dict of {command: frequency} for this user
        z_thresh      - z-score threshold to flag anomalies

    Returns:
        DataFrame with added columns: 'weighted_avg_similarity', 'z_score', 'anomaly'
    """
    cmds = user_df['cleaned_command'].tolist()
    sims = []

    for cmd in cmds:
        if cmd in history_counts:
            sims.append(1.0)  # exact match
        else:
            sims.append(weighted_avg_similarity(cmd, history_counts))

    user_df = user_df.copy()
    user_df['weighted_avg_similarity'] = sims

    # z-score across this user's commands
    user_df['z_score'] = zscore(sims) if len(sims) > 1 else [0.0] * len(sims)
    user_df['anomaly'] = user_df['z_score'] < z_thresh
    user_df['user'] = user  # ensure user column exists if not already
    return user_df
