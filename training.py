from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# ============================================================
# 1) Configuration
# ============================================================

@dataclass
class ProductionConfig:
    time_col: str = "_time"
    user_col: str = "user"
    uid_col: str = "uid"
    remote_col: str = "remote_address"
    device_col: str = "device_ip_address"
    cmd_col: str = "cmd"
    user_type_col: str = "user_type"
    date_col: str = "event_date"

    warmup_days_human: int = 5
    warmup_days_functional: int = 7

    min_events_per_day_human: int = 1
    min_events_per_day_functional: int = 1

    human_contamination: float = 0.01
    functional_contamination: float = 0.01

    n_estimators: int = 300
    random_state: int = 42

    # approximate-normal filtering for training
    max_new_remote_rate_train_human: float = 0.25
    max_new_pair_rate_train_human: float = 0.25

    max_new_remote_rate_train_functional: float = 0.10
    max_new_pair_rate_train_functional: float = 0.10
    max_new_device_rate_train_functional: float = 0.10

    # if you later add policy violation features
    max_policy_violation_rate_train_functional: float = 0.0


# ============================================================
# 2) Command normalization
# ============================================================

def normalize_command(raw_cmd: object) -> str:
    if pd.isna(raw_cmd):
        return "UNKNOWN_CMD"

    cmd = str(raw_cmd).strip().lower()
    if not cmd:
        return "UNKNOWN_CMD"

    if cmd.startswith("show run") or cmd.startswith("show running"):
        return "SHOW_CONFIG"
    if cmd in {"conf t", "configure terminal"} or cmd.startswith("conf "):
        return "CONFIG_MODE"
    if cmd.startswith("reload") or cmd.startswith("reboot"):
        return "RELOAD"
    if cmd.startswith("copy "):
        return "COPY"
    if "user" in cmd and any(tok in cmd for tok in ["add", "delete", "remove", "username"]):
        return "USER_MGMT"

    return re.split(r"\s+", cmd)[0].upper()


# ============================================================
# 3) Preprocessing
# ============================================================

def preprocess_tacacs(df: pd.DataFrame, cfg: ProductionConfig) -> pd.DataFrame:
    required = [
        cfg.user_col,
        cfg.time_col,
        cfg.uid_col,
        cfg.remote_col,
        cfg.device_col,
        cfg.cmd_col,
        cfg.user_type_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out[cfg.time_col] = pd.to_datetime(out[cfg.time_col], errors="coerce")
    out = out.dropna(subset=[cfg.time_col, cfg.user_col])

    for c in [cfg.user_col, cfg.uid_col, cfg.remote_col, cfg.device_col, cfg.cmd_col, cfg.user_type_col]:
        out[c] = out[c].astype(str)

    out["cmd_group"] = out[cfg.cmd_col].map(normalize_command)
    out[cfg.date_col] = out[cfg.time_col].dt.floor("D")

    out = out.sort_values([cfg.user_col, cfg.time_col, cfg.uid_col]).reset_index(drop=True)
    return out


# ============================================================
# 4) Rolling novelty features at event level
# ============================================================

def add_event_novelty_flags(df: pd.DataFrame, cfg: ProductionConfig) -> pd.DataFrame:
    out = df.copy()

    hist_remote: Dict[str, Set[str]] = {}
    hist_device: Dict[str, Set[str]] = {}
    hist_cmd: Dict[str, Set[str]] = {}
    hist_remote_cmd: Dict[str, Set[Tuple[str, str]]] = {}
    hist_device_cmd: Dict[str, Set[Tuple[str, str]]] = {}

    new_remote_flags = []
    new_device_flags = []
    new_command_flags = []
    new_remote_cmd_flags = []
    new_device_cmd_flags = []

    for row in out.itertuples(index=False):
        user = getattr(row, cfg.user_col)
        remote = getattr(row, cfg.remote_col)
        device = getattr(row, cfg.device_col)
        cmd_group = getattr(row, "cmd_group")

        if user not in hist_remote:
            hist_remote[user] = set()
            hist_device[user] = set()
            hist_cmd[user] = set()
            hist_remote_cmd[user] = set()
            hist_device_cmd[user] = set()

        new_remote = int(remote not in hist_remote[user])
        new_device = int(device not in hist_device[user])
        new_command = int(cmd_group not in hist_cmd[user])
        new_remote_cmd = int((remote, cmd_group) not in hist_remote_cmd[user])
        new_device_cmd = int((device, cmd_group) not in hist_device_cmd[user])

        new_remote_flags.append(new_remote)
        new_device_flags.append(new_device)
        new_command_flags.append(new_command)
        new_remote_cmd_flags.append(new_remote_cmd)
        new_device_cmd_flags.append(new_device_cmd)

        hist_remote[user].add(remote)
        hist_device[user].add(device)
        hist_cmd[user].add(cmd_group)
        hist_remote_cmd[user].add((remote, cmd_group))
        hist_device_cmd[user].add((device, cmd_group))

    out["new_remote"] = new_remote_flags
    out["new_device"] = new_device_flags
    out["new_command"] = new_command_flags
    out["new_remote_cmd_pair"] = new_remote_cmd_flags
    out["new_device_cmd_pair"] = new_device_cmd_flags

    return out


# ============================================================
# 5) Event risk
# ============================================================

def add_event_risk(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # max theoretical score here = 25 + 20 + 15 + 30 + 35 = 125
    out["event_risk"] = (
        25 * out["new_remote"]
        + 20 * out["new_device"]
        + 15 * out["new_command"]
        + 30 * out["new_remote_cmd_pair"]
        + 35 * out["new_device_cmd_pair"]
    )
    out["event_risk_norm"] = out["event_risk"] / 125.0
    return out


# ============================================================
# 6) Aggregate to user-day
# ============================================================

def aggregate_user_day(event_df: pd.DataFrame, cfg: ProductionConfig) -> pd.DataFrame:
    rows = []

    for (user, day), g in event_df.groupby([cfg.user_col, cfg.date_col], sort=False):
        n = len(g)
        if n == 0:
            continue

        user_type = g[cfg.user_type_col].iloc[0]

        row = {
            cfg.user_col: user,
            cfg.date_col: day,
            cfg.user_type_col: user_type,

            "total_events": n,
            "log_events": math.log1p(n),

            "new_remote_rate": g["new_remote"].mean(),
            "new_device_rate": g["new_device"].mean(),
            "new_command_rate": g["new_command"].mean(),
            "new_remote_cmd_pair_rate": g["new_remote_cmd_pair"].mean(),
            "new_device_cmd_pair_rate": g["new_device_cmd_pair"].mean(),

            "new_remote_count": int(g["new_remote"].sum()),
            "new_device_count": int(g["new_device"].sum()),
            "new_command_count": int(g["new_command"].sum()),
            "new_remote_cmd_pair_count": int(g["new_remote_cmd_pair"].sum()),
            "new_device_cmd_pair_count": int(g["new_device_cmd_pair"].sum()),

            "avg_risk_norm": g["event_risk_norm"].mean(),
            "max_risk_norm": g["event_risk_norm"].max(),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values([cfg.user_col, cfg.date_col]).reset_index(drop=True)
    return out


# ============================================================
# 7) Merge existing graph features
# ============================================================

def merge_graph_features(
    user_day_df: pd.DataFrame,
    graph_df: pd.DataFrame,
    cfg: ProductionConfig,
) -> pd.DataFrame:
    """
    graph_df expected columns:
      - user
      - event_date
      - entropy
      - degree
      - clustering_coefficient
    """
    required = [cfg.user_col, cfg.date_col, "entropy", "degree", "clustering_coefficient"]
    missing = [c for c in required if c not in graph_df.columns]
    if missing:
        raise ValueError(f"Graph feature table missing columns: {missing}")

    g = graph_df.copy()
    g[cfg.date_col] = pd.to_datetime(g[cfg.date_col]).dt.floor("D")

    merged = user_day_df.merge(
        g[[cfg.user_col, cfg.date_col, "entropy", "degree", "clustering_coefficient"]],
        on=[cfg.user_col, cfg.date_col],
        how="left",
    )

    # Normalize graph features into more model-friendly scales
    merged["entropy_norm"] = merged["entropy"].fillna(0.0)

    # degree should not be used raw
    merged["degree_norm"] = (
        merged["degree"].fillna(0.0) / merged["total_events"].replace(0, 1)
    )

    merged["clustering_coeff"] = merged["clustering_coefficient"].fillna(0.0)

    return merged


# ============================================================
# 8) Warm-up metadata
# ============================================================

def add_warmup_metadata(df: pd.DataFrame, cfg: ProductionConfig) -> pd.DataFrame:
    out = df.copy()

    first_seen = out.groupby(cfg.user_col)[cfg.date_col].transform("min")
    out["days_since_first_seen"] = (out[cfg.date_col] - first_seen).dt.days

    out["warmup_days_required"] = np.where(
        out[cfg.user_type_col].eq("functional"),
        cfg.warmup_days_functional,
        cfg.warmup_days_human,
    )

    out["is_warmup"] = out["days_since_first_seen"] < out["warmup_days_required"]
    return out


# ============================================================
# 9) Feature sets
# ============================================================

def get_human_features() -> List[str]:
    return [
        "entropy_norm",
        "degree_norm",
        "clustering_coeff",
        "new_remote_rate",
        "new_device_rate",
        "new_command_rate",
        "new_remote_cmd_pair_rate",
        "avg_risk_norm",
        "max_risk_norm",
        "log_events",
    ]


def get_functional_features() -> List[str]:
    return [
        "entropy_norm",
        "degree_norm",
        "clustering_coeff",
        "new_remote_rate",
        "new_device_rate",
        "new_command_rate",
        "new_remote_cmd_pair_rate",
        "new_device_cmd_pair_rate",
        "avg_risk_norm",
        "max_risk_norm",
        "log_events",
        # later you can add:
        # "policy_violation_rate",
        # "unknown_policy_ratio",
    ]


# ============================================================
# 10) Training set selection
# ============================================================

def build_human_training_set(df: pd.DataFrame, cfg: ProductionConfig) -> pd.DataFrame:
    out = df.copy()
    out = out[out[cfg.user_type_col].eq("human")]
    out = out[~out["is_warmup"]]
    out = out[out["total_events"] >= cfg.min_events_per_day_human]
    out = out[out["new_remote_rate"] < cfg.max_new_remote_rate_train_human]
    out = out[out["new_remote_cmd_pair_rate"] < cfg.max_new_pair_rate_train_human]
    return out


def build_functional_training_set(df: pd.DataFrame, cfg: ProductionConfig) -> pd.DataFrame:
    out = df.copy()
    out = out[out[cfg.user_type_col].eq("functional")]
    out = out[~out["is_warmup"]]
    out = out[out["total_events"] >= cfg.min_events_per_day_functional]
    out = out[out["new_remote_rate"] < cfg.max_new_remote_rate_train_functional]
    out = out[out["new_device_rate"] < cfg.max_new_device_rate_train_functional]
    out = out[out["new_remote_cmd_pair_rate"] < cfg.max_new_pair_rate_train_functional]

    if "policy_violation_rate" in out.columns:
        out = out[out["policy_violation_rate"] <= cfg.max_policy_violation_rate_train_functional]

    return out


# ============================================================
# 11) Train separate models
# ============================================================

def train_human_model(train_df: pd.DataFrame, cfg: ProductionConfig) -> IsolationForest:
    features = get_human_features()
    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        contamination=cfg.human_contamination,
        random_state=cfg.random_state,
    )
    model.fit(train_df[features])
    return model


def train_functional_model(train_df: pd.DataFrame, cfg: ProductionConfig) -> IsolationForest:
    features = get_functional_features()
    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        contamination=cfg.functional_contamination,
        random_state=cfg.random_state,
    )
    model.fit(train_df[features])
    return model


# ============================================================
# 12) Score by user type
# ============================================================

def score_human_days(df: pd.DataFrame, model: IsolationForest) -> pd.DataFrame:
    out = df.copy()
    features = get_human_features()
    scores = model.decision_function(out[features])
    pred = model.predict(out[features])

    out["iforest_score"] = scores
    out["iforest_pred"] = pred
    out["is_anomaly"] = pred == -1
    out["model_type"] = "human_iforest"
    return out


def score_functional_days(df: pd.DataFrame, model: IsolationForest) -> pd.DataFrame:
    out = df.copy()
    features = get_functional_features()
    scores = model.decision_function(out[features])
    pred = model.predict(out[features])

    out["iforest_score"] = scores
    out["iforest_pred"] = pred
    out["is_anomaly"] = pred == -1
    out["model_type"] = "functional_iforest"
    return out


# ============================================================
# 13) Explanation layer
# ============================================================

def explain_row(row: pd.Series) -> List[str]:
    reasons = []

    if row.get("new_remote_rate", 0) > 0:
        reasons.append(f"new_remote_rate={row['new_remote_rate']:.3f}")
    if row.get("new_device_rate", 0) > 0:
        reasons.append(f"new_device_rate={row['new_device_rate']:.3f}")
    if row.get("new_remote_cmd_pair_rate", 0) > 0:
        reasons.append(f"new_remote_cmd_pair_rate={row['new_remote_cmd_pair_rate']:.3f}")
    if row.get("new_device_cmd_pair_rate", 0) > 0:
        reasons.append(f"new_device_cmd_pair_rate={row['new_device_cmd_pair_rate']:.3f}")
    if row.get("max_risk_norm", 0) > 0.5:
        reasons.append(f"max_risk_norm={row['max_risk_norm']:.3f}")
    if row.get("entropy_norm", 0) > 0.8:
        reasons.append(f"entropy_norm={row['entropy_norm']:.3f}")
    if row.get("degree_norm", 0) > 0.5:
        reasons.append(f"degree_norm={row['degree_norm']:.3f}")
    if row.get("policy_violation_rate", 0) > 0:
        reasons.append(f"policy_violation_rate={row['policy_violation_rate']:.3f}")

    if not reasons:
        reasons.append("isolated_by_feature_combination")

    return reasons


def add_explanations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["reasons"] = out.apply(explain_row, axis=1)
    return out


# ============================================================
# 14) Full production pipeline
# ============================================================

def run_production_pipeline(
    raw_tacacs_df: pd.DataFrame,
    graph_features_df: pd.DataFrame,
    cfg: Optional[ProductionConfig] = None,
) -> Dict[str, pd.DataFrame]:
    if cfg is None:
        cfg = ProductionConfig()

    # event-level
    clean_df = preprocess_tacacs(raw_tacacs_df, cfg)
    event_df = add_event_novelty_flags(clean_df, cfg)
    event_df = add_event_risk(event_df)

    # user-day
    user_day_df = aggregate_user_day(event_df, cfg)
    if user_day_df.empty:
        raise ValueError("No user-day rows generated.")

    # merge graph features
    user_day_df = merge_graph_features(user_day_df, graph_features_df, cfg)

    # add warm-up metadata
    user_day_df = add_warmup_metadata(user_day_df, cfg)

    # training sets
    human_train = build_human_training_set(user_day_df, cfg)
    functional_train = build_functional_training_set(user_day_df, cfg)

    if human_train.empty:
        raise ValueError("Human training set is empty.")
    if functional_train.empty:
        raise ValueError("Functional training set is empty.")

    # train separate models
    human_model = train_human_model(human_train, cfg)
    functional_model = train_functional_model(functional_train, cfg)

    # score all rows using matching model
    human_rows = user_day_df[user_day_df[cfg.user_type_col].eq("human")].copy()
    functional_rows = user_day_df[user_day_df[cfg.user_type_col].eq("functional")].copy()

    scored_human = score_human_days(human_rows, human_model) if not human_rows.empty else human_rows
    scored_functional = (
        score_functional_days(functional_rows, functional_model)
        if not functional_rows.empty else functional_rows
    )

    scored_all = pd.concat([scored_human, scored_functional], ignore_index=True)
    scored_all = add_explanations(scored_all)

    return {
        "clean_events": clean_df,
        "event_features": event_df,
        "user_day_features": user_day_df,
        "human_train": human_train,
        "functional_train": functional_train,
        "scored_user_days": scored_all.sort_values("iforest_score"),
    }