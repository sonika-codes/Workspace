from __future__ import annotations

import math
import re
import joblib
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from sklearn.ensemble import IsolationForest


# ============================================================
# 1) CONFIG
# ============================================================

@dataclass
class Config:
    # raw TACACS columns
    user_col: str = "user"
    time_col: str = "_time"
    uid_col: str = "uid"
    remote_col: str = "remote_address"
    device_col: str = "device_ip_address"
    cmd_col: str = "cmd"
    user_type_col: str = "user_type"

    # derived columns
    date_col: str = "event_date"

    # warmup
    warmup_days_human: int = 5
    warmup_days_functional: int = 7

    # minimum activity
    min_events_per_day_human: int = 1
    min_events_per_day_functional: int = 1

    # model params
    n_estimators: int = 300
    random_state: int = 42
    human_contamination: float = 0.01
    functional_contamination: float = 0.01

    # train-set filtering
    max_new_remote_rate_train_human: float = 0.25
    max_new_pair_rate_train_human: float = 0.25

    max_new_remote_rate_train_functional: float = 0.10
    max_new_device_rate_train_functional: float = 0.10
    max_new_pair_rate_train_functional: float = 0.10
    max_policy_violation_rate_train_functional: float = 0.0

    # persistence
    human_model_path: str = "human_model.pkl"
    functional_model_path: str = "functional_model.pkl"
    baseline_path: str = "baseline.pkl"


# ============================================================
# 2) COMMAND NORMALIZATION
# ============================================================

def normalize_command(raw_cmd: object) -> str:
    """
    Replace with your environment-specific normalization later.
    """
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
# 3) PREPROCESSING
# ============================================================

def preprocess_tacacs(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
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
# 4) POLICY RULES
# ============================================================

@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    any_remote: bool = False
    any_device: bool = False
    any_command: bool = False
    allowed_remote_ips: Optional[Set[str]] = None
    allowed_device_ips: Optional[Set[str]] = None
    allowed_commands: Optional[Set[str]] = None


def ensure_set_or_none(x):
    if x is None:
        return None
    if isinstance(x, set):
        return x
    if isinstance(x, list):
        return set(str(v) for v in x)
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return None
        return set(v.strip() for v in x.split(","))
    return None


def build_policy_lookup(policy_rules_df: Optional[pd.DataFrame]) -> Dict[str, List[PolicyRule]]:
    """
    Expected columns:
      user, rule_id, any_remote, any_device, any_command,
      allowed_remote_ips, allowed_device_ips, allowed_commands
    """
    if policy_rules_df is None or policy_rules_df.empty:
        return {}

    required = [
        "user", "rule_id", "any_remote", "any_device", "any_command",
        "allowed_remote_ips", "allowed_device_ips", "allowed_commands"
    ]
    missing = [c for c in required if c not in policy_rules_df.columns]
    if missing:
        raise ValueError(f"Policy rules table missing columns: {missing}")

    lookup: Dict[str, List[PolicyRule]] = {}

    for row in policy_rules_df.itertuples(index=False):
        user = str(getattr(row, "user"))
        rule = PolicyRule(
            rule_id=str(getattr(row, "rule_id")),
            any_remote=bool(getattr(row, "any_remote")),
            any_device=bool(getattr(row, "any_device")),
            any_command=bool(getattr(row, "any_command")),
            allowed_remote_ips=ensure_set_or_none(getattr(row, "allowed_remote_ips")),
            allowed_device_ips=ensure_set_or_none(getattr(row, "allowed_device_ips")),
            allowed_commands=ensure_set_or_none(getattr(row, "allowed_commands")),
        )
        lookup.setdefault(user, []).append(rule)

    return lookup


def field_match(value: str, any_flag: bool, allowlist: Optional[Set[str]]) -> Tuple[str, bool]:
    """
    status in {"ANY", "ALLOW", "VIOLATION", "UNKNOWN"}
    """
    if any_flag:
        return "ANY", True
    if allowlist is None:
        return "UNKNOWN", False
    return ("ALLOW", True) if value in allowlist else ("VIOLATION", False)


def evaluate_rules_for_event(remote: str, device: str, cmd_group: str, rules: List[PolicyRule]) -> Dict[str, Any]:
    """
    Event is ALLOW if any rule fully matches.
    If no rule matches:
      - UNKNOWN if any rule is indeterminate
      - else VIOLATION
    """
    if not rules:
        return {
            "policy_status": "UNKNOWN",
            "matched_rule_id": None,
            "best_rule_id": None,
            "best_field_status": {
                "remote": "UNKNOWN",
                "device": "UNKNOWN",
                "command": "UNKNOWN",
            },
            "any_present": False,
        }

    any_indeterminate = False
    best = None  # (match_count, rule_id, field_status, any_present)

    for r in rules:
        remote_status, remote_ok = field_match(remote, r.any_remote, r.allowed_remote_ips)
        device_status, device_ok = field_match(device, r.any_device, r.allowed_device_ips)
        cmd_status, cmd_ok = field_match(cmd_group, r.any_command, r.allowed_commands)

        field_status = {
            "remote": remote_status,
            "device": device_status,
            "command": cmd_status,
        }
        any_present = (
            remote_status == "ANY" or
            device_status == "ANY" or
            cmd_status == "ANY"
        )

        if "VIOLATION" in field_status.values():
            match_count = int(remote_ok) + int(device_ok) + int(cmd_ok)
        else:
            if remote_ok and device_ok and cmd_ok:
                return {
                    "policy_status": "ALLOW",
                    "matched_rule_id": r.rule_id,
                    "best_rule_id": r.rule_id,
                    "best_field_status": field_status,
                    "any_present": any_present,
                }
            any_indeterminate = True
            match_count = int(remote_ok) + int(device_ok) + int(cmd_ok)

        if best is None or match_count > best[0]:
            best = (match_count, r.rule_id, field_status, any_present)

    _, best_rule_id, best_field_status, best_any = best

    return {
        "policy_status": "UNKNOWN" if any_indeterminate else "VIOLATION",
        "matched_rule_id": None,
        "best_rule_id": best_rule_id,
        "best_field_status": best_field_status,
        "any_present": best_any,
    }


# ============================================================
# 5) ROLLING NOVELTY FEATURES - TRAINING VERSION
# ============================================================

def compute_novelty_training(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Uses only past data per user. No future leakage.
    """
    out = df.copy()

    hist_remote: Dict[str, Set[str]] = {}
    hist_device: Dict[str, Set[str]] = {}
    hist_cmd: Dict[str, Set[str]] = {}
    hist_remote_cmd: Dict[str, Set[Tuple[str, str]]] = {}
    hist_device_cmd: Dict[str, Set[Tuple[str, str]]] = {}

    new_remote = []
    new_device = []
    new_command = []
    new_remote_cmd_pair = []
    new_device_cmd_pair = []

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

        h_remote = hist_remote[user]
        h_device = hist_device[user]
        h_cmd = hist_cmd[user]
        h_remote_cmd = hist_remote_cmd[user]
        h_device_cmd = hist_device_cmd[user]

        new_remote.append(int(remote not in h_remote))
        new_device.append(int(device not in h_device))
        new_command.append(int(cmd_group not in h_cmd))
        new_remote_cmd_pair.append(int((remote, cmd_group) not in h_remote_cmd))
        new_device_cmd_pair.append(int((device, cmd_group) not in h_device_cmd))

        h_remote.add(remote)
        h_device.add(device)
        h_cmd.add(cmd_group)
        h_remote_cmd.add((remote, cmd_group))
        h_device_cmd.add((device, cmd_group))

    out["new_remote"] = new_remote
    out["new_device"] = new_device
    out["new_command"] = new_command
    out["new_remote_cmd_pair"] = new_remote_cmd_pair
    out["new_device_cmd_pair"] = new_device_cmd_pair

    return out


# ============================================================
# 6) ROLLING NOVELTY FEATURES - INFERENCE VERSION
# ============================================================

def compute_novelty_inference(df: pd.DataFrame, baseline: Dict[str, Dict[str, Set]], cfg: Config) -> pd.DataFrame:
    """
    Uses persisted baseline. Does not update it here.
    """
    out = df.copy()

    new_remote = []
    new_device = []
    new_command = []
    new_remote_cmd_pair = []
    new_device_cmd_pair = []

    for row in out.itertuples(index=False):
        user = getattr(row, cfg.user_col)
        remote = getattr(row, cfg.remote_col)
        device = getattr(row, cfg.device_col)
        cmd_group = getattr(row, "cmd_group")

        if user not in baseline:
            baseline[user] = {
                "remote": set(),
                "device": set(),
                "cmd": set(),
                "remote_cmd": set(),
                "device_cmd": set(),
            }

        h = baseline[user]

        new_remote.append(int(remote not in h["remote"]))
        new_device.append(int(device not in h["device"]))
        new_command.append(int(cmd_group not in h["cmd"]))
        new_remote_cmd_pair.append(int((remote, cmd_group) not in h["remote_cmd"]))
        new_device_cmd_pair.append(int((device, cmd_group) not in h["device_cmd"]))

    out["new_remote"] = new_remote
    out["new_device"] = new_device
    out["new_command"] = new_command
    out["new_remote_cmd_pair"] = new_remote_cmd_pair
    out["new_device_cmd_pair"] = new_device_cmd_pair

    return out


def update_baseline(df: pd.DataFrame, baseline: Dict[str, Dict[str, Set]], cfg: Config) -> Dict[str, Dict[str, Set]]:
    """
    Update AFTER scoring.
    """
    for row in df.itertuples(index=False):
        user = getattr(row, cfg.user_col)
        remote = getattr(row, cfg.remote_col)
        device = getattr(row, cfg.device_col)
        cmd_group = getattr(row, "cmd_group")

        if user not in baseline:
            baseline[user] = {
                "remote": set(),
                "device": set(),
                "cmd": set(),
                "remote_cmd": set(),
                "device_cmd": set(),
            }

        h = baseline[user]
        h["remote"].add(remote)
        h["device"].add(device)
        h["cmd"].add(cmd_group)
        h["remote_cmd"].add((remote, cmd_group))
        h["device_cmd"].add((device, cmd_group))

    return baseline


# ============================================================
# 7) FUNCTIONAL POLICY FEATURES
# ============================================================

def add_functional_policy_features(event_df: pd.DataFrame, policy_lookup: Dict[str, List[PolicyRule]], cfg: Config) -> pd.DataFrame:
    out = event_df.copy()

    policy_statuses = []
    policy_risks = []
    control_gap_risks = []

    remote_violation = []
    device_violation = []
    command_violation = []

    remote_unknown = []
    device_unknown = []
    command_unknown = []

    best_rule_ids = []
    matched_rule_ids = []

    for row in out.itertuples(index=False):
        user = getattr(row, cfg.user_col)
        user_type = getattr(row, cfg.user_type_col)
        remote = getattr(row, cfg.remote_col)
        device = getattr(row, cfg.device_col)
        cmd_group = getattr(row, "cmd_group")

        if user_type != "functional":
            policy_statuses.append("NA")
            policy_risks.append(0.0)
            control_gap_risks.append(0.0)

            remote_violation.append(0)
            device_violation.append(0)
            command_violation.append(0)

            remote_unknown.append(0)
            device_unknown.append(0)
            command_unknown.append(0)

            best_rule_ids.append(None)
            matched_rule_ids.append(None)
            continue

        rules = policy_lookup.get(user, [])
        result = evaluate_rules_for_event(remote, device, cmd_group, rules)
        field_status = result["best_field_status"]

        rv = int(field_status.get("remote") == "VIOLATION")
        dv = int(field_status.get("device") == "VIOLATION")
        cv = int(field_status.get("command") == "VIOLATION")

        ru = int(field_status.get("remote") == "UNKNOWN")
        du = int(field_status.get("device") == "UNKNOWN")
        cu = int(field_status.get("command") == "UNKNOWN")

        prisk = 60 * rv + 60 * dv + 80 * cv
        cg_risk = 10 * ru + 10 * du + 15 * cu

        policy_statuses.append(result["policy_status"])
        policy_risks.append(prisk)
        control_gap_risks.append(cg_risk)

        remote_violation.append(rv)
        device_violation.append(dv)
        command_violation.append(cv)

        remote_unknown.append(ru)
        device_unknown.append(du)
        command_unknown.append(cu)

        best_rule_ids.append(result["best_rule_id"])
        matched_rule_ids.append(result["matched_rule_id"])

    out["policy_status"] = policy_statuses
    out["policy_risk"] = policy_risks
    out["control_gap_risk"] = control_gap_risks

    out["remote_violation"] = remote_violation
    out["device_violation"] = device_violation
    out["command_violation"] = command_violation

    out["remote_unknown"] = remote_unknown
    out["device_unknown"] = device_unknown
    out["command_unknown"] = command_unknown

    out["best_rule_id"] = best_rule_ids
    out["matched_rule_id"] = matched_rule_ids

    return out


# ============================================================
# 8) EVENT RISK
# ============================================================

def add_event_risk(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    novelty_risk = (
        25 * out["new_remote"] +
        20 * out["new_device"] +
        15 * out["new_command"] +
        30 * out["new_remote_cmd_pair"] +
        35 * out["new_device_cmd_pair"]
    )

    if "policy_risk" not in out.columns:
        out["policy_risk"] = 0.0
    if "control_gap_risk" not in out.columns:
        out["control_gap_risk"] = 0.0

    out["event_risk"] = novelty_risk + out["policy_risk"] + out["control_gap_risk"]
    out["event_risk_norm"] = np.minimum(out["event_risk"] / 325.0, 1.0)

    return out


# ============================================================
# 9) USER-DAY AGGREGATION
# ============================================================

def aggregate_user_day(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["user", "event_date"], sort=False)

    policy_cols_present = all(c in df.columns for c in [
        "remote_violation", "device_violation", "command_violation",
        "remote_unknown", "device_unknown", "command_unknown",
        "policy_risk", "control_gap_risk"
    ])

    agg_dict = {
        "user_type": "first",
        "new_remote": "mean",
        "new_device": "mean",
        "new_command": "mean",
        "new_remote_cmd_pair": "mean",
        "new_device_cmd_pair": "mean",
        "event_risk_norm": ["mean", "max"],
    }

    if policy_cols_present:
        agg_dict.update({
            "remote_violation": "mean",
            "device_violation": "mean",
            "command_violation": "mean",
            "remote_unknown": "mean",
            "device_unknown": "mean",
            "command_unknown": "mean",
            "policy_risk": "mean",
            "control_gap_risk": "mean",
        })

    out = grouped.agg(agg_dict)

    base_cols = [
        "user_type",
        "new_remote_rate",
        "new_device_rate",
        "new_command_rate",
        "new_remote_cmd_pair_rate",
        "new_device_cmd_pair_rate",
        "avg_risk_norm",
        "max_risk_norm",
    ]

    if policy_cols_present:
        extra_cols = [
            "remote_violation_rate",
            "device_violation_rate",
            "command_violation_rate",
            "remote_unknown_rate",
            "device_unknown_rate",
            "command_unknown_rate",
            "avg_policy_risk",
            "avg_control_gap_risk",
        ]
        out.columns = base_cols + extra_cols
    else:
        out.columns = base_cols

    out["total_events"] = grouped.size()
    out["log_events"] = np.log1p(out["total_events"])

    if policy_cols_present:
        out["policy_violation_rate"] = (
            out["remote_violation_rate"] +
            out["device_violation_rate"] +
            out["command_violation_rate"]
        )
        out["unknown_policy_ratio"] = (
            out["remote_unknown_rate"] +
            out["device_unknown_rate"] +
            out["command_unknown_rate"]
        )

    return out.reset_index()


# ============================================================
# 10) MERGE EXISTING GRAPH FEATURES
# ============================================================

def merge_graph_features(user_day_df: pd.DataFrame, graph_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    graph_df expected columns:
      user, event_date, entropy, degree, clustering_coefficient
    """
    required = ["user", "event_date", "entropy", "degree", "clustering_coefficient"]
    missing = [c for c in required if c not in graph_df.columns]
    if missing:
        raise ValueError(f"Graph feature table missing columns: {missing}")

    g = graph_df.copy()
    g["event_date"] = pd.to_datetime(g["event_date"]).dt.floor("D")

    out = user_day_df.merge(
        g[["user", "event_date", "entropy", "degree", "clustering_coefficient"]],
        on=["user", "event_date"],
        how="left"
    )

    out["entropy_norm"] = out["entropy"].fillna(0.0)
    out["degree_norm"] = out["degree"].fillna(0.0) / out["total_events"].replace(0, 1)
    out["clustering_coeff"] = out["clustering_coefficient"].fillna(0.0)

    return out


# ============================================================
# 11) WARMUP
# ============================================================

def add_warmup_metadata(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()

    first_seen = out.groupby(cfg.user_col)[cfg.date_col].transform("min")
    out["days_since_first_seen"] = (out[cfg.date_col] - first_seen).dt.days

    out["warmup_days_required"] = np.where(
        out[cfg.user_type_col].eq("functional"),
        cfg.warmup_days_functional,
        cfg.warmup_days_human
    )

    out["is_warmup"] = out["days_since_first_seen"] < out["warmup_days_required"]
    return out


# ============================================================
# 12) FEATURE LISTS
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
        "policy_violation_rate",
        "unknown_policy_ratio",
        "avg_policy_risk",
        "avg_control_gap_risk",
        "avg_risk_norm",
        "max_risk_norm",
        "log_events",
    ]


# ============================================================
# 13) TRAIN SET BUILDERS
# ============================================================

def build_human_training_set(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out = out[out[cfg.user_type_col].eq("human")]
    out = out[~out["is_warmup"]]
    out = out[out["total_events"] >= cfg.min_events_per_day_human]
    out = out[out["new_remote_rate"] < cfg.max_new_remote_rate_train_human]
    out = out[out["new_remote_cmd_pair_rate"] < cfg.max_new_pair_rate_train_human]
    return out


def build_functional_training_set(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
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
# 14) TRAIN MODELS
# ============================================================

def train_models(human_train: pd.DataFrame, functional_train: pd.DataFrame, cfg: Config):
    human_features = get_human_features()
    functional_features = get_functional_features()

    if human_train.empty:
        raise ValueError("Human training set is empty.")
    if functional_train.empty:
        raise ValueError("Functional training set is empty.")

    human_model = IsolationForest(
        contamination=cfg.human_contamination,
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state,
    )
    human_model.fit(human_train[human_features])

    functional_model = IsolationForest(
        contamination=cfg.functional_contamination,
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state,
    )
    functional_model.fit(functional_train[functional_features])

    return human_model, functional_model


# ============================================================
# 15) SCORE MODELS
# ============================================================

def score_human_days(df: pd.DataFrame, model: IsolationForest) -> pd.DataFrame:
    out = df.copy()
    features = get_human_features()
    out["iforest_score"] = model.decision_function(out[features])
    out["iforest_pred"] = model.predict(out[features])
    out["is_anomaly"] = out["iforest_pred"] == -1
    out["model_type"] = "human_iforest"
    return out


def score_functional_days(df: pd.DataFrame, model: IsolationForest) -> pd.DataFrame:
    out = df.copy()
    features = get_functional_features()
    out["iforest_score"] = model.decision_function(out[features])
    out["iforest_pred"] = model.predict(out[features])
    out["is_anomaly"] = out["iforest_pred"] == -1
    out["model_type"] = "functional_iforest"
    return out


# ============================================================
# 16) EXPLANATIONS
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
    if row.get("policy_violation_rate", 0) > 0:
        reasons.append(f"policy_violation_rate={row['policy_violation_rate']:.3f}")
    if row.get("unknown_policy_ratio", 0) > 0:
        reasons.append(f"unknown_policy_ratio={row['unknown_policy_ratio']:.3f}")
    if row.get("max_risk_norm", 0) > 0.5:
        reasons.append(f"max_risk_norm={row['max_risk_norm']:.3f}")
    if row.get("entropy_norm", 0) > 0.8:
        reasons.append(f"entropy_norm={row['entropy_norm']:.3f}")
    if row.get("degree_norm", 0) > 0.5:
        reasons.append(f"degree_norm={row['degree_norm']:.3f}")

    if not reasons:
        reasons.append("isolated_by_feature_combination")

    return reasons


def add_explanations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["reasons"] = out.apply(explain_row, axis=1)
    return out


# ============================================================
# 17) BASELINE PERSISTENCE
# ============================================================

def load_baseline(cfg: Config) -> Dict[str, Dict[str, Set]]:
    try:
        baseline = joblib.load(cfg.baseline_path)
    except FileNotFoundError:
        baseline = {}
    return baseline


def save_baseline(baseline: Dict[str, Dict[str, Set]], cfg: Config) -> None:
    joblib.dump(baseline, cfg.baseline_path)


# ============================================================
# 18) MODEL PERSISTENCE
# ============================================================

def save_models(human_model: IsolationForest, functional_model: IsolationForest, cfg: Config) -> None:
    joblib.dump(human_model, cfg.human_model_path)
    joblib.dump(functional_model, cfg.functional_model_path)


def load_models(cfg: Config):
    human_model = joblib.load(cfg.human_model_path)
    functional_model = joblib.load(cfg.functional_model_path)
    return human_model, functional_model


# ============================================================
# 19) WEEKLY TRAINING PIPELINE
# ============================================================

def training_pipeline(
    raw_tacacs_df: pd.DataFrame,
    graph_features_df: pd.DataFrame,
    policy_rules_df: Optional[pd.DataFrame],
    cfg: Config,
) -> Dict[str, pd.DataFrame]:
    """
    Weekly / periodic job.
    Recomputes features from historical data and retrains models.
    """
    policy_lookup = build_policy_lookup(policy_rules_df)

    # preprocess
    df = preprocess_tacacs(raw_tacacs_df, cfg)

    # rolling novelty using historical-only within training window
    event_df = compute_novelty_training(df, cfg)

    # policy features for functional accounts
    event_df = add_functional_policy_features(event_df, policy_lookup, cfg)

    # risk
    event_df = add_event_risk(event_df)

    # aggregate user-day
    user_day_df = aggregate_user_day(event_df)

    # merge graph features
    user_day_df = merge_graph_features(user_day_df, graph_features_df, cfg)

    # warmup
    user_day_df = add_warmup_metadata(user_day_df, cfg)

    # build train sets
    human_train = build_human_training_set(user_day_df, cfg)
    functional_train = build_functional_training_set(user_day_df, cfg)

    # train models
    human_model, functional_model = train_models(human_train, functional_train, cfg)

    # save models
    save_models(human_model, functional_model, cfg)

    return {
        "clean_events": df,
        "event_features": event_df,
        "user_day_features": user_day_df,
        "human_train": human_train,
        "functional_train": functional_train,
    }


# ============================================================
# 20) DAILY INFERENCE PIPELINE
# ============================================================

def inference_pipeline(
    raw_tacacs_df: pd.DataFrame,
    graph_features_df: pd.DataFrame,
    policy_rules_df: Optional[pd.DataFrame],
    cfg: Config,
) -> Dict[str, pd.DataFrame]:
    """
    Daily job.
    Loads persisted models and baseline, scores today's data, then updates baseline.
    """
    human_model, functional_model = load_models(cfg)
    baseline = load_baseline(cfg)
    policy_lookup = build_policy_lookup(policy_rules_df)

    # preprocess
    df = preprocess_tacacs(raw_tacacs_df, cfg)

    # novelty using persisted baseline
    event_df = compute_novelty_inference(df, baseline, cfg)

    # policy features
    event_df = add_functional_policy_features(event_df, policy_lookup, cfg)

    # risk
    event_df = add_event_risk(event_df)

    # aggregate user-day
    user_day_df = aggregate_user_day(event_df)

    # merge graph features
    user_day_df = merge_graph_features(user_day_df, graph_features_df, cfg)

    # warmup metadata
    user_day_df = add_warmup_metadata(user_day_df, cfg)

    # split and score
    human_rows = user_day_df[user_day_df[cfg.user_type_col].eq("human")].copy()
    functional_rows = user_day_df[user_day_df[cfg.user_type_col].eq("functional")].copy()

    scored_parts = []

    if not human_rows.empty:
        scored_parts.append(score_human_days(human_rows, human_model))

    if not functional_rows.empty:
        f_scored = score_functional_days(functional_rows, functional_model)

        # optional direct policy alert
        if "policy_violation_rate" in f_scored.columns:
            f_scored["direct_policy_alert"] = f_scored["policy_violation_rate"] > 0
        else:
            f_scored["direct_policy_alert"] = False

        scored_parts.append(f_scored)

    if scored_parts:
        scored_user_days = pd.concat(scored_parts, ignore_index=True)
        scored_user_days = add_explanations(scored_user_days)
        scored_user_days = scored_user_days.sort_values("iforest_score").reset_index(drop=True)
    else:
        scored_user_days = pd.DataFrame()

    # update baseline AFTER scoring
    baseline = update_baseline(df, baseline, cfg)
    save_baseline(baseline, cfg)

    return {
        "clean_events": df,
        "event_features": event_df,
        "user_day_features": user_day_df,
        "scored_user_days": scored_user_days,
    }


# ============================================================
# 21) EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    cfg = Config()

    # Example raw TACACS data
    raw_tacacs_df = pd.DataFrame({
        "user": ["alice", "alice", "svc_backup", "svc_backup", "alice"],
        "_time": [
            "2026-03-01 09:00:00",
            "2026-03-01 09:05:00",
            "2026-03-01 10:00:00",
            "2026-03-02 10:00:00",
            "2026-03-02 09:00:00",
        ],
        "uid": ["1", "2", "3", "4", "5"],
        "remote_address": ["10.1.1.5", "10.1.1.5", "10.2.2.2", "10.2.2.9", "10.9.9.9"],
        "device_ip_address": ["router1", "router2", "backup1", "backup1", "router9"],
        "cmd": ["show run", "conf t", "copy run", "reload", "reload"],
        "user_type": ["human", "human", "functional", "functional", "human"],
    })

    # Example graph feature table
    graph_features_df = pd.DataFrame({
        "user": ["alice", "alice", "svc_backup", "svc_backup"],
        "event_date": ["2026-03-01", "2026-03-02", "2026-03-01", "2026-03-02"],
        "entropy": [0.2, 0.8, 0.1, 0.4],
        "degree": [5, 8, 2, 3],
        "clustering_coefficient": [0.7, 0.6, 0.9, 0.8],
    })

    # Example policy rules
    policy_rules_df = pd.DataFrame({
        "user": ["svc_backup", "svc_backup"],
        "rule_id": ["r1", "r2"],
        "any_remote": [False, True],
        "any_device": [False, False],
        "any_command": [False, False],
        "allowed_remote_ips": [["10.2.2.2"], None],
        "allowed_device_ips": [["backup1"], ["backup1"]],
        "allowed_commands": [["COPY"], ["SHOW_CONFIG"]],
    })

    # TRAINING EXAMPLE
    training_artifacts = training_pipeline(
        raw_tacacs_df=raw_tacacs_df,
        graph_features_df=graph_features_df,
        policy_rules_df=policy_rules_df,
        cfg=cfg,
    )

    print("\n=== TRAIN USER-DAY FEATURES ===")
    print(training_artifacts["user_day_features"])

    # Initialize empty baseline for demo
    save_baseline({}, cfg)

    # INFERENCE EXAMPLE
    inference_artifacts = inference_pipeline(
        raw_tacacs_df=raw_tacacs_df,
        graph_features_df=graph_features_df,
        policy_rules_df=policy_rules_df,
        cfg=cfg,
    )

    print("\n=== SCORED USER-DAYS ===")
    print(inference_artifacts["scored_user_days"])