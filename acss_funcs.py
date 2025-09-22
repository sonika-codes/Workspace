# unified_pipeline.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Iterable

# -------------------------
# Config-like constants
# -------------------------
DAILY_METRIC_NAMES = {
    "total calls volume",
    "total manual call volume",
    "user active status",
}
BASELINE_METRIC_NAMES = {
    "average total calls",
    "average manual calls",
    "average user status",
}
DEVIATION_METRIC_NAMES = {
    "total calls deviation",
    "manual calls deviation",
    "user status deviation",
}

# -------------------------
# Time helpers
# -------------------------
def _hour_of_week(ts: pd.Series) -> pd.Series:
    """1..168 (Mon 00:00=1 ... Sun 23:00=168)"""
    return (ts.dt.dayofweek * 24 + ts.dt.hour) + 1

def _week_start_monday(ts: pd.Series) -> pd.Series:
    """Return Monday 00:00 for each timestamp (as Timestamp)"""
    return ts.dt.to_period("W-MON").dt.start_time

def _as_date(ts: pd.Timestamp) -> pd.Timestamp:
    """Normalize to midnight (keeps tz if present)."""
    return pd.Timestamp(ts).normalize()

# -------------------------
# IO helpers
# -------------------------
def load_unified(unified_path: Path) -> pd.DataFrame:
    if unified_path.exists():
        return pd.read_parquet(unified_path)
    cols = ["user","metric_name","metric_value","index_value","week_start_date","date","n_weeks"]
    return pd.DataFrame(columns=cols)

def save_unified(unified: pd.DataFrame, unified_path: Path) -> None:
    unified.to_parquet(unified_path, index=False)

# -------------------------
# Daily CSV → DAILY rows
# -------------------------
def build_daily_rows_from_csv(csv_path: Path,
                              manual_pattern: str = r"\bmanual\b") -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Read one day's CSV and produce DAILY metric rows (long) for that calendar day.
    Returns: (daily_rows, as_of_day, as_of_week)
    Output schema: user, metric_name, metric_value, index_value, week_start_date, date, n_weeks(NA)
    """
    df = pd.read_csv(
        csv_path,
        dtype={"user":"string","account":"string","id_chat":"string"},
        parse_dates=["dttm"],
        infer_datetime_format=True,
    )
    if df.empty:
        raise ValueError(f"{csv_path} has no rows.")

    # Prep
    df["dttm"] = pd.to_datetime(df["dttm"], errors="coerce")
    df = df.dropna(subset=["dttm","user"])
    df["user"] = df["user"].str.strip()
    df["id_chat"] = df["id_chat"].astype("string").str.strip()

    # Infer the day from the latest timestamp in this file
    as_of_day = _as_date(df["dttm"].max())
    as_of_week = _week_start_monday(pd.Series(as_of_day)).iloc[0]

    # Compute features
    df["index_value"] = _hour_of_week(df["dttm"])
    df["is_manual"] = df["id_chat"].astype(str).str.contains(manual_pattern, case=False, na=False)

    # Constrain to this calendar day (guard against stragglers in file)
    day_start = as_of_day
    day_end = as_of_day + pd.Timedelta(days=1)
    dday = df[(df["dttm"] >= day_start) & (df["dttm"] < day_end)]

    # Build full grid: all users in today's file × the 24 hours of that weekday
    users = pd.Index(df["user"].unique(), name="user")
    dow = day_start.dayofweek
    day_hours = pd.Index(range(dow*24 + 1, dow*24 + 24 + 1), name="index_value")
    base = pd.DataFrame(index=pd.MultiIndex.from_product([users, day_hours], names=["user","index_value"]))

    # Aggregations
    total_calls = dday.groupby(["user","index_value"]).size().rename("total_calls")
    manual_calls = dday[dday["is_manual"]].groupby(["user","index_value"]).size().rename("manual_calls")

    base = base.join(total_calls, how="left").join(manual_calls, how="left")
    base[["total_calls","manual_calls"]] = base[["total_calls","manual_calls"]].fillna(0).astype(int)
    base["active_status"] = (base["total_calls"] > 0).astype(int)

    # Long format with your metric names
    daily_long = base.reset_index().melt(
        id_vars=["user","index_value"],
        value_vars=["total_calls","manual_calls","active_status"],
        var_name="metric_key",
        value_name="metric_value"
    )
    name_map = {
        "total_calls": "total calls volume",
        "manual_calls": "total manual call volume",
        "active_status": "user active status",
    }
    daily_long["metric_name"] = daily_long["metric_key"].map(name_map)
    daily_long.drop(columns=["metric_key"], inplace=True)

    # Attach dates
    daily_long["week_start_date"] = as_of_week
    daily_long["date"] = as_of_day
    daily_long["n_weeks"] = pd.NA

    # Final schema & sort
    daily_long = daily_long[["user","metric_name","metric_value","index_value","week_start_date","date","n_weeks"]]
    daily_long = daily_long.sort_values(["user","metric_name","index_value"], ignore_index=True)
    return daily_long, as_of_day, as_of_week

# -------------------------
# Recompute BASELINE from unified’s prior DAILY rows
# -------------------------
def recompute_baseline_from_unified(unified_prior: pd.DataFrame,
                                    as_of_week: pd.Timestamp) -> pd.DataFrame:
    """
    Compute historic averages using ONLY prior daily rows (date < current week).
    Returns baseline rows (long) with metric_name in BASELINE_METRIC_NAMES and n_weeks set.
    """
    if unified_prior.empty:
        # No history yet
        cols = ["user","metric_name","metric_value","index_value","week_start_date","date","n_weeks"]
        return pd.DataFrame(columns=cols)

    # Filter to DAILY rows only, from weeks strictly before as_of_week
    daily = unified_prior[
        (unified_prior["metric_name"].isin(DAILY_METRIC_NAMES)) &
        (unified_prior["week_start_date"].notna()) &
        (unified_prior["week_start_date"] < as_of_week)
    ].copy()

    if daily.empty:
        # Still no prior weeks
        cols = ["user","metric_name","metric_value","index_value","week_start_date","date","n_weeks"]
        return pd.DataFrame(columns=cols)

    # Map daily → keys for averaging
    # We'll compute per (user, index_value, week_start_date) then average across weeks.
    key_map = {
        "total calls volume": ("average total calls",),
        "total manual call volume": ("average manual calls",),
        "user active status": ("average user status",),
    }
    daily["_avg_name"] = daily["metric_name"].map(lambda x: key_map[x][0])

    # First, aggregate per week to be safe (should already be unique per (week, index_value))
    per_week = (
        daily.groupby(["user","index_value","week_start_date","_avg_name"], as_index=False)
             .agg(metric_value=("metric_value","sum"))
    )

    # Then average across distinct weeks
    baseline = (
        per_week
        .groupby(["user","index_value","_avg_name"], as_index=False)
        .agg(metric_value=("metric_value","mean"),
             n_weeks=("week_start_date","nunique"))
    )
    baseline.rename(columns={"_avg_name":"metric_name"}, inplace=True)

    # Fill week_start_date/date as NA (per your schema)
    baseline["week_start_date"] = pd.NaT
    baseline["date"] = pd.NaT

    # Final schema & sort
    baseline = baseline[["user","metric_name","metric_value","index_value","week_start_date","date","n_weeks"]]
    baseline = baseline.sort_values(["user","metric_name","index_value"], ignore_index=True)
    return baseline

# -------------------------
# Build DEVIATIONS (today − average)
# -------------------------
def build_deviations(daily_rows: pd.DataFrame,
                     baseline_rows: pd.DataFrame,
                     as_of_day: pd.Timestamp,
                     as_of_week: pd.Timestamp,
                     cold_start_policy: str = "suppress",
                     include_active: bool = True) -> pd.DataFrame:
    """
    Join today's DAILY with BASELINE to compute deviations.
    Returns rows with metric_name in DEVIATION_METRIC_NAMES.
    """
    if baseline_rows.empty:
        # No history → handle cold-start
        if cold_start_policy == "suppress":
            return pd.DataFrame(columns=daily_rows.columns)
        elif cold_start_policy == "zero":
            # Create zero deviations for today (same shape as daily)
            dev = daily_rows.copy()
            name_map = {
                "total calls volume": "total calls deviation",
                "total manual call volume": "manual calls deviation",
                "user active status": "user status deviation",
            }
            dev["metric_name"] = dev["metric_name"].map(name_map)
            dev["metric_value"] = 0.0
            dev["n_weeks"] = pd.NA
            return dev
        elif cold_start_policy == "nan":
            dev = daily_rows.copy()
            name_map = {
                "total calls volume": "total calls deviation",
                "total manual call volume": "manual calls deviation",
                "user active status": "user status deviation",
            }
            dev["metric_name"] = dev["metric_name"].map(name_map)
            dev["metric_value"] = np.nan
            dev["n_weeks"] = pd.NA
            return dev
        else:
            raise ValueError("cold_start_policy must be 'suppress' | 'zero' | 'nan'")

    # Pivot baseline → easy joins
    # Keep only the metrics we will subtract against
    b = baseline_rows.pivot_table(
        index=["user","index_value"],
        columns="metric_name",
        values="metric_value",
        aggfunc="first"
    ).reset_index()

    # Today's daily in wide form
    d = daily_rows.pivot_table(
        index=["user","index_value","week_start_date","date"],
        columns="metric_name", values="metric_value", aggfunc="first"
    ).reset_index()

    merged = d.merge(b, on=["user","index_value"], how="left", suffixes=("", "_avg"))

    # Compute deviations
    out_rows = []

    # total calls
    if "total calls volume" in merged.columns and "average total calls" in merged.columns:
        dev = merged["total calls volume"] - merged["average total calls"]
        out_rows.append(pd.DataFrame({
            "user": merged["user"],
            "metric_name": "total calls deviation",
            "metric_value": dev,
            "index_value": merged["index_value"],
            "week_start_date": merged["week_start_date"],
            "date": merged["date"],
            "n_weeks": pd.NA,
        }))

    # manual calls
    if "total manual call volume" in merged.columns and "average manual calls" in merged.columns:
        dev = merged["total manual call volume"] - merged["average manual calls"]
        out_rows.append(pd.DataFrame({
            "user": merged["user"],
            "metric_name": "manual calls deviation",
            "metric_value": dev,
            "index_value": merged["index_value"],
            "week_start_date": merged["week_start_date"],
            "date": merged["date"],
            "n_weeks": pd.NA,
        }))

    # user status (optional)
    if include_active and ("user active status" in merged.columns) and ("average user status" in merged.columns):
        dev = merged["user active status"] - merged["average user status"]
        out_rows.append(pd.DataFrame({
            "user": merged["user"],
            "metric_name": "user status deviation",
            "metric_value": dev,
            "index_value": merged["index_value"],
            "week_start_date": merged["week_start_date"],
            "date": merged["date"],
            "n_weeks": pd.NA,
        }))

    deviations = pd.concat(out_rows, ignore_index=True)

    # Cold-start suppression at row level where baseline missing (NaN)
    if cold_start_policy in {"suppress","nan"}:
        # If baseline missing → metric_value becomes NaN; drop if "suppress"
        if cold_start_policy == "suppress":
            deviations = deviations.dropna(subset=["metric_value"]).reset_index(drop=True)

    return deviations

# -------------------------
# One daily run (inference)
# -------------------------
def run_daily(csv_path: Path, unified_path: Path,
              cold_start_policy: str = "suppress",
              manual_pattern: str = r"\bmanual\b",
              include_active_deviation: bool = True) -> pd.DataFrame:
    """
    - Loads (or creates) unified table.
    - Builds today's DAILY rows from csv_path.
    - Recomputes BASELINE from unified's prior DAILY rows.
    - Builds DEVIATIONS (today - average) with cold-start policy.
    - Replaces old baseline rows, appends today's daily+deviation+new baseline, saves unified.
    Returns: the full unified table after the update.
    """
    unified = load_unified(unified_path)

    daily_rows, as_of_day, as_of_week = build_daily_rows_from_csv(csv_path, manual_pattern=manual_pattern)

    # Recompute baseline from prior unified DAILY rows only (exclude current week)
    baseline_rows = recompute_baseline_from_unified(unified_prior=unified, as_of_week=as_of_week)

    # Build deviations
    deviation_rows = build_deviations(
        daily_rows=daily_rows,
        baseline_rows=baseline_rows,
        as_of_day=as_of_day,
        as_of_week=as_of_week,
        cold_start_policy=cold_start_policy,
        include_active=include_active_deviation,
    )

    # Replace old baseline snapshot
    if not unified.empty:
        is_baseline = unified["metric_name"].isin(BASELINE_METRIC_NAMES)
        unified = unified.loc[~is_baseline].copy()

    # Append new rows
    new_rows = pd.concat([daily_rows, deviation_rows, baseline_rows], ignore_index=True)
    unified = pd.concat([unified, new_rows], ignore_index=True)

    # Persist
    save_unified(unified, unified_path)
    return unified

# -------------------------
# Optional: backfill a folder of daily CSVs chronologically
# -------------------------
def backfill_folder(csv_folder: Path,
                    unified_path: Path,
                    pattern: str = "*.csv",
                    cold_start_policy: str = "suppress",
                    manual_pattern: str = r"\bmanual\b",
                    include_active_deviation: bool = True) -> pd.DataFrame:
    """
    Process all CSVs in chronological order of their *max dttm*.
    """
    files = list(csv_folder.glob(pattern))
    if not files:
        raise RuntimeError("No CSV files found.")

    # Infer "day" for sort
    file_days = []
    for fp in files:
        try:
            df = pd.read_csv(fp, usecols=["dttm"], parse_dates=["dttm"])
            if df.empty:
                continue
            day = _as_date(df["dttm"].max())
            file_days.append((day, fp))
        except Exception:
            # fallback: use filename ordering
            file_days.append((pd.NaT, fp))
    # Sort by day then by name
    file_days.sort(key=lambda t: (pd.Timestamp.min if pd.isna(t[0]) else t[0], str(t[1])))

    unified = load_unified(unified_path)
    for _, fp in file_days:
        unified = run_daily(
            csv_path=fp,
            unified_path=unified_path,
            cold_start_policy=cold_start_policy,
            manual_pattern=manual_pattern,
            include_active_deviation=include_active_deviation,
        )
    return unified

# -------------------------
# If you want to run directly:
# -------------------------
if __name__ == "__main__":
    # Example usage:
    data_dir = Path("/path/to/your/daily_csvs")
    unified_out = data_dir / "unified.parquet"

    # One-day inference:
    # run_daily(data_dir / "calls_2025-09-21.csv", unified_out, cold_start_policy="suppress")

    # Or backfill all days in folder:
    # backfill_folder(data_dir, unified_out, pattern="*.csv", cold_start_policy="suppress")
    pass
