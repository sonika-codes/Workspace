import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import date as date_type

# ---------- Helpers ----------
def _hour_of_week(ts: pd.Series) -> pd.Series:
    """1..168 index: Mon 00:00–00:59 -> 1, ..., Sun 23:00–23:59 -> 168"""
    return (ts.dt.dayofweek * 24 + ts.dt.hour) + 1

def _week_start_monday(ts: pd.Series) -> pd.Series:
    """Anchor to Monday 00:00 for each timestamp."""
    return ts.dt.to_period('W-MON').dt.start_time

def _prep(df: pd.DataFrame, manual_pattern=r"\bmanual\b") -> pd.DataFrame:
    """Parse dttm, add index_value, week_start, is_manual."""
    out = df.copy()
    out['dttm'] = pd.to_datetime(out['dttm'], errors='coerce')
    out = out.dropna(subset=['dttm'])
    out['index_value'] = _hour_of_week(out['dttm'])
    out['week_start']  = _week_start_monday(out['dttm'])
    out['is_manual']   = out['id_chat'].astype(str).str.contains(manual_pattern, case=False, na=False)
    return out

def _full_grid(users: pd.Index, week_starts: pd.Index, hours=range(1, 169)) -> pd.MultiIndex:
    """All users × all week_starts × 168 hours."""
    return pd.MultiIndex.from_product(
        [users, week_starts, pd.Index(hours, name='index_value')],
        names=['user', 'week_start', 'index_value']
    )

# ---------- 1) Weekly metrics ----------
def build_weekly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-week metrics with FULL grid (zeros where missing):
      columns: user, week_start, index_value, total_calls, manual_calls, active_status
    """
    d = _prep(df)

    # observed aggregates
    total_calls = (
        d.groupby(['user', 'week_start', 'index_value'])
         .size().rename('total_calls')
    )
    manual_calls = (
        d[d['is_manual']]
        .groupby(['user', 'week_start', 'index_value'])
        .size().rename('manual_calls')
    )

    users = pd.Index(d['user'].unique(), name='user')
    all_weeks = pd.Index(d['week_start'].sort_values().unique(), name='week_start')
    base = pd.DataFrame(index=_full_grid(users, all_weeks))

    base = base.join(total_calls, how='left').join(manual_calls, how='left')
    base[['total_calls','manual_calls']] = base[['total_calls','manual_calls']].fillna(0).astype(int)
    base['active_status'] = (base['total_calls'] > 0).astype(int)

    out = base.reset_index()
    return out[['user','week_start','index_value','total_calls','manual_calls','active_status']]

# ---------- 2) Historic baseline (averages & std across weeks) ----------
def compute_historic_baseline(
    weekly_metrics: pd.DataFrame,
    end_week: Optional[pd.Timestamp] = None,
    lookback_weeks: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute baseline per (user, index_value) across past weeks.
    - end_week: exclude this week and anything after it (typically the week of 'as_of_date')
    - lookback_weeks: if provided, limit to that many weeks before end_week
    Returns columns:
      user, index_value, n_weeks, avg_total_calls, std_total_calls,
      avg_manual_calls, std_manual_calls, avg_active_rate
    """
    wm = weekly_metrics.copy()

    if end_week is not None:
        wm = wm[wm['week_start'] < end_week]

    if (lookback_weeks is not None) and (end_week is not None):
        start_cut = end_week - pd.to_timedelta(7 * lookback_weeks, unit='D')
        wm = wm[wm['week_start'] >= start_cut]

    # group over historical weeks
    g = wm.groupby(['user','index_value'], as_index=False)
    base = g.agg(
        n_weeks=('week_start', 'nunique'),
        avg_total_calls=('total_calls','mean'),
        std_total_calls=('total_calls','std'),
        avg_manual_calls=('manual_calls','mean'),
        std_manual_calls=('manual_calls','std'),
        avg_active_rate=('active_status','mean'),
    )

    # Replace NaN std (single-week) with 0.0
    for c in ['std_total_calls','std_manual_calls']:
        base[c] = base[c].fillna(0.0)

    return base

# ---------- 3) Single-day metrics ----------
def build_day_metrics(df: pd.DataFrame, day: date_type) -> pd.DataFrame:
    """
    Build metrics for a specific calendar day.
    Returns: user, index_value, total_calls, manual_calls, active_status
    (Only the 24 index_values that fall on that day.)
    """
    d = _prep(df)
    # filter to that day (localize/convert earlier if needed)
    day_start = pd.Timestamp(day).normalize()
    day_end = day_start + pd.Timedelta(days=1)
    dd = d[(d['dttm'] >= day_start) & (d['dttm'] < day_end)]

    # If you want a FULL user × 24-hour grid for all users:
    users = pd.Index(d['user'].unique(), name='user')
    day_hours = pd.Index(sorted(dd['index_value'].unique())) if not dd.empty else pd.Index([], name='index_value')

    # If no calls for some users that day, we still want rows for the 24 day-hours.
    # Compute the 24 index_values that belong to this day-of-week:
    dow = day_start.dayofweek  # Monday=0
    day_hour_indices = pd.Index(range(dow*24 + 1, dow*24 + 24 + 1), name='index_value')
    full_grid = pd.MultiIndex.from_product([users, day_hour_indices], names=['user','index_value'])
    base = pd.DataFrame(index=full_grid)

    total_calls = (
        dd.groupby(['user','index_value'])
          .size().rename('total_calls')
    )
    manual_calls = (
        dd[dd['is_manual']]
          .groupby(['user','index_value'])
          .size().rename('manual_calls')
    )

    base = base.join(total_calls, how='left').join(manual_calls, how='left')
    base[['total_calls','manual_calls']] = base[['total_calls','manual_calls']].fillna(0).astype(int)
    base['active_status'] = (base['total_calls'] > 0).astype(int)
    return base.reset_index()[['user','index_value','total_calls','manual_calls','active_status']]

# ---------- 4) Deviation: day vs historic baseline ----------
def compute_deviation(
    day_metrics: pd.DataFrame,
    baseline: pd.DataFrame,
    eps: float = 1e-9
) -> pd.DataFrame:
    """
    Join day metrics with baseline and compute diffs, pct_diffs, z-scores.
    Returns tidy long format for both total_calls and manual_calls.
    """
    m = day_metrics.merge(baseline, on=['user','index_value'], how='left', validate='m:1')

    # Fill missing baseline with zeros (means) and NaNs (std); keep n_weeks to diagnose coverage
    m['avg_total_calls'] = m['avg_total_calls'].fillna(0.0)
    m['avg_manual_calls'] = m['avg_manual_calls'].fillna(0.0)
    m['std_total_calls'] = m['std_total_calls'].fillna(0.0)
    m['std_manual_calls'] = m['std_manual_calls'].fillna(0.0)
    m['n_weeks'] = m['n_weeks'].fillna(0).astype(int)

    def safe_pct_diff(curr, avg):
        return np.where(np.abs(avg) < eps, np.where(curr == 0, 0.0, np.inf), (curr - avg) / (avg + 0.0))

    def safe_z(curr, avg, std):
        # If std==0 → z=0 when curr==avg, else +/- inf depending on direction
        return np.where(std < eps, np.where(np.isclose(curr, avg), 0.0, np.sign(curr - avg) * np.inf), (curr - avg) / std)

    # Build two metric blocks then stack
    tot = m[['user','index_value','total_calls','avg_total_calls','std_total_calls','n_weeks']].copy()
    tot['metric_name'] = 'total calls volume'
    tot.rename(columns={'total_calls':'metric_value','avg_total_calls':'baseline_avg','std_total_calls':'baseline_std'}, inplace=True)
    tot['diff'] = tot['metric_value'] - tot['baseline_avg']
    tot['pct_diff'] = safe_pct_diff(tot['metric_value'], tot['baseline_avg'])
    tot['z'] = safe_z(tot['metric_value'], tot['baseline_avg'], tot['baseline_std'])

    man = m[['user','index_value','manual_calls','avg_manual_calls','std_manual_calls','n_weeks']].copy()
    man['metric_name'] = 'total manual call volume'
    man.rename(columns={'manual_calls':'metric_value','avg_manual_calls':'baseline_avg','std_manual_calls':'baseline_std'}, inplace=True)
    man['diff'] = man['metric_value'] - man['baseline_avg']
    man['pct_diff'] = safe_pct_diff(man['metric_value'], man['baseline_avg'])
    man['z'] = safe_z(man['metric_value'], man['baseline_avg'], man['baseline_std'])

    out = pd.concat([tot, man], ignore_index=True)
    # Optional: also compute active-status deviation vs avg_active_rate if you want
    return out[['user','metric_name','index_value','metric_value','baseline_avg','baseline_std','n_weeks','diff','pct_diff','z']]

# ---------- 5) Orchestrator ----------
def run_hour_of_week_pipeline(
    df: pd.DataFrame,
    as_of_day: date_type,
    lookback_weeks: Optional[int] = 8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline:
      1) weekly_metrics: full user×week×hour grid (zeros filled)
      2) historic_baseline: averages/std up to (but excluding) the week of `as_of_day`,
         optionally limited to `lookback_weeks`
      3) day_deviation: deviations for `as_of_day` (24 hours of that weekday) vs baseline

    Returns: (weekly_metrics, historic_baseline, day_deviation)
    """
    weekly_metrics = build_weekly_metrics(df)

    # The week that contains as_of_day (Monday 00:00)
    as_of_week = _week_start_monday(pd.Series(pd.Timestamp(as_of_day))).iloc[0]

    baseline = compute_historic_baseline(
        weekly_metrics=weekly_metrics,
        end_week=as_of_week,
        lookback_weeks=lookback_weeks
    )

    day_metrics = build_day_metrics(df, as_of_day)
    day_deviation = compute_deviation(day_metrics, baseline)

    return weekly_metrics, baseline, day_deviation
