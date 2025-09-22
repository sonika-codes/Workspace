import pandas as pd
import numpy as np
from datetime import date as date_type
from typing import Optional

# ---------- Helpers ----------
def _hour_of_week(ts: pd.Series) -> pd.Series:
    return (ts.dt.dayofweek * 24 + ts.dt.hour) + 1  # 1..168

def _week_start_monday(ts: pd.Series) -> pd.Series:
    return ts.dt.to_period('W-MON').dt.start_time

def _prep(df: pd.DataFrame, manual_pattern=r"\bmanual\b") -> pd.DataFrame:
    d = df.copy()
    d['dttm'] = pd.to_datetime(d['dttm'], errors='coerce')
    d = d.dropna(subset=['dttm'])
    d['index_value'] = _hour_of_week(d['dttm'])
    d['week_start']  = _week_start_monday(d['dttm'])
    d['is_manual']   = d['id_chat'].astype(str).str.contains(manual_pattern, case=False, na=False)
    return d

def _weekly_wide(df: pd.DataFrame) -> pd.DataFrame:
    d = _prep(df)
    # aggregates
    total_calls = d.groupby(['user','week_start','index_value']).size().rename('total_calls')
    manual_calls = d[d['is_manual']].groupby(['user','week_start','index_value']).size().rename('manual_calls')

    # full user × week × 168 grid
    users = pd.Index(d['user'].unique(), name='user')
    weeks = pd.Index(d['week_start'].sort_values().unique(), name='week_start')
    hours = pd.Index(range(1, 169), name='index_value')
    base = pd.DataFrame(index=pd.MultiIndex.from_product([users, weeks, hours], names=['user','week_start','index_value']))

    base = base.join(total_calls, how='left').join(manual_calls, how='left')
    base[['total_calls','manual_calls']] = base[['total_calls','manual_calls']].fillna(0).astype(int)
    base['active_status'] = (base['total_calls'] > 0).astype(int)
    return base.reset_index()

def _weekly_long(wide: pd.DataFrame) -> pd.DataFrame:
    long = wide.melt(
        id_vars=['user','week_start','index_value'],
        value_vars=['total_calls','manual_calls','active_status'],
        var_name='metric_name', value_name='metric_value'
    )
    name_map = {
        'total_calls': 'total calls volume',
        'manual_calls': 'total manual call volume',
        'active_status': 'user active status'
    }
    long['metric_name'] = long['metric_name'].map(name_map)
    return long

def _day_wide(df: pd.DataFrame, day: date_type) -> pd.DataFrame:
    d = _prep(df)
    day_start = pd.Timestamp(day).normalize()
    day_end = day_start + pd.Timedelta(days=1)
    dd = d[(d['dttm'] >= day_start) & (d['dttm'] < day_end)]

    users = pd.Index(d['user'].unique(), name='user')
    dow = day_start.dayofweek
    day_hours = pd.Index(range(dow*24 + 1, dow*24 + 24 + 1), name='index_value')
    base = pd.DataFrame(index=pd.MultiIndex.from_product([users, day_hours], names=['user','index_value']))

    total = dd.groupby(['user','index_value']).size().rename('total_calls')
    manual = dd[dd['is_manual']].groupby(['user','index_value']).size().rename('manual_calls')

    base = base.join(total, how='left').join(manual, how='left')
    base[['total_calls','manual_calls']] = base[['total_calls','manual_calls']].fillna(0).astype(int)
    base['active_status'] = (base['total_calls'] > 0).astype(int)
    return base.reset_index()

def _baseline_wide(weekly_wide: pd.DataFrame,
                   end_week: pd.Timestamp,
                   lookback_weeks: Optional[int]) -> pd.DataFrame:
    wm = weekly_wide[weekly_wide['week_start'] < end_week].copy()
    if lookback_weeks is not None:
        start_cut = end_week - pd.to_timedelta(7 * lookback_weeks, unit='D')
        wm = wm[wm['week_start'] >= start_cut]
    g = wm.groupby(['user','index_value'], as_index=False)
    base = g.agg(
        n_weeks=('week_start', 'nunique'),
        avg_total_calls=('total_calls','mean'),
        avg_manual_calls=('manual_calls','mean'),
        avg_active_rate=('active_status','mean')
    )
    return base

# ---------- Unified table builder ----------
def build_unified_metrics_table(
    df: pd.DataFrame,
    as_of_day: date_type,
    lookback_weeks: Optional[int] = 8,
    manual_pattern: str = r"\bmanual\b",
    cold_start_policy: str = "suppress",   # 'suppress' | 'zero' | 'nan'
    include_active_deviation: bool = True
) -> pd.DataFrame:
    """
    Returns ONE table with columns:
      user, metric_name, metric_value, index_value, week_start_date, date, n_weeks

    Row types:
      - Daily metrics (for as_of_day): metric_name ∈ {'user active status','total calls volume','total manual call volume'}
        (week_start_date, date filled; n_weeks=NA)
      - Historic averages (prior full weeks): metric_name ∈ {'average user status','average total calls','average manual calls'}
        (n_weeks filled; week_start_date, date = NA)
      - Deviations (today - average): metric_name ∈ {'user status deviation','total calls deviation','manual calls deviation'}
        (week_start_date, date filled; n_weeks=NA)
    """
    # prep + compute week caches
    ww = _weekly_wide(df.assign(id_chat=df['id_chat']))  # ensure columns exist
    as_of_week = _week_start_monday(pd.Series(pd.Timestamp(as_of_day))).iloc[0]
    day_w = _day_wide(df.assign(id_chat=df['id_chat']), as_of_day)

    # ---- DAILY METRICS (long) ----
    daily_long = day_w.melt(
        id_vars=['user','index_value'],
        value_vars=['total_calls','manual_calls','active_status'],
        var_name='metric_name', value_name='metric_value'
    )
    daily_name_map = {
        'total_calls': 'total calls volume',
        'manual_calls': 'total manual call volume',
        'active_status': 'user active status'
    }
    daily_long['metric_name'] = daily_long['metric_name'].map(daily_name_map)
    daily_long['week_start_date'] = as_of_week
    daily_long['date'] = pd.Timestamp(as_of_day).normalize()
    daily_long['n_weeks'] = pd.NA

    # ---- BASELINE (averages across prior weeks) ----
    base_w = _baseline_wide(ww, as_of_week, lookback_weeks)
    base_rows = []
    # average total calls
    tmp = base_w[['user','index_value','n_weeks','avg_total_calls']].copy()
    tmp['metric_name'] = 'average total calls'
    tmp.rename(columns={'avg_total_calls':'metric_value'}, inplace=True)
    base_rows.append(tmp)
    # average manual calls
    tmp = base_w[['user','index_value','n_weeks','avg_manual_calls']].copy()
    tmp['metric_name'] = 'average manual calls'
    tmp.rename(columns={'avg_manual_calls':'metric_value'}, inplace=True)
    base_rows.append(tmp)
    # average user status
    tmp = base_w[['user','index_value','n_weeks','avg_active_rate']].copy()
    tmp['metric_name'] = 'average user status'
    tmp.rename(columns={'avg_active_rate':'metric_value'}, inplace=True)
    base_rows.append(tmp)

    baseline_long = pd.concat(base_rows, ignore_index=True)
    baseline_long['week_start_date'] = pd.NaT
    baseline_long['date'] = pd.NaT

    # ---- DEVIATIONS (today - average) ----
    m = day_w.merge(base_w, on=['user','index_value'], how='left', validate='m:1')
    m[['avg_total_calls','avg_manual_calls','avg_active_rate']] = m[['avg_total_calls','avg_manual_calls','avg_active_rate']].fillna(0.0)
    m['n_weeks'] = m['n_weeks'].fillna(0).astype(int)

    def _dev_block(curr_col, avg_col, label):
        dev = m[curr_col] - m[avg_col]
        if cold_start_policy == 'suppress':
            dev_series = dev.where(m['n_weeks'] > 0)  # will become NaN and drop below
            out = pd.DataFrame({'user': m['user'], 'index_value': m['index_value'],
                                'metric_name': label, 'metric_value': dev_series,
                                'week_start_date': as_of_week, 'date': pd.Timestamp(as_of_day).normalize(),
                                'n_weeks': pd.NA})
            out = out.dropna(subset=['metric_value'])
        elif cold_start_policy == 'zero':
            out = pd.DataFrame({'user': m['user'], 'index_value': m['index_value'],
                                'metric_name': label, 'metric_value': dev.where(m['n_weeks'] > 0, 0.0),
                                'week_start_date': as_of_week, 'date': pd.Timestamp(as_of_day).normalize(),
                                'n_weeks': pd.NA})
        elif cold_start_policy == 'nan':
            out = pd.DataFrame({'user': m['user'], 'index_value': m['index_value'],
                                'metric_name': label, 'metric_value': dev.where(m['n_weeks'] > 0, np.nan),
                                'week_start_date': as_of_week, 'date': pd.Timestamp(as_of_day).normalize(),
                                'n_weeks': pd.NA})
        else:
            raise ValueError("cold_start_policy must be 'suppress' | 'zero' | 'nan'")
        return out

    dev_rows = [
        _dev_block('total_calls',  'avg_total_calls',  'total calls deviation'),
        _dev_block('manual_calls', 'avg_manual_calls', 'manual calls deviation')
    ]
    if include_active_deviation:
        dev_rows.append(_dev_block('active_status','avg_active_rate','user status deviation'))

    deviation_long = pd.concat(dev_rows, ignore_index=True)

    # ---- UNION ALL ----
    daily_fmt = daily_long[['user','metric_name','metric_value','index_value','week_start_date','date','n_weeks']]
    baseline_fmt = baseline_long[['user','metric_name','metric_value','index_value','week_start_date','date','n_weeks']]
    deviation_fmt = deviation_long[['user','metric_name','metric_value','index_value','week_start_date','date','n_weeks']]

    unified = pd.concat([daily_fmt, baseline_fmt, deviation_fmt], ignore_index=True)
    # Stable ordering (optional)
    unified = unified.sort_values(['user','metric_name','date','week_start_date','index_value'], na_position='last', ignore_index=True)
    return unified
