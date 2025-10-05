# app_acss_dashboard.py
# Dash AG Grid dashboard:
#   - Date range (defaults to latest date only)
#   - Table 1: Users-by-day summary
#   - Table 2: User drill-down (suspicious report rows)
#   - Table 3: Account drill-down (suspicious report rows)

from __future__ import annotations
import os
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_ag_grid as dag

# ============================================================
# ===============  LOAD DATA (EDIT THIS BLOCK)  ==============
# ============================================================
def load_bigquery_account_logs() -> pd.DataFrame:
    """
    Expected columns: dttm, user, account, id_chat
    Replace the demo with your real BigQuery load (pd.read_gbq or bigquery.Client).
    """
    # --- DEMO DATA (keeps app runnable) ---
    rng = pd.date_range("2025-08-01", "2025-08-24 22:00", freq="6H")
    users = ["alice", "bob", "charlie", "diana", "eve"]
    accts = [f"ACCT{i:04d}" for i in range(1, 31)]
    df = pd.DataFrame({
        "dttm": rng.repeat(5),
        "user": [users[i % len(users)] for i in range(len(rng)*5)],
        "account": [accts[i % len(accts)] for i in range(len(rng)*5)],
        "id_chat": ["manual call" if i % 4 == 0 else "other call" for i in range(len(rng)*5)],
    })
    return df

def load_postgres_suspicious_report() -> pd.DataFrame:
    """
    Expected columns:
      dttm, user, account, id_chat, date,
      dept_fctn_id, dept_fctn_desc, emp_status_ind, user_comm_id,
      job_desc, job_cd, iso_country_cd, num_of_unique_suspicious_accounts
    Replace the demo with your real Postgres load (pd.read_sql with SQLAlchemy engine).
    """
    # --- DEMO DATA synthesized from account_logs ---
    account_logs = load_bigquery_account_logs()
    account_logs["date"] = pd.to_datetime(account_logs["dttm"]).dt.date

    # Fake the extra org columns
    org = {
        "dept_fctn_id": "SVC",
        "dept_fctn_desc": "Service",
        "emp_status_ind": "A",
        "user_comm_id": "NA",
        "job_desc": "CSA",
        "job_cd": "CSA1",
        "iso_country_cd": "US",
    }
    # num_of_unique_suspicious_accounts: per (user, date) distinct accounts
    uniq_per_ud = (account_logs
                   .groupby(["user", "date"])["account"]
                   .nunique()
                   .reset_index()
                   .rename(columns={"account": "num_of_unique_suspicious_accounts"}))
    suspicious = account_logs.merge(uniq_per_ud, on=["user", "date"], how="left")
    for k, v in org.items():
        suspicious[k] = v
    return suspicious

# Real loads
account_logs = load_bigquery_account_logs()
suspicious = load_postgres_suspicious_report()

# Normalize datetime once
account_logs["dttm"] = pd.to_datetime(account_logs["dttm"], errors="coerce")
suspicious["dttm"] = pd.to_datetime(suspicious["dttm"], errors="coerce")
suspicious["date"] = pd.to_datetime(suspicious["date"], errors="coerce").dt.date

# ============================================================
# =====================  HELPERS  ============================
# ============================================================
def end_inclusive(end_date_like) -> pd.Timestamp:
    ed = pd.to_datetime(end_date_like)
    return ed + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

def stringify_dt(df: pd.DataFrame, cols=("dttm", "first_seen", "last_seen")) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    return out

def filter_suspicious_by_range(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    sd = pd.to_datetime(start_date)
    ed = end_inclusive(end_date)
    return df[(df["dttm"] >= sd) & (df["dttm"] <= ed)].copy()

def build_users_by_day_table(susp_in_range: pd.DataFrame) -> pd.DataFrame:
    """
    First table:
      - One row per (user, date)
      - suspicious_accounts_for_day = COUNT(DISTINCT account) that day (from suspicious report)
      - active_days = COUNT(DISTINCT date) for that user in the selected range (replicated on each row)
      - first_seen = MIN(dttm) for that user in the range
      - last_seen  = MAX(dttm) for that user in the range
    """
    if len(susp_in_range) == 0:
        return pd.DataFrame(columns=[
            "date", "user", "suspicious_accounts_for_day", "active_days", "first_seen", "last_seen"
        ])

    # per (user, date)
    per_ud = (susp_in_range
              .groupby(["user", susp_in_range["dttm"].dt.date], as_index=False)
              .agg(
                  suspicious_accounts_for_day=("account", "nunique"),
                  first_seen=("dttm", "min"),
                  last_seen=("dttm", "max"),
              )
              .rename(columns={"dttm": "date"}))

    per_ud = per_ud.rename(columns={"dttm": "date"})
    per_ud["date"] = per_ud["date"]  # already a date from groupby key

    # per-user aggregates across the range
    per_u = (susp_in_range
             .assign(date_only=susp_in_range["dttm"].dt.date)
             .groupby("user", as_index=False)
             .agg(
                 active_days=("date_only", "nunique"),
                 user_first_seen=("dttm", "min"),
                 user_last_seen=("dttm", "max"),
             ))

    # join
    out = per_ud.merge(per_u, on="user", how="left")
    # final columns / names
    out = (out
           .rename(columns={"user_first_seen": "first_seen", "user_last_seen": "last_seen"})
           .loc[:, ["date", "user", "suspicious_accounts_for_day", "active_days", "first_seen", "last_seen"]]
           .sort_values(["date", "suspicious_accounts_for_day", "user"], ascending=[False, False, True])
           .reset_index(drop=True))
    return out

# Date picker defaults -> latest date only
all_min_date = suspicious["dttm"].min().date()
all_max_date = suspicious["dttm"].max().date()
default_start = all_max_date
default_end = all_max_date

# ============================================================
# ======================  APP UI  ============================
# ============================================================
app = Dash(__name__)
app.title = "ACSS | Suspicious Activity"

app.layout = html.Div([
    html.H3("ACSS | Suspicious Activity Dashboard"),

    # Date range (defaults to latest date only)
    html.Div([
        html.Div("Date range", style={"fontWeight": 600, "marginRight": 8}),
        dcc.DatePickerRange(
            id="picker-range",
            min_date_allowed=all_min_date,
            max_date_allowed=all_max_date,
            start_date=default_start,
            end_date=default_end,
            display_format="YYYY-MM-DD",
            clearable=True,
        ),
    ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "12px"}),

    # Stores for filtered suspicious report (we don't need to store account_logs for your current flows)
    dcc.Store(id="store-suspicious-filtered"),

    html.Hr(),

    # ===================== Table 1 =====================
    html.H4("Users by Day (summary)"),
    dag.AgGrid(
        id="tbl-users-by-day",
        rowData=[],  # filled by date filter callback
        columnDefs=[
            {"headerName": "date", "field": "date", "sortable": True, "filter": True},
            {"headerName": "user", "field": "user", "sortable": True, "filter": True},
            {"headerName": "suspicious_accounts_for_day", "field": "suspicious_accounts_for_day",
             "type": "numericColumn", "sortable": True, "filter": "agNumberColumnFilter"},
            {"headerName": "active_days", "field": "active_days",
             "type": "numericColumn", "sortable": True, "filter": "agNumberColumnFilter"},
            {"headerName": "first_seen", "field": "first_seen", "sortable": True, "filter": "agDateColumnFilter"},
            {"headerName": "last_seen", "field": "last_seen", "sortable": True, "filter": "agDateColumnFilter"},
        ],
        defaultColDef={
            "resizable": True,
            "sortable": True,
            "filter": True,
            "floatingFilter": True,
        },
        dashGridOptions={"rowSelection": "single", "rowHeight": 34, "animateRows": True},
        style={"height": "380px", "width": "100%"},
    ),

    html.Hr(),

    # ===================== Table 2 =====================
    html.H4("User Drill-down (suspicious report rows)"),
    html.Div(id="user-selected", style={"marginBottom": 6}),
    dag.AgGrid(
        id="tbl-user-drill",
        rowData=[],
        columnDefs=[
            {"headerName": "dttm", "field": "dttm"},
            {"headerName": "user", "field": "user"},
            {"headerName": "account", "field": "account"},
            {"headerName": "id_chat", "field": "id_chat"},
            {"headerName": "date", "field": "date"},
            {"headerName": "dept_fctn_id", "field": "dept_fctn_id"},
            {"headerName": "dept_fctn_desc", "field": "dept_fctn_desc"},
            {"headerName": "emp_status_ind", "field": "emp_status_ind"},
            {"headerName": "user_comm_id", "field": "user_comm_id"},
            {"headerName": "job_desc", "field": "job_desc"},
            {"headerName": "job_cd", "field": "job_cd"},
            {"headerName": "iso_country_cd", "field": "iso_country_cd"},
            {"headerName": "num_of_unique_suspicious_accounts", "field": "num_of_unique_suspicious_accounts", "type": "numericColumn"},
        ],
        defaultColDef={"resizable": True, "sortable": True, "filter": True, "floatingFilter": True},
        dashGridOptions={"rowSelection": "single", "rowHeight": 32, "enableCellTextSelection": True},
        style={"height": "380px", "width": "100%"},
    ),

    html.Hr(),

    # ===================== Table 3 =====================
    html.H4("Account Drill-down (suspicious report rows)"),
    html.Div(id="account-selected", style={"marginBottom": 6}),
    dag.AgGrid(
        id="tbl-account-drill",
        rowData=[],
        columnDefs=[
            {"headerName": "dttm", "field": "dttm"},
            {"headerName": "user", "field": "user"},
            {"headerName": "account", "field": "account"},
            {"headerName": "id_chat", "field": "id_chat"},
            {"headerName": "date", "field": "date"},
            {"headerName": "dept_fctn_id", "field": "dept_fctn_id"},
            {"headerName": "dept_fctn_desc", "field": "dept_fctn_desc"},
            {"headerName": "emp_status_ind", "field": "emp_status_ind"},
            {"headerName": "user_comm_id", "field": "user_comm_id"},
            {"headerName": "job_desc", "field": "job_desc"},
            {"headerName": "job_cd", "field": "job_cd"},
            {"headerName": "iso_country_cd", "field": "iso_country_cd"},
            {"headerName": "num_of_unique_suspicious_accounts", "field": "num_of_unique_suspicious_accounts", "type": "numericColumn"},
        ],
        defaultColDef={"resizable": True, "sortable": True, "filter": True, "floatingFilter": True},
        dashGridOptions={"rowSelection": "single", "rowHeight": 32, "animateRows": True},
        style={"height": "380px", "width": "100%"},
    ),
])

# ============================================================
# =====================  CALLBACKS  ==========================
# ============================================================

# 1) Apply date range → populate Store + Table 1
@callback(
    Output("store-suspicious-filtered", "data"),
    Output("tbl-users-by-day", "rowData"),
    Input("picker-range", "start_date"),
    Input("picker-range", "end_date"),
)
def cb_date_filter(start_date, end_date):
    # guard cleared picker -> reset to defaults
    if not start_date or not end_date:
        start_date, end_date = default_start, default_end

    susp_range = filter_suspicious_by_range(suspicious, start_date, end_date)

    # Build Table 1 (users by day)
    tbl1 = build_users_by_day_table(susp_range)
    # Stringify datetime columns for JSON safety
    tbl1 = stringify_dt(tbl1, cols=("first_seen", "last_seen"))
    susp_out = stringify_dt(susp_range, cols=("dttm",))  # store keeping strings

    return susp_out.to_dict("records"), tbl1.to_dict("records")

# 2) Row click in Table 1 → populate Table 2 (user drill)
@callback(
    Output("user-selected", "children"),
    Output("tbl-user-drill", "rowData"),
    Output("tbl-user-drill", "selectedRows"),   # reset selection when new user picked
    Output("tbl-account-drill", "rowData"),     # clear Table 3 when user changes
    Input("tbl-users-by-day", "selectedRows"),  # AG Grid prop (list of dicts)
    State("store-suspicious-filtered", "data"),
)
def cb_user_drill(selected_rows, susp_store):
    if not selected_rows or not susp_store:
        return "Select a row above to drill into a user.", [], [], []

    sel_user = selected_rows[0].get("user")
    # filter suspicious (store contains strings for dttm)
    susp_df = pd.DataFrame(susp_store)
    # keep only this user
    user_df = susp_df[susp_df["user"] == sel_user].copy()

    # ensure ordering, then keep the same column set as Table 2 definitions
    if "dttm" in user_df.columns:
        user_df["dttm"] = pd.to_datetime(user_df["dttm"], errors="coerce")
        user_df = user_df.sort_values("dttm")
        user_df["dttm"] = user_df["dttm"].dt.strftime("%Y-%m-%d %H:%M:%S")

    header = f"User: {sel_user}"
    return header, user_df.to_dict("records"), [], []  # clear Table 3

# 3) Row click in Table 2 (account selected) → Table 3 (account drill)
@callback(
    Output("account-selected", "children"),
    Output("tbl-account-drill", "rowData"),
    Input("tbl-user-drill", "selectedRows"),     # AG Grid prop
    State("store-suspicious-filtered", "data"),
)
def cb_account_drill(selected_rows, susp_store):
    if not selected_rows or not susp_store:
        return "Select an account in the table above.", []

    sel_acct = selected_rows[0].get("account")
    susp_df = pd.DataFrame(susp_store)

    acc_df = susp_df[susp_df["account"] == sel_acct].copy()
    if "dttm" in acc_df.columns:
        acc_df["dttm"] = pd.to_datetime(acc_df["dttm"], errors="coerce")
        acc_df = acc_df.sort_values("dttm")
        acc_df["dttm"] = acc_df["dttm"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return f"Account: {sel_acct}", acc_df.to_dict("records")

# ============================================================
# =======================  MAIN  =============================
# ============================================================
if __name__ == "__main__":
    # pip install dash dash-ag-grid pandas
    app.run_server(debug=True)
