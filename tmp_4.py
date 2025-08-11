import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ------------ helpers (short versions) ------------

def _proportion_based_clustering(df):
    freq = df.groupby(["command","account_type"]).size().unstack(fill_value=0)
    freq = freq.rename(columns={"human":"human_freq","functional":"functional_freq"})
    for col in ("human_freq","functional_freq"):
        if col not in freq: freq[col] = 0
    tot = (freq["human_freq"] + freq["functional_freq"]).replace(0, np.nan)
    props = pd.DataFrame({
        "human_prop": (freq["human_freq"]/tot).fillna(0.0),
        "functional_prop": (freq["functional_freq"]/tot).fillna(0.0)
    }, index=freq.index)
    X = props[["human_prop","functional_prop"]].to_numpy()
    km = KMeans(n_clusters=2, random_state=42)
    props["cluster"] = km.fit_predict(X)
    sil = silhouette_score(X, props["cluster"]) if len(np.unique(props["cluster"]))>1 else np.nan
    c0 = props[props.cluster==0][["human_prop","functional_prop"]].mean().to_numpy()
    c1 = props[props.cluster==1][["human_prop","functional_prop"]].mean().to_numpy()
    centroid_dist = float(np.linalg.norm(c0-c1))
    # purity per command + cluster means
    props["purity"] = props[["human_prop","functional_prop"]].max(axis=1)
    purity_by_cluster = props.groupby("cluster")["purity"].mean().rename("mean_purity")
    return props, {"silhouette": float(sil), "centroid_distance": centroid_dist,
                   "mean_purity_cluster0": float(purity_by_cluster.get(0, np.nan)),
                   "mean_purity_cluster1": float(purity_by_cluster.get(1, np.nan))}

def _hour_of_week_clustering(df, n_clusters=3):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour_of_week"] = df["timestamp"].dt.dayofweek * 24 + df["timestamp"].dt.hour
    mat = df.groupby(["command","hour_of_week"]).size().unstack(fill_value=0).sort_index(axis=1)
    mat = mat.div(mat.sum(axis=1), axis=0).fillna(0.0)
    scaler = StandardScaler(); X = scaler.fit_transform(mat.values)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if len(np.unique(labels))>1 else np.nan
    return mat.assign(cluster=labels), {"silhouette": float(sil)}

def _purity_table(props, cluster_id, top_n=10):
    sub = props[props["cluster"]==cluster_id].copy()
    return (sub.sort_values("purity", ascending=False)
               .head(top_n)[["human_prop","functional_prop","purity"]]
               .rename_axis("command"))

# ------------ simple synthetic baseline (same schema) ------------

def _generate_synthetic(
    n_humans=60, n_services=25, total_events=160_000,
    human_bus_hours=(8,18), noise=0.06, seed=7
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2025, 1, 6, 0, 0, 0)
    human_users = [f"u{i:03d}" for i in range(n_humans)]
    svc_users   = [f"svc{i:03d}" for i in range(n_services)]
    human_cmds  = [f"show cmd {i:03d}" for i in range(70)]
    svc_cmds    = ["backup config","sync metrics","deploy service prod","rotate logs",
                   "collect diagnostics","refresh cache","restart daemon","archive reports",
                   "svc op 008","svc op 009","svc op 010","svc op 011"]

    hours = np.arange(168); dow = hours//24; hod = hours%24
    wk_mask = dow<5; bh_mask = (hod>=human_bus_hours[0]) & (hod<human_bus_hours[1])
    w_h = np.ones_like(hours, float)*0.1; w_h[wk_mask & bh_mask]=1.0
    w_h = (1-noise)*(w_h/w_h.sum()) + noise*(np.ones_like(hours)/168)
    w_svc_uniform = np.ones_like(hours, float)/168

    rows = []
    H = int(total_events*0.6); S = total_events-H
    # humans: biased hours, diverse commands
    h_hours = rng.choice(hours, size=H, p=w_h)
    for h in h_hours:
        ts = start + timedelta(hours=int(h), minutes=int(rng.integers(0,60)))
        rows.append((ts, rng.choice(human_users), rng.choice(human_cmds), "human"))
    # services: static commands; mild per-command peaks
    for _ in range(S):
        cmd = rng.choice(svc_cmds)
        base = (hash(cmd)%24); peak = [(d*24 + (base+k)%24) for d in range(7) for k in range(3)]
        w = np.ones_like(hours,float)*0.2; w[peak]=1.0
        w = (1-noise)*(w/w.sum()) + noise*(np.ones_like(hours)/168)
        h = rng.choice(hours, p=w)
        ts = start + timedelta(hours=int(h), minutes=int(rng.integers(0,60)))
        rows.append((ts, rng.choice(svc_users), cmd, "functional"))
    df = pd.DataFrame(rows, columns=["timestamp","user","command","account_type"])
    return df.sort_values("timestamp").reset_index(drop=True)

# ------------ one-call comparator ------------

def compare_real_vs_synthetic(real_df: pd.DataFrame,
                              synthetic_df: pd.DataFrame | None = None,
                              n_hour_clusters: int = 3,
                              top_n: int = 8) -> dict:
    """
    Runs role-proportion and hour-of-week clustering on real data and a synthetic baseline.
    Prints side-by-side metrics; returns all artifacts.
    real_df must have columns: timestamp, user, command, account_type
    """
    syn = synthetic_df if synthetic_df is not None else _generate_synthetic()

    # Role proportions
    props_real, stats_props_real = _proportion_based_clustering(real_df)
    props_syn,  stats_props_syn  = _proportion_based_clustering(syn)

    # Hour-of-week
    hour_real, stats_hour_real = _hour_of_week_clustering(real_df, n_clusters=n_hour_clusters)
    hour_syn,  stats_hour_syn  = _hour_of_week_clustering(syn,     n_clusters=n_hour_clusters)

    # Pretty print summary
    summary = pd.DataFrame({
        "metric": [
            "role_silhouette", "role_centroid_distance",
            "role_mean_purity_cluster0", "role_mean_purity_cluster1",
            "hour_of_week_silhouette"
        ],
        "real": [
            stats_props_real["silhouette"],
            stats_props_real["centroid_distance"],
            stats_props_real["mean_purity_cluster0"],
            stats_props_real["mean_purity_cluster1"],
            stats_hour_real["silhouette"]
        ],
        "synthetic": [
            stats_props_syn["silhouette"],
            stats_props_syn["centroid_distance"],
            stats_props_syn["mean_purity_cluster0"],
            stats_props_syn["mean_purity_cluster1"],
            stats_hour_syn["silhouette"]
        ]
    })
    print("\n=== Side-by-side metrics (Real vs Synthetic) ===")
    print(summary.to_string(index=False))

    # Top purity tables (role-proportion)
    print("\n--- Real: top role‑pure commands per cluster ---")
    for c in sorted(props_real["cluster"].unique()):
        print(f"\nCluster {c}")
        print(_purity_table(props_real, c, top_n=top_n).head(top_n).to_string())

    print("\n--- Synthetic: top role‑pure commands per cluster ---")
    for c in sorted(props_syn["cluster"].unique()):
        print(f"\nCluster {c}")
        print(_purity_table(props_syn, c, top_n=top_n).head(top_n).to_string())

    return {
        "summary_table": summary,
        "real": {"props": props_real, "hour_mat": hour_real,
                 "stats_props": stats_props_real, "stats_hour": stats_hour_real},
        "synthetic": {"props": props_syn, "hour_mat": hour_syn,
                      "stats_props": stats_props_syn, "stats_hour": stats_hour_syn}
    }
