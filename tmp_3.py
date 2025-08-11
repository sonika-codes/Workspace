# Fix minor syntax error in plt.figure call (tuple vs comma-separated)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def generate_synthetic_tacacs(
    n_humans=50,
    n_services=20,
    total_events=120_000,
    human_weekday_bias=0.8,
    human_bus_hours=(8, 18),
    human_command_pool_size=60,
    service_command_pool_size=12,
    service_has_time_peaks=True,
    noise=0.05,
    seed=42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2025, 1, 6, 0, 0, 0)
    human_users = [f"u{i:03d}" for i in range(n_humans)]
    service_users = [f"svc{i:03d}" for i in range(n_services)]

    human_cmds = [f"show cmd {i:03d}" for i in range(human_command_pool_size)]
    human_cmds[:10] = [
        "show ip route","show arp","show users","show interface status","ping 8.8.8.8",
        "show mac address-table","show running-config","traceroute 1.1.1.1",
        "show vlan brief","show lldp neighbors"
    ]
    service_cmds = [f"svc op {i:03d}" for i in range(service_command_pool_size)]
    service_cmds[:8] = [
        "backup config","sync metrics","deploy service prod","rotate logs",
        "collect diagnostics","refresh cache","restart daemon","archive reports"
    ]

    hours = np.arange(168)
    dow = hours // 24
    hod = hours % 24

    human_hour_weights = np.ones_like(hours, dtype=float) * 0.1
    weekday_mask = dow < 5
    bus_mask = (hod >= human_bus_hours[0]) & (hod < human_bus_hours[1])
    human_hour_weights[weekday_mask & bus_mask] = 1.0
    human_hour_weights = (1 - noise) * (human_hour_weights / human_hour_weights.sum()) + noise * (np.ones_like(hours)/168)

    service_hour_weights = np.ones_like(hours, dtype=float) / 168.0

    rows = []
    svc_cmd_peaks = {}
    if service_has_time_peaks:
        for i, cmd in enumerate(service_cmds):
            base_hour = (i * 3) % 24
            peak_hours = [(d*24 + (base_hour + k) % 24) for d in range(7) for k in range(3)]
            w = np.ones_like(hours, dtype=float) * 0.2
            w[peak_hours] = 1.0
            w = (1 - noise) * (w / w.sum()) + noise * (np.ones_like(hours)/168)
            svc_cmd_peaks[cmd] = w

    human_events = int(total_events * 0.6)
    service_events = total_events - human_events

    human_hour_choices = rng.choice(hours, size=human_events, p=human_hour_weights)
    human_users_choices = rng.choice(human_users, size=human_events)
    human_cmd_choices = rng.choice(human_cmds, size=human_events)
    for h, u, c in zip(human_hour_choices, human_users_choices, human_cmd_choices):
        ts = start + timedelta(hours=int(h), minutes=int(rng.integers(0,60)), seconds=int(rng.integers(0,60)))
        rows.append((ts, u, c, "human"))

    service_cmd_choices = rng.choice(service_cmds, size=service_events)
    service_users_choices = rng.choice(service_users, size=service_events)
    for u, c in zip(service_users_choices, service_cmd_choices):
        p = svc_cmd_peaks[c] if service_has_time_peaks else service_hour_weights
        h = rng.choice(hours, p=p)
        ts = start + timedelta(hours=int(h), minutes=int(rng.integers(0,60)), seconds=int(rng.integers(0,60)))
        rows.append((ts, u, c, "functional"))

    df = pd.DataFrame(rows, columns=["timestamp","user","command","account_type"])
    return df.sort_values("timestamp").reset_index(drop=True)


def proportion_based_clustering(df):
    freq = df.groupby(["command","account_type"]).size().unstack(fill_value=0)
    freq = freq.rename(columns={"human":"human_freq","functional":"functional_freq"})
    if "human_freq" not in freq: freq["human_freq"] = 0
    if "functional_freq" not in freq: freq["functional_freq"] = 0

    total = (freq["human_freq"] + freq["functional_freq"]).replace(0, np.nan)
    props = pd.DataFrame({
        "human_prop": (freq["human_freq"]/total).fillna(0.0),
        "functional_prop": (freq["functional_freq"]/total).fillna(0.0)
    }, index=freq.index)

    X = props[["human_prop","functional_prop"]].to_numpy()
    km = KMeans(n_clusters=2, random_state=42)
    props["cluster"] = km.fit_predict(X)

    sil = silhouette_score(X, props["cluster"]) if len(np.unique(props["cluster"]))>1 else np.nan
    c0 = props[props.cluster==0][["human_prop","functional_prop"]].mean().to_numpy()
    c1 = props[props.cluster==1][["human_prop","functional_prop"]].mean().to_numpy()
    centroid_distance = float(np.linalg.norm(c0-c1))

    return props, {"silhouette": float(sil), "centroid_distance": centroid_distance}


def plot_proportion_scatter(props):
    plt.figure(figsize=(7,6))
    for c in sorted(props["cluster"].unique()):
        s = props[props["cluster"]==c]
        plt.scatter(s["human_prop"], s["functional_prop"], s=18, alpha=0.8, label=f"Cluster {c}")
    grid = np.linspace(0,1,100)
    plt.plot(grid, 1-grid, linestyle="--")
    plt.xlim(-0.02,1.02); plt.ylim(-0.02,1.02)
    plt.xlabel("Human proportion"); plt.ylabel("Functional proportion")
    plt.title("Command role-proportion clustering")
    plt.legend(); plt.tight_layout(); plt.show()


def purity_tables(props):
    t = props.copy()
    t["purity"] = t[["human_prop","functional_prop"]].max(axis=1)
    by_cluster = {}
    for c in sorted(t["cluster"].unique()):
        by_cluster[c] = t[t.cluster==c].sort_values("purity", ascending=False)
    return by_cluster


def hour_of_week_clustering(df, n_clusters=3):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour_of_week"] = df["timestamp"].dt.dayofweek * 24 + df["timestamp"].dt.hour

    mat = df.groupby(["command","hour_of_week"]).size().unstack(fill_value=0).sort_index(axis=1)
    mat = mat.div(mat.sum(axis=1), axis=0).fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(mat.values)

    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(X_scaled)
    mat["cluster"] = labels

    p = PCA(n_components=2, random_state=42)
    coords = p.fit_transform(X_scaled)
    mat["pc1"] = coords[:,0]; mat["pc2"] = coords[:,1]

    sil = silhouette_score(X_scaled, labels) if len(np.unique(labels))>1 else np.nan
    return mat, {"silhouette": float(sil)}


def plot_hour_of_week_scatter(mat):
    plt.figure(figsize=(7,5))
    for c in sorted(mat["cluster"].unique()):
        s = mat[mat["cluster"]==c]
        plt.scatter(s["pc1"], s["pc2"], s=18, alpha=0.8, label=f"Cluster {c}")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("Hour-of-week pattern clustering (PCA view)")
    plt.legend(); plt.tight_layout(); plt.show()


def plot_hour_of_week_heatmaps(original_df, clustered_mat):
    merged = original_df.copy()
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged["hour_of_week"] = merged["timestamp"].dt.dayofweek * 24 + merged["timestamp"].dt.hour

    clusters = clustered_mat[["cluster"]]
    merged = merged.merge(clusters, left_on="command", right_index=True, how="left")

    for c in sorted(clustered_mat["cluster"].unique()):
        sub = merged[merged["cluster"]==c]
        counts = sub.groupby("hour_of_week").size().reindex(range(168), fill_value=0)
        heat = counts.values.reshape(7,24)
        maxv = heat.max() if heat.max()>0 else 1.0
        heat = heat / maxv

        plt.figure(figsize=(10,3.8))
        plt.imshow(heat, aspect="auto")
        plt.colorbar()
        plt.title(f"Cluster {c} — Hour-of-week usage heatmap")
        plt.ylabel("Day of week (0=Mon)"); plt.xlabel("Hour of day (0-23)")
        plt.tight_layout(); plt.show()


# --- Demo run ---
synthetic_df = generate_synthetic_tacacs(
    n_humans=60, n_services=25, total_events=160_000,
    human_weekday_bias=0.8, human_bus_hours=(8,18),
    human_command_pool_size=70, service_command_pool_size=12,
    service_has_time_peaks=True, noise=0.06, seed=7
)

props, prop_stats = proportion_based_clustering(synthetic_df)
plot_proportion_scatter(props)
ptables = purity_tables(props)

hour_mat, hour_stats = hour_of_week_clustering(synthetic_df, n_clusters=3)
plot_hour_of_week_scatter(hour_mat)
plot_hour_of_week_heatmaps(synthetic_df, hour_mat)

for c, tab in ptables.items():
    print(f"\nRole-proportion — cluster {c}: top 8 by purity")
    print(tab.head(8)[["human_prop","functional_prop","purity"]])

print("\nStats:")
print("Proportion clustering:", prop_stats)
print("Hour-of-week clustering:", hour_stats)


