import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_hour_of_week_heatmap(df, role="human", normalize="global", title=None):
    """
    Plot an hour-of-week heatmap (0-167 -> 7x24) for a given account_type.

    normalize:
      - "none": raw counts
      - "global": divide by max cell value
      - "row": normalize per day (each row sums to 1 if any events that day)
      - "col": normalize per hour-of-day (each column scaled by its max)
    """
    d = df[df["account_type"] == role].copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp"])
    d["hour_of_week"] = d["timestamp"].dt.dayofweek * 24 + d["timestamp"].dt.hour

    # counts per hour_of_week (0..167)
    counts = d.groupby("hour_of_week").size().reindex(range(168), fill_value=0)
    mat = counts.values.reshape(7, 24)  # rows=Mon..Sun, cols=0..23

    if normalize == "global":
        m = mat.max()
        mat = mat / m if m > 0 else mat
    elif normalize == "row":
        row_sums = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, np.where(row_sums == 0, 1, row_sums))
    elif normalize == "col":
        col_max = mat.max(axis=0, keepdims=True)
        mat = np.divide(mat, np.where(col_max == 0, 1, col_max))
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be one of: none, global, row, col")

    plt.figure(figsize=(10, 3.6))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.title(title or f"{role.capitalize()} accounts — Hour-of-Week heatmap ({normalize})")
    plt.ylabel("Day of week (0=Mon … 6=Sun)")
    plt.xlabel("Hour of day (0–23)")
    plt.tight_layout()
    plt.show()


def compare_roles_hour_of_week(df, normalize="global"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 3.6))
    for ax, role in zip(axes, ["human", "functional"]):
        d = df[df["account_type"] == role].copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
        d = d.dropna(subset=["timestamp"])
        d["hour_of_week"] = d["timestamp"].dt.dayofweek * 24 + d["timestamp"].dt.hour

        counts = d.groupby("hour_of_week").size().reindex(range(168), fill_value=0)
        mat = counts.values.reshape(7, 24)

        if normalize == "global":
            m = mat.max()
            mat = mat / m if m > 0 else mat
        elif normalize == "row":
            row_sums = mat.sum(axis=1, keepdims=True)
            mat = np.divide(mat, np.where(row_sums == 0, 1, row_sums))
        elif normalize == "col":
            col_max = mat.max(axis=0, keepdims=True)
            mat = np.divide(mat, np.where(col_max == 0, 1, col_max))
        elif normalize == "none":
            pass
        else:
            raise ValueError("normalize must be one of: none, global, row, col")

        im = ax.imshow(mat, aspect="auto")
        ax.set_title(f"{role.capitalize()} ({normalize})")
        ax.set_ylabel("Day (0=Mon)")
        ax.set_xlabel("Hour (0–23)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.suptitle("Hour-of-Week usage by role")
    plt.tight_layout()
    plt.show()

def top_users_hour_of_week_grid(df, role="human", top_n=6, normalize="global"):
    d = df[df["account_type"] == role].copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp"])
    # pick top N by event count
    top_users = d["user"].value_counts().head(top_n).index.tolist()

    n = len(top_users)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5*cols, 3.2*rows))
    axes = np.atleast_2d(axes)

    for idx, u in enumerate(top_users):
        r, c = divmod(idx, cols)
        dd = d[d["user"] == u].copy()
        dd["hour_of_week"] = dd["timestamp"].dt.dayofweek * 24 + dd["timestamp"].dt.hour
        counts = dd.groupby("hour_of_week").size().reindex(range(168), fill_value=0)
        mat = counts.values.reshape(7, 24)
        if normalize == "global":
            m = mat.max()
            mat = mat / m if m > 0 else mat
        elif normalize == "row":
            row_sums = mat.sum(axis=1, keepdims=True)
            mat = np.divide(mat, np.where(row_sums == 0, 1, row_sums))
        elif normalize == "col":
            col_max = mat.max(axis=0, keepdims=True)
            mat = np.divide(mat, np.where(col_max == 0, 1, col_max))
        elif normalize == "none":
            pass

        axes[r, c].imshow(mat, aspect="auto")
        axes[r, c].set_title(f"{u} ({role})")
        axes[r, c].set_ylabel("Day"); axes[r, c].set_xlabel("Hour")

    # hide any empty subplots
    for k in range(n, rows*cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")

    fig.suptitle(f"Top {top_n} {role} users — Hour-of-Week heatmaps ({normalize})")
    plt.tight_layout()
    plt.show()

# 2-panel overview (recommended for standup)
compare_roles_hour_of_week(real_df, normalize="global")   # or "row"

# individual role
plot_hour_of_week_heatmap(real_df, role="human", normalize="row")
plot_hour_of_week_heatmap(real_df, role="functional", normalize="row")

# per-user grids
top_users_hour_of_week_grid(real_df, role="human", top_n=6, normalize="global")
top_users_hour_of_week_grid(real_df, role="functional", top_n=6, normalize="global")



# 2-panel overview (recommended for standup)
compare_roles_hour_of_week(real_df, normalize="global")   # or "row"

# individual role
plot_hour_of_week_heatmap(real_df, role="human", normalize="row")
plot_hour_of_week_heatmap(real_df, role="functional", normalize="row")

# per-user grids
top_users_hour_of_week_grid(real_df, role="human", top_n=6, normalize="global")
top_users_hour_of_week_grid(real_df, role="functional", top_n=6, normalize="global")
