import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def build_user_user_graph(df):
    from collections import defaultdict
    from itertools import combinations

    # user -> feature set
    user_feats = defaultdict(set)
    for _, r in df.iterrows():
        u = str(r['user'])
        if 'user_type' in r:        user_feats[u].add(f"user_type:{r['user_type']}")
        if 'remote_network' in r:   user_feats[u].add(f"remote_network:{r['remote_network']}")
        if 'device_ip_address' in r:user_feats[u].add(f"device_ip:{r['device_ip_address']}")
        ck = r.get('cmd_keywords')
        if isinstance(ck, str) and ck.strip():
            for t in ck.split():
                user_feats[u].add(f"cmd:{t}")

    # feature -> list(users)
    feat_users = defaultdict(list)
    for u, feats in user_feats.items():
        for f in feats:
            feat_users[f].append(u)

    # accumulate shared-feature counts
    inter = defaultdict(int)
    for us in feat_users.values():
        us = sorted(set(us))
        for a, b in combinations(us, 2):
            inter[(a, b)] += 1

    # build projected graph
    G = nx.Graph()
    G.add_nodes_from(user_feats.keys())
    for (a, b), k in inter.items():
        if k > 0:
            G.add_edge(a, b, weight=float(k))
    return G

def visualize_user_graph_sample(df, sample_users=150, min_edge_weight=1, seed=42):
    """Static preview of the densest slice of the graph."""
    G = build_user_user_graph(df)
    if G.number_of_nodes() == 0:
        print("Empty graph.")
        return

    # pick top users by weighted degree
    deg_w = dict(G.degree(weight='weight'))
    top_nodes = [n for n, _ in sorted(deg_w.items(), key=lambda x: x[1], reverse=True)[:sample_users]]
    H = G.subgraph(top_nodes).copy()

    # optional edge weight thresholding for clarity
    if min_edge_weight > 1:
        to_drop = [(u, v) for u, v, d in H.edges(data=True) if d.get('weight', 1.0) < min_edge_weight]
        H.remove_edges_from(to_drop)

    # layout
    pos = nx.spring_layout(H, seed=seed, k=None)  # force-directed

    # node aesthetics
    node_sizes = [max(80, 12*deg_w.get(n, 0)) for n in H.nodes()]
    # color by user_type if present
    u2t = dict(zip(df['user'], df['user_type'])) if 'user_type' in df.columns else {}
    types = [u2t.get(n, 'unknown') for n in H.nodes()]
    # map types → integers → colormap
    unique_types = {t:i for i, t in enumerate(sorted(set(types)))}
    node_colors = [unique_types[t] for t in types]

    # edge aesthetics
    ew = [max(0.5, d.get('weight', 1.0)**0.8) for _, _, d in H.edges(data=True)]

    plt.figure(figsize=(10, 8))
    nodes = nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color=node_colors, cmap='tab10')
    nx.draw_networkx_edges(H, pos, width=ew, alpha=0.35)
    # light labels only for the very top few to avoid clutter
    label_cut = set([n for n, _ in sorted(deg_w.items(), key=lambda x: x[1], reverse=True)[:min(20, len(H))]])
    nx.draw_networkx_labels(H, pos, labels={n:n for n in H.nodes() if n in label_cut}, font_size=9)
    plt.title("User–User Projection (sample)\nNode size = weighted degree, Edge width = shared features")
    plt.axis('off')
    # legend for user_type
    handles = [plt.Line2D([0],[0], marker='o', linestyle='', label=t, markersize=8) for t in unique_types]
    for h, t in zip(handles, unique_types.keys()):
        h.set_color(plt.cm.tab10(unique_types[t]))
    plt.legend(handles, unique_types.keys(), title="user_type", loc='lower left', bbox_to_anchor=(1.02, 0))
    plt.tight_layout()
    plt.show()
