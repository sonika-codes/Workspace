\import math
import networkx as nx
import pandas as pd
from collections import defaultdict
from itertools import combinations

def _H_and_Hnorm_from_weights(weights):
    """Helper: Shannon entropy (bits) + normalized entropy in [0,1] from a list of weights."""
    weights = [w for w in weights if w > 0]
    k = len(weights)
    if k == 0:
        return 0.0, 0.0
    total = float(sum(weights))
    probs = [w / total for w in weights]
    H = -sum(p * math.log(p, 2) for p in probs)
    H_norm = 0.0 if k <= 1 else H / math.log(k, 2)
    return H, H_norm

def extract_neighbor_type_entropy(df):
    """
    Build a one-mode (user–user) projection from user–feature data and compute:
      • degree (weighted) – total connection strength to other users
      • clustering (weighted) – how interconnected the user's neighbors are
      • neighbor_type_entropy – diversity (in bits) of neighbor user TYPES
      • neighbor_type_entropy_norm – entropy normalized to [0,1]

    Assumptions
    -----------
    - `df` has at least columns: 'user', 'user_type'. The following improve the projection:
         'remote_network', 'device_ip_address', 'cmd_keywords' (space-separated tokens).
    - Edge weight in the projected user–user graph = raw count of shared features
      (user_type, remote_network, device_ip, individual command tokens).
      More shared features → stronger (closer) connection.

    Metric Interpretations
    ----------------------
    DEGREE (weighted):
      Sum of edge weights to all neighbors.
      • High = strong total similarity to many users (generalist / highly overlapping behavior)
      • Low  = few overlaps (specialist / niche behavior)

    CLUSTERING COEFFICIENT (weighted):
      How interconnected a user's neighbors are with one another (triangle density with weight).
      • High = embedded in a tight group (neighbors also share many features among themselves)
      • Low  = acts as a bridge between otherwise separate groups

    NEIGHBOR-TYPE ENTROPY (normalized):
      Diversity of neighbor TYPES (e.g., human vs service), using edge weights as type strength.
      • High (~1.0) = connections spread evenly across types
      • Low  (~0.0) = connections concentrated in a single type

    Example
    -------
    >>> import pandas as pd
    >>> data = [
    ...   # user,  user_type,   remote_network, device_ip_address, cmd_keywords
    ...   ("alice", "human",     "10.1.0.0/16",  "10.1.2.3",       "show config"),
    ...   ("alice", "human",     "10.1.0.0/16",  "10.1.2.3",       "show status"),
    ...   ("bob",   "human",     "10.1.0.0/16",  "10.1.2.3",       "show config"),
    ...   ("bob",   "human",     "10.2.0.0/16",  "10.2.5.6",       "show status"),
    ...   ("svc1",  "service",   "10.2.0.0/16",  "10.2.5.6",       "config apply"),
    ...   ("svc1",  "service",   "10.2.0.0/16",  "10.2.5.6",       "show status"),
    ... ]
    >>> df = pd.DataFrame(data, columns=["user","user_type","remote_network","device_ip_address","cmd_keywords"])
    >>> gf = extract_neighbor_type_entropy(df)
    >>> gf.loc[["alice","bob","svc1"], ["degree","clustering","neighbor_type_entropy_norm","user_type"]]
               degree  clustering  neighbor_type_entropy_norm user_type
    alice         ...        ...                        ...      human
    bob           ...        ...                        ...      human
    svc1          ...        ...                        ...    service
    # (Exact numbers depend on shared features; the columns illustrate what to read.)

    Returns
    -------
    pandas.DataFrame
        Index = user
        Columns:
          - degree (float)
          - clustering (float)
          - neighbor_type_entropy (float, bits)
          - neighbor_type_entropy_norm (float in [0,1])
          - user_type (str)
    """
    if 'user_type' not in df.columns:
        raise ValueError("df must contain a 'user_type' column for neighbor-type entropy.")

    # ---- STEP 1: Build user -> feature sets (namespaced to avoid collisions)
    user_feats = defaultdict(set)
    for _, r in df.iterrows():
        u = str(r['user'])
        # Always namespace features to keep fields distinct in the projection
        if 'user_type' in r and pd.notna(r['user_type']):
            user_feats[u].add(f"user_type:{r['user_type']}")
        if 'remote_network' in r and pd.notna(r['remote_network']):
            user_feats[u].add(f"remote_network:{r['remote_network']}")
        if 'device_ip_address' in r and pd.notna(r['device_ip_address']):
            user_feats[u].add(f"device_ip:{r['device_ip_address']}")
        ck = r.get('cmd_keywords')
        if isinstance(ck, str) and ck.strip():
            for t in ck.split():
                user_feats[u].add(f"cmd:{t}")

    users = list(user_feats.keys())

    # ---- STEP 2: feature -> users index
    feat_users = defaultdict(list)
    for u, feats in user_feats.items():
        for f in feats:
            feat_users[f].append(u)

    # ---- STEP 3: accumulate shared-feature counts between user pairs
    inter = defaultdict(int)
    for us in feat_users.values():
        us = sorted(set(us))
        for a, b in combinations(us, 2):
            inter[(a, b)] += 1

    # ---- STEP 4: projected user–user graph with COUNT weights (closeness)
    G = nx.Graph()
    G.add_nodes_from(users)
    for (a, b), k in inter.items():
        if k > 0:
            G.add_edge(a, b, weight=float(k))

    # ---- STEP 5: metrics

    # DEGREE (weighted):
    #   Sum of edge weights to all neighbors.
    #   High = strong total similarity to many users; Low = niche/isolated behavior.
    degree_w = dict(G.degree(weight='weight'))

    # CLUSTERING COEFFICIENT (weighted):
    #   How interconnected the neighbors are (triangle density with weight).
    #   High = tight, cohesive neighborhood; Low = bridging across groups.
    clustering_w = nx.clustering(G, weight='weight')

    # NEIGHBOR-TYPE ENTROPY (and normalized):
    #   Diversity of neighbor TYPES weighted by edge strengths.
    #   High (~1) = evenly spread across types; Low (~0) = concentrated in one type.
    u2t = dict(zip(df['user'], df['user_type']))
    neigh_type_ent = {}
    neigh_type_ent_norm = {}
    for u in users:
        type_weights = defaultdict(float)
        for v in G.neighbors(u):
            t = u2t.get(v, 'unknown')
            type_weights[t] += G[u][v].get('weight', 1.0)
        H, Hn = _H_and_Hnorm_from_weights(type_weights.values())
        neigh_type_ent[u] = H
        neigh_type_ent_norm[u] = Hn

    # ---- STEP 6: package output
    gf = pd.DataFrame(index=users)
    gf['degree'] = pd.Series(degree_w)
    gf['clustering'] = pd.Series(clustering_w)
    gf['neighbor_type_entropy'] = pd.Series(neigh_type_ent)
    gf['neighbor_type_entropy_norm'] = pd.Series(neigh_type_ent_norm)
    gf['user_type'] = gf.index.map(lambda u: u2t.get(u, 'unknown'))

    return gf.fillna(0.0)
