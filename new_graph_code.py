import networkx as nx
from networkx.algorithms import bipartite
from itertools import chain

def user_projection_with_clustering(df):
    B = nx.Graph()

    # Build bipartite graph: users on one side, "feature" nodes on the other
    users = set(df['user'])
    B.add_nodes_from(users, bipartite=0)

    def feat(tag, val):   # keep namespaces so different fields don't collide
        return (tag, str(val))

    for _, row in df.iterrows():
        feats = [
            feat('user_type', row['user_type']),
            feat('remote_network', row['remote_network']),
            feat('device_ip', row['device_ip_address'])
        ]
        if isinstance(row.get('cmd_keywords', ''), str):
            feats += [feat('cmd', t) for t in row['cmd_keywords'].split()]
        # add edges user -> each feature
        for f in feats:
            B.add_node(f, bipartite=1)
            B.add_edge(row['user'], f)

    # Project to users; edge weight = number of shared features
    G_users = bipartite.weighted_projected_graph(B, users)

    # Now triangles can exist among users that share features with each other
    clustering = nx.clustering(G_users, weight='weight')
    degree = dict(G_users.degree())
    return nx.set_node_attributes(G_users, clustering, 'clustering'), degree, clustering
