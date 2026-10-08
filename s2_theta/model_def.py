"""S2-theta model definition: 18-spin Ising, Z1-legal mask from engineered S2."""
import numpy as np


K = 6   # actions
T = 3   # timesteps
N = K * T  # 18 spins

VIS_N = K  # t=0 spins are visible (clamped to expert action)


def build_mask():
    """Z1-legal sparsity mask from the engineered S2Agent J.

    Returns (edge_list, h_static_eng, J_eng) where edge_list has unique
    undirected edges (i<j) with nonzero engineered coupling.
    """
    import sys, os
    sys.path.insert(0, os.path.expanduser('~/workspace/flynav-redteam/thermopilot'))
    from harness import S2Agent
    a = S2Agent(n_sweeps=10, seed=0)
    J = a.J
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)
             if abs(J[i, j]) > 1e-12]
    return edges, a.h_static.copy(), J.copy()


def build_nodes_edges():
    """THRML nodes, edges, and DSATUR colorings."""
    from thrml import Block, SpinNode
    import networkx as nx
    edges_idx, h_eng, J_eng = build_mask()
    nodes = [SpinNode() for _ in range(N)]
    edges = [(nodes[i], nodes[j]) for (i, j) in edges_idx]
    vis_nodes = nodes[:VIS_N]
    data_set = set(vis_nodes)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    coloring = nx.coloring.greedy_color(G, strategy="DSATUR")
    n_colors = max(coloring.values()) + 1
    free_blocks = [Block([n for n, c in coloring.items() if c == k])
                   for k in range(n_colors)]
    free_blocks = [b for b in free_blocks if len(b.nodes) > 0]
    clamped_blocks = [Block([n for n in b.nodes if n not in data_set])
                      for b in free_blocks]
    clamped_blocks = [b for b in clamped_blocks if len(b.nodes) > 0]
    return nodes, edges, edges_idx, vis_nodes, free_blocks, clamped_blocks, h_eng, J_eng


def param_count(edges_idx):
    return N + len(edges_idx)  # 18 biases + 105 weights = 123
