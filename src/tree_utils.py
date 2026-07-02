"""Shared tree-topology utilities (CPU-friendly, no grad).

Primary contents: squaretrick absorption basis. For each node v and leaf l,
returns P(random walk from v, absorbing at leaves, hits l first). Each row
sums to 1. Unrooted, purely topological — intended as a permutation-equivariant
companion to the rooted subtree basis.
"""
from __future__ import annotations

import torch


def squaretrick_basis_from_edges(
    n_nodes: int,
    edges: torch.LongTensor,
    leaf_idx: torch.LongTensor,
    max_iters: int = 64,
    tol: float = 1e-5,
) -> torch.Tensor:
    """Random-walk absorption basis [n_nodes, n_leaves].

    Args:
        n_nodes: total node count.
        edges: [E, 2] long tensor of edges (either direction; symmetrized inside).
        leaf_idx: [n_leaves] long tensor of absorbing node indices.
    """
    device = edges.device
    dtype = torch.float32
    n_leaves = int(leaf_idx.numel())

    adj = torch.zeros(n_nodes, n_nodes, device=device, dtype=dtype)
    if edges.numel() > 0:
        src, dst = edges[:, 0], edges[:, 1]
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0
    deg = adj.sum(dim=1).clamp(min=1)
    A = adj / deg.unsqueeze(1)

    leaf_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    leaf_mask[leaf_idx] = True
    A[leaf_mask] = 0
    A[leaf_idx, leaf_idx] = 1.0

    X = torch.full((n_nodes, n_leaves), 1.0 / max(n_leaves, 1), device=device, dtype=dtype)
    X[leaf_idx] = 0.0
    X[leaf_idx, torch.arange(n_leaves, device=device)] = 1.0

    for _ in range(max_iters):
        X_old = X
        A = A @ A
        X = A @ X_old
        if (X - X_old).abs().max() <= tol:
            break
    return X / X.norm(dim=1, keepdim=True).clamp(min=1e-9)


def squaretrick_basis_from_edge_index(
    edge_index: torch.LongTensor, n_leaves: int
) -> torch.Tensor:
    """Middle-out-format adapter. edge_index: [n_nodes, 3] with -1 padding.

    Leaves occupy the first n_leaves rows.
    """
    n_nodes = edge_index.shape[0]
    device = edge_index.device
    rows = torch.arange(n_nodes, device=device).unsqueeze(-1).expand(-1, 3)
    valid = edge_index >= 0
    edges = torch.stack([rows[valid], edge_index[valid]], dim=-1)
    leaf_idx = torch.arange(n_leaves, device=device)
    return squaretrick_basis_from_edges(n_nodes, edges, leaf_idx)


def squaretrick_basis_from_dgl(graph, leaf_idx: torch.LongTensor) -> torch.Tensor:
    """DGL graph adapter. Uses graph.edges() (direction irrelevant; symmetrized)."""
    src, dst = graph.edges()
    edges = torch.stack([src, dst], dim=-1)
    return squaretrick_basis_from_edges(graph.num_nodes(), edges, leaf_idx)
