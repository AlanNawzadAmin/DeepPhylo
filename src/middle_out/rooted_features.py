"""Compute rooted tree features (depth, subtree leaf count, normalized basis)
from an unrooted edge_index.

The unrooted tree has a trifurcating root (last row of edge_index).  Given the
root token leaf ID, we orient the tree by treating the root token's parent as the
true root, then compute per-node depth, subtree leaf count, and a normalized
basis matrix (each node's row = uniform average over its subtree leaves).

All operations are vectorized on-device — no Python loops over nodes.
"""

import torch


def _build_edge_pairs(edge_index: torch.Tensor) -> torch.Tensor:
    """Extract all undirected edges as [n_edges, 2] from the edge_index.

    edge_index: [n_nodes, 3].
      - Leaves: [parent, -1, -1]
      - Internals: [parent, child1, child2]
      - Unrooted root (last row): [child1, child2, child3]
    """
    n_nodes = edge_index.shape[0]
    device = edge_index.device
    node_ids = torch.arange(n_nodes, device=device)
    unrooted_root = n_nodes - 1

    pairs = []

    # Leaf and internal nodes: column 0 is parent (except for the unrooted root)
    mask_has_parent = node_ids != unrooted_root
    parents = edge_index[mask_has_parent, 0]
    children_of_parent = node_ids[mask_has_parent]
    valid = parents >= 0
    pairs.append(torch.stack([parents[valid], children_of_parent[valid]], dim=1))

    # Internal nodes (including unrooted root): columns 1,2 are children
    # Unrooted root: all 3 columns are children
    for col in range(3):
        if col == 0:
            # Column 0 is child only for the unrooted root
            src = unrooted_root * torch.ones(1, dtype=torch.long, device=device)
            dst = edge_index[unrooted_root, 0:1]
        else:
            dst = edge_index[:, col]
            src = node_ids
        valid = dst >= 0
        if valid.any():
            pairs.append(torch.stack([src[valid], dst[valid]], dim=1))

    all_pairs = torch.cat(pairs, dim=0)  # [E, 2] directed
    # Make undirected: add reverse
    all_pairs = torch.cat([all_pairs, all_pairs.flip(1)], dim=0)
    # Deduplicate
    all_pairs = torch.unique(all_pairs, dim=0)
    return all_pairs  # [E_undir, 2]


def compute_rooted_features(
    edge_index: torch.Tensor,
    root_token_id: int,
    n_leaves: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute depth, subtree leaf count, and normalized basis for every node.

    Args:
        edge_index: [n_nodes, 3] — unrooted tree edge encoding.
        root_token_id: Leaf index of the root token (always present).
        n_leaves: Number of leaves (IDs 0..n_leaves-1).

    Returns:
        depths:         [n_nodes] long — depth from rooted root (0 = root).
        subtree_counts: [n_nodes] long — number of leaves in the subtree.
        basis:          [n_nodes, n_leaves] float — L2-normalized subtree membership.
                        basis[i] = 1/sqrt(subtree_counts[i]) for each leaf in i's subtree,
                        so each row has unit L2 norm and `basis @ leaf_features` stays
                        O(1) RMS regardless of subtree size.
    """
    n_nodes = edge_index.shape[0]
    device = edge_index.device

    # Rooted root = parent of root token in unrooted tree
    rooted_root = edge_index[root_token_id, 0].item()

    # Build undirected edge list
    edges = _build_edge_pairs(edge_index)  # [E, 2]
    src, dst = edges[:, 0], edges[:, 1]

    # --- BFS from rooted_root to assign depths ---
    depths = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
    depths[rooted_root] = 0
    parent_of = torch.full((n_nodes,), -1, dtype=torch.long, device=device)

    frontier = torch.tensor([rooted_root], device=device)
    depth = 0
    while frontier.numel() > 0:
        mask = torch.isin(src, frontier)
        nbrs = dst[mask]
        unvisited = depths[nbrs] == -1
        new_nodes = nbrs[unvisited].unique()
        depth += 1
        depths[new_nodes] = depth
        mask2 = torch.isin(src, frontier) & torch.isin(dst, new_nodes)
        parent_of[dst[mask2]] = src[mask2]
        frontier = new_nodes

    # --- Bottom-up: subtree counts + basis ---
    subtree_counts = torch.zeros(n_nodes, dtype=torch.long, device=device)
    subtree_counts[:n_leaves] = 1

    # Basis: leaves start one-hot, propagate up
    basis = torch.zeros(n_nodes, n_leaves, device=device)
    basis[:n_leaves, :n_leaves] = torch.eye(n_leaves, device=device)

    max_depth = depths.max().item()
    for d in range(max_depth, 0, -1):
        at_depth = (depths == d).nonzero(as_tuple=True)[0]
        parents = parent_of[at_depth]
        subtree_counts.scatter_add_(0, parents, subtree_counts[at_depth])
        # Accumulate basis rows into parents
        basis.index_add_(0, parents, basis[at_depth])

    basis = basis / subtree_counts.clamp(min=1).float().sqrt().unsqueeze(1)

    return depths, subtree_counts, basis
