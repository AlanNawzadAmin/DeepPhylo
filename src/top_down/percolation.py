"""Percolation graph operations: enums, graph population, observations, targets, constraints."""

from enum import Enum

import dgl
import torch

from ..bottom_up.tree_utils import pairwise_hamming_frac
from ..spd_attn import build_basis_from_graph
from .graph_ops import build_obs_graph_aux, get_hosts
from .pda import _pda_left_prob_from_counts
from .sampling import sample_X_t


def compute_pda_log_prior(host_idx_in_obs, left_child, right_child, subtree_counts, host_counts):
    """Precompute PDA log-prior offset [n_leaves, 2] on CPU."""
    n_leaves = host_idx_in_obs.numel()
    L = torch.zeros(n_leaves, dtype=torch.long)
    R = torch.zeros(n_leaves, dtype=torch.long)

    left_at_host = left_child[host_idx_in_obs]
    has_left = left_at_host.ne(-1)
    if has_left.any():
        L[has_left] = subtree_counts[left_at_host[has_left]].long()

    right_at_host = right_child[host_idx_in_obs]
    has_right = right_at_host.ne(-1)
    if has_right.any():
        R[has_right] = subtree_counts[right_at_host[has_right]].long()

    M = host_counts[host_idx_in_obs].long()
    p_left = _pda_left_prob_from_counts(L, R, M)
    p_left = p_left.clamp(min=1e-6, max=1 - 1e-6)
    return torch.stack((torch.log(p_left), torch.log1p(-p_left)), dim=1)


def build_obs_basis(obs_graph, n_leaves, host_idx_in_obs):
    """Build [n_obs, n_leaves, 2] stacked basis.

    Channel 0: rooted subtree basis (rows sum to 1 over leaves in subtree).
    Channel 1: unrooted random-walk absorption basis (squaretrick). To give
    every taxon a distinct absorbing column (top down hosts can host multiple
    taxa), we augment obs_graph with one pendant-per-taxon attached to that
    taxon's host, and use the pendants as the absorbing states. Hosts stay
    internal, so a host's row still reflects its topological position w.r.t.
    taxa at other hosts — not just the trivial "1/K on own taxa, 0 elsewhere".

    Sampling maintains `basis_raw` incrementally in _percolate_leaf; if it's
    present we normalize it for channel 0. Otherwise (training-time snapshot
    graphs) fall back to the prop_nodes_topo recompute.
    """
    from ..tree_utils import squaretrick_basis_from_edges

    if "basis_raw" in obs_graph.ndata:
        raw = obs_graph.ndata["basis_raw"]
        rooted = raw / raw.sum(dim=1, keepdim=True).clamp(min=1)
    else:
        rooted = build_basis_from_graph(obs_graph, host_idx_in_obs)

    device = rooted.device
    n_obs = obs_graph.num_nodes()
    host = host_idx_in_obs.to(device)
    src, dst = obs_graph.edges()
    pendant_ids = torch.arange(n_obs, n_obs + n_leaves, device=device)
    aug_edges = torch.cat([
        torch.stack([src.to(device), dst.to(device)], dim=-1),
        torch.stack([pendant_ids, host], dim=-1),
    ], dim=0)
    sq = squaretrick_basis_from_edges(
        n_obs + n_leaves, aug_edges, pendant_ids,
    ).to(device)
    topo = sq[:n_obs]  # drop pendants' own (identity) rows
    return torch.stack([rooted, topo], dim=-1)


class PercolationDirection(Enum):
    NEITHER = 0
    LEFT = 1
    RIGHT = 2


class LeafConstraint(Enum):
    FINAL = 0
    WAITING = 1
    MUST_LEFT = 2
    MUST_RIGHT = 3
    FREE = 4
    SYMMETRIC = 5


def populate_leafset(graph, leaf_idx, counts):
    n_leaves = leaf_idx.numel()
    graph.ndata["leafset"] = torch.full(
        (graph.num_nodes(), n_leaves), -1, dtype=torch.long, device=graph.device
    )
    graph.ndata["leafset"][leaf_idx, torch.arange(n_leaves, device=graph.device)] = (
        counts
    )
    dgl.prop_nodes_topo(
        graph,
        lambda edges: {"m": edges.src["leafset"]},
        lambda nodes: {
            "leafset": torch.maximum(
                (nodes.mailbox["m"] - 1).max(dim=1)[0], nodes.data["leafset"]
            )
        },
    )
    host_counts = graph.ndata["leafset"].eq(0).sum(dim=0)
    assert (host_counts == 1).all().item(), "Taxon has multiple hosts"


def get_obs_graph(graph):
    graph.ndata["have"] = graph.ndata["leafset"].eq(0).sum(dim=1)
    compute_subtree_counts(graph)
    graph.ndata["observed"] = graph.ndata["subtree_count"] > 0
    obs_graph = dgl.node_subgraph(graph, graph.ndata["observed"])
    return obs_graph


def compute_subtree_counts(graph):
    graph.ndata["subtree_count"] = graph.ndata["have"].float().clone()
    dgl.prop_nodes_topo(
        graph,
        message_func=lambda e: {"m": e.src["subtree_count"]},
        reduce_func=lambda n: {
            "subtree_count": n.mailbox["m"].sum(dim=1) + n.data["subtree_count"]
        },
    )


def compute_target(graph, leaf_idx):
    n_leaves = leaf_idx.numel()
    target = torch.full(
        (n_leaves,),
        PercolationDirection.NEITHER.value,
        device=graph.device,
    )
    leaf_positions = torch.arange(n_leaves, device=graph.device)
    leaf_dirs = torch.zeros(
        graph.num_nodes(), n_leaves, dtype=torch.long, device=graph.device
    )
    leaf_dirs[leaf_idx, leaf_positions] = -2
    graph.ndata["_leaf_dirs_tmp"] = leaf_dirs
    dgl.prop_nodes_topo(
        graph,
        message_func=lambda e: {
            "m": (e.src["_leaf_dirs_tmp"] != 0).long()
            * torch.where(e.data["direction"] == 0, -1, 1).view(-1, 1)
        },
        reduce_func=lambda n: {
            "_leaf_dirs_tmp": torch.where(
                n.data["_leaf_dirs_tmp"] != 0,
                n.data["_leaf_dirs_tmp"],
                n.mailbox["m"].sum(dim=1),
            )
        },
    )
    leaf_dirs = graph.ndata.pop("_leaf_dirs_tmp")

    hosts = graph.ndata["leafset"][:, leaf_positions].eq(0).float().argmax(dim=0)
    leaf_dir_at_host = leaf_dirs[hosts, leaf_positions]
    valid_host = leaf_dir_at_host != -2
    target[valid_host] = torch.where(
        leaf_dir_at_host[valid_host] == 1,
        PercolationDirection.RIGHT.value,
        PercolationDirection.LEFT.value,
    )
    return target


def compute_constraints(
    obs_graph,
    leaf_to_host,
    no_observed_children,
    left_child,
    right_child,
):
    device = obs_graph.device
    n_leaves = leaf_to_host.numel()
    host_depth = obs_graph.ndata["d"][leaf_to_host]

    constraints = torch.full(
        (n_leaves,), LeafConstraint.FREE.value, device=device, dtype=torch.long
    )

    # FINAL: single leaf at host with no observed children.
    single_leaf_at_host = (
        leaf_to_host.bincount(minlength=obs_graph.num_nodes())[leaf_to_host] == 1
    )
    final_mask = single_leaf_at_host & no_observed_children[leaf_to_host]
    constraints[final_mask] = LeafConstraint.FINAL.value

    # Noise depth = shallowest host depth among non-FINAL leaves.
    non_final = ~final_mask
    if non_final.any():
        noise_depth = host_depth[non_final].min().item()
    else:
        return constraints
    at_noise_layer = host_depth == noise_depth

    # WAITING: not at the noise layer and not final.
    waiting_mask = ~at_noise_layer & non_final
    constraints[waiting_mask] = LeafConstraint.WAITING.value

    # SYMMETRIC: at noise layer, host has no observed children, not already FINAL.
    symmetric_mask = at_noise_layer & no_observed_children[leaf_to_host] & ~final_mask
    constraints[symmetric_mask] = LeafConstraint.SYMMETRIC.value

    # MUST: single leaf at noise-layer host with only one observed child.
    has_left_child = left_child[leaf_to_host].ne(-1)
    has_right_child = right_child[leaf_to_host].ne(-1)
    must_left_mask = single_leaf_at_host & at_noise_layer & has_right_child & ~has_left_child
    must_right_mask = single_leaf_at_host & at_noise_layer & has_left_child & ~has_right_child
    constraints[must_left_mask] = LeafConstraint.MUST_LEFT.value
    constraints[must_right_mask] = LeafConstraint.MUST_RIGHT.value

    return constraints


def shuffle_left_right_edges(graph):
    _, v = graph.edges()
    assert v[::2].eq(v[1::2]).all(), (
        "Expected edges to be ordered as (left, right) pairs"
    )

    left_first = torch.randint(2, (graph.num_edges() // 2,), device=graph.device).bool()
    graph.edata["direction"] = torch.zeros(
        graph.num_edges(), dtype=torch.bool, device=graph.device
    )
    graph.edata["direction"][0::2] = left_first
    graph.edata["direction"][1::2] = ~left_first
    return graph


def collate_fn_preprocessed(batch, alpha, biased_schedule=True):
    assert len(batch) == 1, "Batch size > 1 not supported in preprocessed collate_fn"
    graphs, n_is_weights = zip(*batch)
    n_is_weight = n_is_weights[0]
    graph = dgl.batch(graphs)
    graph = shuffle_left_right_edges(graph)
    leaf_idx = torch.where(graph.in_degrees().eq(0))[0]
    obs_graph, t, T, t_depth, t_layer, target, importance_weight = sample_X_t(graph, alpha, biased_schedule=biased_schedule)

    cols_to_keep = graph.ndata["x"].sum(dim=0).gt(0)
    graph.ndata["x"] = graph.ndata["x"][:, cols_to_keep]
    seqs = graph.ndata["x"][leaf_idx]

    n_obs_per_graph = obs_graph.batch_num_nodes()
    leaf_idx = torch.where(graph.in_degrees().eq(0))[0]
    node_to_tree_id = torch.repeat_interleave(
        torch.arange(graph.batch_size, device=graph.device), graph.batch_num_nodes()
    )
    leaf_to_tree_id = node_to_tree_id[leaf_idx]
    tree_to_n_leaves = torch.bincount(leaf_to_tree_id, minlength=graph.batch_size)
    leaf_idx_within_tree = torch.cat(
        [torch.arange(n, device=graph.device) for n in tree_to_n_leaves]
    )
    hosts = get_hosts(graph, leaf_idx, leaf_idx_within_tree)
    host_idx_in_obs = graph.ndata["observed"].cumsum(0)[hosts] - 1

    spd, no_observed_children, left_child, right_child = build_obs_graph_aux(
        [obs_graph],
        tree_to_n_leaves,
        n_obs_per_graph,
        host_idx_in_obs,
        leaf_to_tree_id,
    )

    constraints = compute_constraints(
        obs_graph,
        host_idx_in_obs,
        no_observed_children,
        left_child,
        right_child,
    )

    obs_basis = build_obs_basis(obs_graph, seqs.shape[0], host_idx_in_obs)
    hamming_leaf = pairwise_hamming_frac(seqs.long())
    pda_log_prior = compute_pda_log_prior(
        host_idx_in_obs, left_child, right_child,
        obs_graph.ndata["subtree_count"], obs_graph.ndata["have"],
    )

    return (
        obs_graph,
        spd,
        t,
        t_depth,
        t_layer,
        target,
        T,
        host_idx_in_obs,
        tree_to_n_leaves,
        seqs,
        constraints,
        no_observed_children,
        n_obs_per_graph,
        left_child,
        right_child,
        importance_weight * n_is_weight,
        obs_basis,
        hamming_leaf,
        pda_log_prior,
    )
