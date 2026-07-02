"""Percolation sampling logic for top down diffusion."""

import math

import dgl
import numpy as np
import torch

from .graph_ops import compute_depths
from .pda import (
    _compute_I_scalar,
    _compute_qA_table,
    _neg_log_split_prob,
    _pda_denominator_term_scalar,
)


def compute_tau(t, alpha):
    return -torch.log(1 - t.clamp(max=1 - 1e-6)) / alpha


def compute_weight(t, alpha):
    tau = compute_tau(t, alpha)
    tau_prime = 1 / alpha / (1 - t.clamp(max=1 - 1e-6))
    return tau_prime / (torch.exp(tau) - 1 + 1e-6)


def compute_top_down_progress(leaf_depths, t_step):
    steps_per_layer = torch.bincount(leaf_depths).flip((0,)).cumsum(0).flip((0,))
    steps_deep_to_shallow = steps_per_layer[1:].flip(0)
    cumulative = steps_deep_to_shallow.cumsum(0)
    current_layer = torch.searchsorted(cumulative, t_step)
    t_depth = current_layer.float() / leaf_depths.max()
    prior_steps = torch.cat([cumulative.new_zeros(1), cumulative])[current_layer]
    L = steps_deep_to_shallow[current_layer]
    pos = t_step - prior_steps
    t_layer = pos / L
    return t_depth, t_layer


def _build_cluster_table(L, R, beta=-1.5):
    """Build the O(N^2) importance table  pi(M, a) = I(a, M-a) h(a|M)  for one cluster.

    Returns ``(weights, Ms, a_s)`` where *weights* is a 1-D float64 tensor
    of positive table entries and *Ms* / *a_s* are plain Python lists of the
    corresponding M and a values.  Returns ``(None, None, None)`` if the
    table is empty (e.g. L = R = 1 where KL = 0 everywhere).
    """
    N = L + R
    if N <= 1:
        return None, None, None

    M_vals = torch.arange(1, N + 1, dtype=torch.float64)
    a_vals = torch.arange(0, L + 1, dtype=torch.float64)
    M_grid = M_vals.unsqueeze(1).expand(N, L + 1)
    a_grid = a_vals.unsqueeze(0).expand(N, L + 1)
    b_grid = M_grid - a_grid

    valid = (a_grid >= 0) & (b_grid >= 0) & (a_grid <= L) & (b_grid <= R)

    log_fact = torch.zeros(N + 1, dtype=torch.float64)
    for i in range(1, N + 1):
        log_fact[i] = log_fact[i - 1] + math.log(i)

    def _log_comb(n, k):
        n_i = n.long().clamp(0, N)
        k_i = k.long().clamp(0, N)
        lc = log_fact[n_i] - log_fact[k_i] - log_fact[(n_i - k_i).clamp(0, N)]
        lc[k < 0] = -float("inf")
        lc[k > n] = -float("inf")
        return lc

    log_h = (
        _log_comb(torch.full_like(a_grid, L, dtype=torch.long), a_grid.long())
        + _log_comb(torch.full_like(b_grid, R, dtype=torch.long), b_grid.long())
        - _log_comb(torch.full_like(M_grid, N, dtype=torch.long), M_grid.long())
    )
    log_h[~valid] = -float("inf")
    h = log_h.exp()

    sA_grid = (L - a_grid).clamp(min=0)
    sB_grid = (R - b_grid).clamp(min=0)
    qA = _compute_qA_table(sA_grid, sB_grid, M_grid, valid, beta=beta)

    M_safe = M_grid.clamp(min=1)
    zeros = torch.zeros_like(a_grid, dtype=torch.float64)

    I_table = torch.zeros_like(a_grid, dtype=torch.float64)
    I_table += torch.where(a_grid > 0,
                           a_grid / M_safe * torch.log(1.0 / qA.clamp(min=1e-30)),
                           zeros)
    I_table += torch.where(b_grid > 0,
                           b_grid / M_safe * torch.log(1.0 / (1.0 - qA).clamp(min=1e-30)),
                           zeros)
    I_table[~valid] = 0.0
    I_table[(a_grid == L) & (b_grid == R)] = 0.0  # I(L, R) := 0

    pi_table = h * I_table
    pi_table[~valid] = 0.0

    pos = pi_table > 1e-30
    if not pos.any():
        return None, None, None

    weights = pi_table[pos]
    idx = pos.nonzero(as_tuple=False)
    Ms = (idx[:, 0] + 1).tolist()
    a_s = idx[:, 1].tolist()
    return weights, Ms, a_s


def _compute_tree_layer_info(graph, leaf_idx, leaf_depths, beta=-1.5):
    """Shared first stage: compute subtree L/R, -log p_PDA, and depth-aggregated pi."""
    n_nodes = graph.num_nodes()
    device = graph.device
    max_depth = leaf_depths.max().item()

    sc = torch.zeros(n_nodes, device=device)
    sc[leaf_idx] = 1.0
    graph.ndata["_sc"] = sc
    dgl.prop_nodes_topo(
        graph,
        message_func=lambda e: {"m": e.src["_sc"]},
        reduce_func=lambda n: {"_sc": n.mailbox["m"].sum(1) + n.data["_sc"]},
    )
    subtree_count = graph.ndata.pop("_sc")

    src, dst = graph.edges()
    is_left = graph.edata["direction"] == 0
    L_node = torch.zeros(n_nodes, device=device)
    R_node = torch.zeros(n_nodes, device=device)
    L_node.scatter_add_(0, dst[is_left], subtree_count[src[is_left]])
    R_node.scatter_add_(0, dst[~is_left], subtree_count[src[~is_left]])

    is_internal = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    is_internal[dst] = True
    int_idx = is_internal.nonzero(as_tuple=False).flatten()
    neg_log_p = _neg_log_split_prob(L_node[int_idx], R_node[int_idx], beta) - math.log(2)

    node_depths = graph.ndata["d"]
    depths_int = node_depths[int_idx].long()
    pi = torch.zeros(max_depth, device=device)
    pi.scatter_add_(0, depths_int, neg_log_p.float())
    pi = pi.clamp(min=1e-10)

    neg_log_p_per_node = torch.zeros(n_nodes, device=device)
    neg_log_p_per_node[int_idx] = neg_log_p.float()

    return dict(
        max_depth=max_depth,
        L_node=L_node,
        R_node=R_node,
        is_internal=is_internal,
        neg_log_p_per_node=neg_log_p_per_node,
        pi=pi,
        src=src,
        dst=dst,
    )


def sample_percolation_counts(graph, leaf_idx, leaf_depths, t, alpha, beta=-1.5):
    device = graph.device

    info = _compute_tree_layer_info(graph, leaf_idx, leaf_depths, beta)
    pi = info["pi"]

    d_idx = torch.multinomial(pi, 1).item()
    d = d_idx + 1

    active = leaf_depths >= d
    n_active = int(active.sum().item())
    descended = torch.zeros_like(leaf_depths, dtype=torch.bool)
    if n_active > 0:
        descended[active] = torch.bernoulli(
            torch.full((n_active,), torch.exp(-compute_tau(t, alpha)), device=device)
        ).bool()

    counts = torch.zeros_like(leaf_depths)
    not_descended = active & ~descended
    counts[not_descended] = leaf_depths[not_descended] - d + 1
    counts[descended] = leaf_depths[descended] - d

    importance_weight = n_active * pi.sum() / pi[d_idx]
    importance_weight = importance_weight * compute_weight(t, alpha)

    return counts, counts.sum(), int(leaf_depths.sum().item()), importance_weight


def sample_percolation_counts_old(graph, leaf_idx, leaf_depths, t, alpha, beta=-1.5):
    T = int(leaf_depths.sum().item())
    t_step = torch.randint(1, T + 1, (1,), device=graph.device).item()
    counts = torch.zeros_like(leaf_depths)
    for _ in range(t_step):
        max_depth = leaf_depths.max()
        candidates = (leaf_depths == max_depth).nonzero(as_tuple=False).flatten()
        chosen = candidates[
            torch.randint(0, candidates.numel(), (1,), device=graph.device)
        ]
        leaf_depths[chosen] -= 1
        counts[chosen] += 1
    importance_weight = T
    return counts, t_step, T, importance_weight


def _leaf_cluster_assignments(graph, info, leaf_idx, leaf_depths, d, cluster_nodes):
    """Walk active leaves up to depth *d* and return their cluster assignments.

    Returns ``(sides, cluster_ids, active_indices)`` where *sides* is 0/1
    (left/right child), *cluster_ids* maps each active leaf to a cluster
    index in ``[0, K)``, and *active_indices* indexes into the leaf arrays.
    """
    device = graph.device
    n_nodes = graph.num_nodes()
    node_depths = graph.ndata["d"]
    src, dst = info["src"], info["dst"]

    parent_of = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
    parent_of[src] = dst
    side_of_child = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
    side_of_child[src] = graph.edata["direction"].long()
    cluster_node_to_idx = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
    cluster_node_to_idx[cluster_nodes] = torch.arange(
        cluster_nodes.numel(), device=device,
    )

    active_indices = (leaf_depths >= d).nonzero(as_tuple=False).flatten()
    active_nodes = leaf_idx[active_indices].clone()
    cur_d = node_depths[active_nodes].long()
    while (cur_d > d).any():
        up = cur_d > d
        active_nodes[up] = parent_of[active_nodes[up]]
        cur_d = node_depths[active_nodes].long()

    sides = side_of_child[active_nodes]
    cluster_ids = cluster_node_to_idx[parent_of[active_nodes]]
    return sides, cluster_ids, active_indices


def _map_cluster_states_to_counts(
    graph, info, leaf_idx, leaf_depths, d,
    cluster_nodes, cluster_M, cluster_a,
):
    """Map per-cluster removal states (a_k, b_k) to per-leaf percolation counts."""
    device = graph.device
    K = cluster_nodes.numel()

    sides, cluster_ids, active_indices = _leaf_cluster_assignments(
        graph, info, leaf_idx, leaf_depths, d, cluster_nodes,
    )
    active = leaf_depths >= d

    descended = torch.zeros(leaf_idx.numel(), dtype=torch.bool, device=device)
    descended[active] = True
    for k in range(K):
        a_k = cluster_a[k]
        b_k = cluster_M[k] - a_k

        if a_k > 0:
            left_pos = ((cluster_ids == k) & (sides == 0)).nonzero(as_tuple=False).flatten()
            perm = torch.randperm(left_pos.numel(), device=device)[:a_k]
            descended[active_indices[left_pos[perm]]] = False

        if b_k > 0:
            right_pos = ((cluster_ids == k) & (sides == 1)).nonzero(as_tuple=False).flatten()
            perm = torch.randperm(right_pos.numel(), device=device)[:b_k]
            descended[active_indices[right_pos[perm]]] = False

    counts = torch.zeros_like(leaf_depths)
    counts[active & ~descended] = leaf_depths[active & ~descended] - d + 1
    counts[active & descended] = leaf_depths[active & descended] - d
    return counts


def _compute_sum_MI(cluster_M, cluster_a, clusters_L, clusters_R, beta):
    """Sum of M_k * I_k over clusters — shared by weight computations."""
    total = 0.0
    for k in range(len(cluster_M)):
        if cluster_M[k] > 0:
            total += cluster_M[k] * _compute_I_scalar(
                cluster_a[k], cluster_M[k] - cluster_a[k],
                clusters_L[k], clusters_R[k], cluster_M[k], beta,
            )
    return total


def sample_percolation_counts_biased(graph, leaf_idx, leaf_depths, beta=-1.5):
    """Sample percolation noise state using the biased importance schedule.

    Implements the 4-step sampling algorithm (Proposition 1 / Section 3):
      1. Sample focus cluster k* ~ -log p_PDA(L_k*, R_k*)
      2. Sample focus state (M_k*, a_k*) from per-cluster importance table
      3. Sample global removal count M from BetaBinomial
      4. Sample remaining cluster states from multivariate hypergeometric

    Returns ``(counts, t_step, T, importance_weight)`` with the same
    interface as :func:`sample_percolation_counts`.
    """
    device = graph.device
    max_depth = leaf_depths.max().item()

    if max_depth == 0:
        return (
            torch.zeros_like(leaf_depths), 0,
            int(leaf_depths.sum().item()),
            torch.tensor(1.0, device=device),
        )

    info = _compute_tree_layer_info(graph, leaf_idx, leaf_depths, beta)
    pi = info["pi"]
    Z_total = pi.sum().item()

    d_idx = torch.multinomial(pi, 1).item()
    d = d_idx + 1

    node_depths = graph.ndata["d"]
    cluster_mask = info["is_internal"] & (node_depths == d_idx)
    cluster_nodes = cluster_mask.nonzero(as_tuple=False).flatten()
    K = cluster_nodes.numel()

    assert K > 0, "Sampled depth has no internal nodes — pi and cluster_mask are inconsistent"

    clusters_L = info["L_node"][cluster_nodes].long().cpu().tolist()
    clusters_R = info["R_node"][cluster_nodes].long().cpu().tolist()
    clusters_neg_log_p = info["neg_log_p_per_node"][cluster_nodes].cpu().tolist()

    # Step 1: Sample focus cluster
    cluster_weights = torch.tensor(clusters_neg_log_p, dtype=torch.float64)
    cluster_weights = cluster_weights.clamp(min=1e-30)
    k_star = torch.multinomial(cluster_weights.float(), 1).item()

    L_star, R_star = clusters_L[k_star], clusters_R[k_star]
    N_star = L_star + R_star

    # Step 2: Sample focus state from importance table
    table_weights, table_Ms, table_as = _build_cluster_table(L_star, R_star, beta)
    assert table_weights is not None, (
        f"Empty importance table for focus cluster (L={L_star}, R={R_star}) "
        "that was sampled with positive weight"
    )
    assert torch.isclose(table_weights.sum(), torch.tensor(clusters_neg_log_p[k_star], dtype=torch.float64), rtol=1e-4), (
        f"Table weights sum {table_weights.sum().item():.6f} != cluster weight "
        f"{clusters_neg_log_p[k_star]:.6f} (Proposition 1 violated)"
    )

    sample_idx = torch.multinomial(table_weights.float(), 1).item()
    M_star = table_Ms[sample_idx]
    a_star = table_as[sample_idx]

    # Step 3: Sample global removal count M | M_k*
    N_total = sum(L + R for L, R in zip(clusters_L, clusters_R))
    cluster_M = [0] * K
    cluster_a = [0] * K
    cluster_M[k_star] = M_star
    cluster_a[k_star] = a_star

    if K == 1:
        M_total = M_star
    else:
        p_val = np.random.beta(float(M_star), float(N_star - M_star + 1))
        l_val = np.random.binomial(N_total - N_star, p_val)
        M_total = M_star + l_val

        # Step 4: Sample remaining clusters
        remaining = M_total - M_star
        remaining_indices = [k for k in range(K) if k != k_star]
        pool = N_total - N_star

        for k in remaining_indices:
            N_k = clusters_L[k] + clusters_R[k]
            if pool == 0 or remaining == 0:
                cluster_M[k] = 0
            else:
                cluster_M[k] = int(np.random.hypergeometric(N_k, pool - N_k, remaining))
                remaining -= cluster_M[k]
            pool -= N_k

        for k in remaining_indices:
            if cluster_M[k] == 0:
                cluster_a[k] = 0
            else:
                cluster_a[k] = int(
                    np.random.hypergeometric(clusters_L[k], clusters_R[k], cluster_M[k])
                )

    counts = _map_cluster_states_to_counts(
        graph, info, leaf_idx, leaf_depths, d,
        cluster_nodes, cluster_M, cluster_a,
    )

    # Importance weight
    sum_MI = _compute_sum_MI(cluster_M, cluster_a, clusters_L, clusters_R, beta)
    assert sum_MI > 0, "sum_MI must be positive — focus cluster was sampled with I > 0"
    importance_weight = torch.tensor(
        M_total * Z_total / sum_MI, device=device, dtype=torch.float,
    )

    t_step = int(counts.sum().item())
    T = int(leaf_depths.sum().item())
    return counts, t_step, T, importance_weight


def _compute_is_weight_for_state(graph, leaf_idx, leaf_depths, counts, beta=-1.5):
    """Evaluate w_IS = M * Z / sum_MI for an arbitrary percolation state."""
    device = graph.device
    M_total = int(counts.sum().item())
    assert M_total > 0, "Cannot compute IS weight for empty percolation state"

    # Frontier depth: not-descended leaves have new_depth = d-1 (the minimum).
    affected = counts > 0
    d = int((leaf_depths[affected] - counts[affected]).min().item()) + 1

    info = _compute_tree_layer_info(graph, leaf_idx, leaf_depths, beta)
    Z_total = info["pi"].sum().item()

    node_depths = graph.ndata["d"]
    cluster_nodes = (
        info["is_internal"] & (node_depths == d - 1)
    ).nonzero(as_tuple=False).flatten()
    K = cluster_nodes.numel()
    assert K > 0, f"No clusters at depth {d - 1} — tree structure inconsistent"

    clusters_L = info["L_node"][cluster_nodes].long().cpu().tolist()
    clusters_R = info["R_node"][cluster_nodes].long().cpu().tolist()

    sides, cluster_ids, active_indices = _leaf_cluster_assignments(
        graph, info, leaf_idx, leaf_depths, d, cluster_nodes,
    )
    not_desc = counts[active_indices] == (leaf_depths[active_indices] - d + 1)
    cluster_M, cluster_a = [0] * K, [0] * K
    for k in range(K):
        ink = cluster_ids == k
        cluster_a[k] = int((ink & (sides == 0) & not_desc).sum().item())
        cluster_M[k] = cluster_a[k] + int(
            (ink & (sides == 1) & not_desc).sum().item()
        )

    s = _compute_sum_MI(cluster_M, cluster_a, clusters_L, clusters_R, beta)
    if s <= 0:
            return torch.tensor(1.0, device=device, dtype=torch.float)
    return torch.tensor(M_total * Z_total / s, device=device, dtype=torch.float)


def sample_X_t(graph, alpha, biased_schedule=True):
    graph.ndata["d"] = compute_depths(graph)
    leaf_idx = torch.where(graph.in_degrees().eq(0))[0]
    leaf_depths = graph.ndata["d"][leaf_idx].long()

    # biased_schedule: bool or float in [0, 1].
    # Float mixes IS (prob=biased_schedule) with uniform (prob=1-biased_schedule),
    # using the balance-heuristic weight: 1 / (π/w_IS + (1-π)/T).
    if isinstance(biased_schedule, bool):
        if biased_schedule:
            counts, t_step, T, importance_weight = sample_percolation_counts_biased(
                graph, leaf_idx, leaf_depths.clone(),
            )
        else:
            counts, t_step, T, importance_weight = sample_percolation_counts_old(
                graph, leaf_idx, leaf_depths.clone(), t=None, alpha=None,
            )
    else:
        pi_mix = float(biased_schedule)
        use_is = torch.rand(1).item() < pi_mix

        if use_is:
            counts, t_step, T, w_is = sample_percolation_counts_biased(
                graph, leaf_idx, leaf_depths.clone(),
            )
        else:
            counts, t_step, T, _ = sample_percolation_counts_old(
                graph, leaf_idx, leaf_depths.clone(), t=None, alpha=None,
            )
            w_is = _compute_is_weight_for_state(
                graph, leaf_idx, leaf_depths, counts,
            )

        # Balance-heuristic mixture weight: 1 / (π/w_IS + (1-π)/T).
        importance_weight = 1.0 / (pi_mix / w_is + (1.0 - pi_mix) / T)
    t_depth, t_layer = compute_top_down_progress(leaf_depths, t_step)
    from .percolation import populate_leafset, get_obs_graph, compute_target
    populate_leafset(graph, leaf_idx, counts)
    obs_graph = get_obs_graph(graph)
    target = compute_target(graph, leaf_idx)
    return obs_graph, t_step, T, t_depth, t_layer, target, importance_weight
