import math
from dataclasses import dataclass
from typing import List

import dgl
import torch
from ete3 import Tree

from ..top_down.graph_ops import compute_depths
from ..spd_attn import build_basis_from_graph, compute_spd_nodes
from .graph_utils import ordered_exposed_nodes
from .tree_levels import precompute_tree_levels as _precompute_tree_levels
from .tree_utils import (
    _estimate_bl_forced_nj,
    _leaf_counts_from_n,
    calculate_rb_target_probs,
    calculate_subtree_sizes,
    ete3_to_dgl_preserve_ids,
    pairwise_hamming_frac,
    run_forward_noising_process,
)


@dataclass
class TrajectoryStep:
    t_index: int
    t: float
    target_probs: torch.Tensor
    rb_entropy: float = 0.0
    candidate_pairs_global: torch.Tensor | None = None
    obs_graph: dgl.DGLGraph | None = None
    obs_node_ids: torch.Tensor | None = None
    exposed_nodes_local: torch.Tensor | None = None
    spd: torch.Tensor | None = None
    exposed_leaf_counts: torch.Tensor | None = None
    basis: torch.Tensor | None = None
    leaf_idx: torch.Tensor | None = None
    depths: torch.Tensor | None = None
    counts: torch.Tensor | None = None


@dataclass
class TrajectoryExample:
    seqs: torch.Tensor
    T: int
    leaf_features_inputs: torch.Tensor
    full_graph: dgl.DGLGraph | None
    steps: List[TrajectoryStep]
    preselected_steps: bool = False
    n_is_weight: float = 1.0
    tree_levels: List | None = None  # precomputed (node_ids, left_ids, right_ids) per level
    hamming_leaf: torch.Tensor | None = None


def _pda_pair_probs(n_data, pairs, gamma=1.0):
    """Normalized Aldous beta-splitting pair probabilities for candidate pairs."""
    pair_t = torch.tensor(pairs, dtype=torch.long)
    lc = (n_data[pair_t].float() + 1) / 2
    weights = gamma * (lc[:, 0] + lc[:, 1] - 2) + 1
    return weights / weights.sum()


def _compute_is_weight_at_step(
    graph, candidate_pairs_all, target_pair_probs_rb, step_idx, removed_order, gamma=1.0
):
    """Compute IS weight z_pi / KL(p || q_PDA) at a single step."""
    n_data = graph.ndata["n"].squeeze(-1)
    N = len(removed_order) + 1
    z_pi = sum(math.log(2 * k - 1) for k in range(1, N))
    p = target_pair_probs_rb[step_idx]
    q = _pda_pair_probs(n_data, candidate_pairs_all[step_idx], gamma=gamma)
    mask = p > 0
    log_p = torch.log(p[mask])
    log_q = torch.log(q[mask])
    kl = (p[mask] * (log_p - log_q)).sum().item()
    h_cross = -(p[mask] * log_q).sum().item()
    if kl <= 0:
        # p == q at this step; IS weight is infinite (step needs no correction).
        # Callers using balance-heuristic will degenerate to uniform weight.
        return float("inf"), 0.0, h_cross
    return z_pi / kl, kl, h_cross


def _importance_sample_step(
    graph, candidate_pairs_all, target_pair_probs_rb, removed_order, gamma=1.0
):
    """Rejection sampling from pi (Algorithm 1).

    Returns (step_index, is_weight) where is_weight = Z_pi / KL(p || q_PDA | E),
    or None if rejected.
    """
    n_data = graph.ndata["n"].squeeze(-1)
    M = len(removed_order)

    # q_PDA(sigma_m | E_m) for each step
    q_pda_per_step = torch.zeros(M)
    for m, root in enumerate(removed_order):
        children = sorted(graph.predecessors(root).tolist())
        q_all = _pda_pair_probs(n_data, candidate_pairs_all[m], gamma=gamma)
        pair_idx = candidate_pairs_all[m].index(tuple(children))
        q_pda_per_step[m] = q_all[pair_idx]

    # Sample m ~ Cat(c_m^+) where c_m^+ = -log q_PDA(sigma_m | E_m)
    c_plus = -torch.log(q_pda_per_step)
    m = torch.multinomial(c_plus, 1).item()

    w_is, kl, h_cross = _compute_is_weight_at_step(
        graph, candidate_pairs_all, target_pair_probs_rb, m, removed_order, gamma,
    )
    rho = kl / h_cross
    if torch.rand(1).item() < rho:
        return m, w_is
    return None


def build_trajectory_example(
    ete_tree: Tree,
    seqs: torch.Tensor,
    *,
    full_traj: bool = True,
    attn_head: bool = True,
    rb: bool = True,
    importance_sample: bool | float = False,
    is_gamma: float = 1.0,
    n_is_weight: float = 1.0,
    canonicalize_bl: bool = False,
    canonicalize_hamming: bool = False,
) -> TrajectoryExample:
    canonicalize = canonicalize_bl or canonicalize_hamming
    assert not (canonicalize_bl and canonicalize_hamming), (
        "canonicalize_bl and canonicalize_hamming are mutually exclusive"
    )
    assert not (importance_sample and full_traj), (
        "importance_sample and full_traj are mutually exclusive"
    )
    assert not (importance_sample and canonicalize), (
        "importance_sample and canonicalization are mutually exclusive"
    )
    if canonicalize:
        assert not rb, "canonicalization requires rb=False"

    graph = ete3_to_dgl_preserve_ids(ete_tree, seqs)
    calculate_subtree_sizes(graph)

    if canonicalize_hamming:
        # Overwrite graph edge BLs with forced-NJ estimates from sequences
        edge_bls = _estimate_bl_forced_nj(ete_tree, seqs)
        for i, (u, v) in enumerate(zip(*graph.edges())):
            key = (u.item(), v.item())
            if key in edge_bls:
                graph.edata["bl"][i] = edge_bls[key]

    # importance_sample: bool or float in [0, 1].
    # Float mixes IS (prob=importance_sample) with uniform (prob=1-importance_sample).
    if isinstance(importance_sample, bool):
        need_is = importance_sample
        pi_mix = None
    else:
        pi_mix = float(importance_sample)
        need_is = torch.rand(1).item() < pi_mix

    if need_is:
        while True:
            cpa, graph, ro = run_forward_noising_process(graph)
            tprb = calculate_rb_target_probs(graph, cpa)
            result = _importance_sample_step(
                graph, cpa, tprb, ro, gamma=is_gamma,
            )
            if result is not None:
                selected_step, w_is = result
                break
        candidate_pairs_all, removed_order = cpa, ro
        target_pair_probs_rb = tprb
    else:
        candidate_pairs_all, graph, removed_order = run_forward_noising_process(
            graph, canonicalize_bl=canonicalize
        )
        target_pair_probs_rb = calculate_rb_target_probs(graph, candidate_pairs_all)

    n_leaves = int(graph.in_degrees().eq(0).sum().item())
    T = n_leaves - 1
    need_full_graph = full_traj or not attn_head
    if pi_mix is not None:
        # Mixture mode: balance-heuristic weight 1 / (π/w_IS + (1-π)/T).
        if not need_is:
            selected_step = torch.randint(T, (1,)).item()
            w_is, _, _ = _compute_is_weight_at_step(
                graph, candidate_pairs_all, target_pair_probs_rb,
                selected_step, removed_order, gamma=is_gamma,
            )
        selected_step_indices = [selected_step]
        n_is_weight = n_is_weight / (pi_mix / w_is + (1.0 - pi_mix) / T)
    elif need_is:
        selected_step_indices = [selected_step]
        n_is_weight = n_is_weight * w_is
    elif full_traj:
        selected_step_indices = range(T)
    else:
        selected_step_indices = [torch.randint(T, (1,)).item()]
        n_is_weight = n_is_weight * T

    steps = []
    for step_idx in selected_step_indices:
        t_index = step_idx + 1
        candidate_pairs_global = candidate_pairs_all[step_idx]
        target_probs_rb = target_pair_probs_rb[step_idx].float().cpu()
        if not rb:
            # Target = one-hot at children of the just-removed node in this
            # specific noising order. With full_traj this gives loss =
            # −log q(this tree construction).
            target_probs = torch.zeros_like(target_probs_rb)
            if target_probs_rb.numel() > 0:
                preds = graph.predecessors(removed_order[step_idx]).tolist()
                target_pair = (min(preds), max(preds))
                target_idx = candidate_pairs_global.index(target_pair)
                target_probs[target_idx] = 1.0

            if canonicalize or target_probs_rb.numel() == 0:
                # Canonical order is deterministic; loss is already −log p(tree|x).
                rb_entropy = 0.0
            else:
                # Random ordering: subtract H(target_probs_rb) per step so the
                # full loss is an unbiased MC estimator of KL(p_RB || q).
                p = target_probs_rb.clamp(min=1e-30)
                rb_entropy = float(-(target_probs_rb * torch.log(p)).sum())
        else:
            # rb=True: target is the full marginal RB distribution. kl_div
            # already returns KL(target_probs_rb || q), which includes the
            # −H(target_probs_rb) term internally. No external correction.
            target_probs = target_probs_rb
            rb_entropy = 0.0

        candidate_pairs_tensor = None
        obs_node_ids = None
        exposed_nodes_local = None
        spd = None
        exposed_leaf_counts = None
        step_basis = None
        step_leaf_idx = None
        step_depths = None
        step_counts = None
        if attn_head:
            available_nodes = ~graph.ndata["removed_t"][:, :t_index].any(dim=1)
            obs_graph = dgl.node_subgraph(graph, available_nodes)
            obs_node_ids = obs_graph.ndata[dgl.NID].clone().cpu()
            exposed_nodes_local = ordered_exposed_nodes(obs_graph).clone().cpu()
            spd = compute_spd_nodes(obs_graph).cpu()
            global_exposed_ids = obs_node_ids[exposed_nodes_local]
            exposed_leaf_counts = _leaf_counts_from_n(
                graph.ndata["n"][global_exposed_ids].squeeze(-1)
            )
            # Precompute basis, leaf_idx, depths, counts on CPU. Basis is the
            # 2-channel stack [N_obs, n_leaves, 2]: rooted subtree + squaretrick.
            from ..tree_utils import squaretrick_basis_from_dgl
            step_leaf_idx = torch.where(obs_graph.in_degrees().eq(0))[0].cpu()
            rooted = build_basis_from_graph(obs_graph, step_leaf_idx).cpu()
            topo = squaretrick_basis_from_dgl(obs_graph, step_leaf_idx).cpu()
            step_basis = torch.stack([rooted, topo], dim=-1)
            step_depths = compute_depths(obs_graph).float().cpu()
            step_counts = obs_graph.ndata["n"].squeeze(-1).float().cpu()
        else:
            if candidate_pairs_global:
                candidate_pairs_tensor = torch.tensor(
                    candidate_pairs_global, dtype=torch.long
                )
                exposed_nodes_sorted = sorted(
                    {n for pair in candidate_pairs_global for n in pair}
                )
                exposed_ids = torch.tensor(exposed_nodes_sorted, dtype=torch.long)
                exposed_leaf_counts = _leaf_counts_from_n(
                    graph.ndata["n"][exposed_ids].squeeze(-1)
                )
            else:
                candidate_pairs_tensor = torch.empty((0, 2), dtype=torch.long)

        steps.append(
            TrajectoryStep(
                t_index=t_index,
                t=t_index / T,
                target_probs=target_probs,
                rb_entropy=rb_entropy,
                candidate_pairs_global=candidate_pairs_tensor,
                obs_graph=obs_graph if attn_head and not need_full_graph else None,
                obs_node_ids=obs_node_ids,
                exposed_nodes_local=exposed_nodes_local,
                spd=spd,
                exposed_leaf_counts=exposed_leaf_counts,
                basis=step_basis,
                leaf_idx=step_leaf_idx,
                depths=step_depths,
                counts=step_counts,
            )
        )

    # Strip ndata that was only needed for trajectory construction,
    # not for GPU-side scoring (build_all_embeddings only uses "n").
    for key in ["x", "removed_t", "internal_n"]:
        if key in graph.ndata:
            del graph.ndata[key]

    # Precompute tree levels for batched embedding propagation on GPU.
    tree_levels = _precompute_tree_levels(graph) if need_full_graph else None

    hamming_leaf = pairwise_hamming_frac(seqs.long()) if attn_head else None

    return TrajectoryExample(
        seqs=seqs,
        T=T,
        leaf_features_inputs=seqs,
        full_graph=graph if need_full_graph else None,
        steps=steps,
        preselected_steps=not full_traj,
        n_is_weight=n_is_weight,
        tree_levels=tree_levels,
        hamming_leaf=hamming_leaf,
    )


def map_fn_build_trajectory_example(
    sample, full_traj, attn_head, rb, importance_sample=False, is_gamma=1.0,
    canonicalize_bl=False, canonicalize_hamming=False,
):
    ete_tree, seqs, n_is_weight = sample
    return build_trajectory_example(
        ete_tree,
        seqs.cpu(),
        full_traj=full_traj,
        attn_head=attn_head,
        rb=rb,
        importance_sample=importance_sample,
        is_gamma=is_gamma,
        n_is_weight=n_is_weight,
        canonicalize_bl=canonicalize_bl,
        canonicalize_hamming=canonicalize_hamming,
    )


def collate_fn_trajectory_examples(batch):
    return list(batch)
