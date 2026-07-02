"""MiddleOut training data structures, tree utilities, and preprocessing.

Split out of the former monolithic ``unified.py``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import dgl
import torch
from ete3 import TreeNode

from ..bottom_up.tree_utils import pairwise_hamming_frac
from ..tree_utils import squaretrick_basis_from_edge_index
from .rooted_features import compute_rooted_features


@dataclass(slots=True)
class MiddleOutStepExample:
    remaining_leaves: list[int]
    missing_leaves: list[int]
    edge_index: torch.LongTensor
    node_basis: torch.Tensor
    spd: torch.LongTensor
    edge_children: torch.LongTensor
    edge_parents: torch.LongTensor
    sparse_target_edge: int
    dense_target_edges: torch.LongTensor
    time: int
    depths: torch.LongTensor | None = None
    subtree_counts: torch.LongTensor | None = None
    node_basis_topo: torch.Tensor | None = None


@dataclass(slots=True)
class MiddleOutTrainStep:
    missing_leaves: torch.LongTensor
    node_basis: torch.Tensor
    edge_parents: torch.LongTensor
    dense_target_edges: torch.LongTensor
    sparse_target_edge: int
    time: int
    spd: torch.LongTensor | None = None
    depths: torch.LongTensor | None = None
    subtree_counts: torch.LongTensor | None = None
    node_basis_topo: torch.Tensor | None = None


@dataclass(slots=True)
class MiddleOutTrainExample:
    seqs: torch.Tensor
    T: int
    steps: list[MiddleOutTrainStep] | None = None
    step: MiddleOutTrainStep | None = None
    is_weight: float = 1.0
    hamming_leaf: torch.Tensor | None = None


def ensure_numeric_node_names(tree: TreeNode) -> TreeNode:
    max_seen = -1
    for node in tree.traverse():
        try:
            node.name = int(node.name)
            max_seen = max(max_seen, int(node.name))
        except (TypeError, ValueError):
            node.name = None
    next_internal = max_seen + 1
    for node in tree.traverse("postorder"):
        if node.name is None:
            node.name = next_internal
            next_internal += 1
    return tree


def normalize_rooted_tree_ids(tree: TreeNode, n_leaves: int) -> TreeNode:
    tree = ensure_numeric_node_names(tree)
    leaf_ids = sorted(int(x.name) for x in tree.get_leaves())
    expected_leaf_ids = list(range(n_leaves))
    if leaf_ids != expected_leaf_ids:
        raise ValueError(
            f"Expected leaf ids {expected_leaf_ids[:10]}..., got {leaf_ids[:10]}..."
        )
    next_internal = n_leaves + 1
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            node.name = int(node.name)
        else:
            node.name = next_internal
            next_internal += 1
    return tree


def augment_rooted_tree_with_token(
    tree: TreeNode, n_leaves: int
) -> tuple[TreeNode, int]:
    tree = normalize_rooted_tree_ids(tree.copy(), n_leaves)
    root_token_id = n_leaves
    tree.get_tree_root().add_child(TreeNode(name=root_token_id))
    return tree, root_token_id


def canonical_unroot(tree: TreeNode, leaf_triplet: list[int] | None = None):
    leaves = sorted(int(x.name) for x in tree.get_leaves())
    if len(leaves) < 3:
        raise ValueError("Tree must have at least 3 leaves")
    if leaf_triplet is None:
        leaf_triplet = leaves[:3]
    a, b, c = leaf_triplet
    leaf_nodes = {
        a: tree.search_nodes(name=a)[0],
        b: tree.search_nodes(name=b)[0],
        c: tree.search_nodes(name=c)[0],
    }
    removed_id = None
    for x, y in combinations([a, b, c], r=2):
        common = tree.get_common_ancestor(leaf_nodes[x], leaf_nodes[y])
        if not common.is_root():
            tree.set_outgroup(common)
            removed_id = int(common.name)
            c1, c2 = common.children
            tree.remove_child(common)
            tree.add_child(c1)
            tree.add_child(c2)
            return tree, removed_id
    root = tree.get_tree_root()
    if len(root.children) == 2:
        child = root.children[0]
        if len(child.children) == 2:
            removed_id = int(child.name)
            c1, c2 = child.children
            tree.remove_child(child)
            tree.add_child(c1)
            tree.add_child(c2)
    return tree, removed_id


def remap_tree_ids(tree: TreeNode):
    tree = ensure_numeric_node_names(tree)
    leaves = sorted(int(x.name) for x in tree.get_leaves())
    internals = [int(x.name) for x in tree.traverse() if not x.is_leaf()]
    root_id = int(tree.get_tree_root().name)
    internals = sorted(x for x in internals if x != root_id) + [root_id]
    leaf_map = {old: new for new, old in enumerate(leaves)}
    internal_map = {old: new + len(leaves) for new, old in enumerate(internals)}
    for node in tree.traverse():
        old = int(node.name)
        node.name = leaf_map[old] if node.is_leaf() else internal_map[old]
    return tree, leaf_map, internal_map


def tree_to_edge_index(tree: TreeNode) -> list[list[int]]:
    nodes = list(tree.traverse())
    n_nodes = len(nodes)
    edge_index = [[-1, -1, -1] for _ in range(n_nodes)]
    for node in tree.traverse():
        node_id = int(node.name)
        if node.is_root():
            children = [int(c.name) for c in node.get_children()]
            if len(children) != 3:
                raise ValueError("Root must have 3 children for unrooted encoding")
            edge_index[node_id] = children
        elif node.is_leaf():
            edge_index[node_id] = [int(node.up.name), -1, -1]
        else:
            children = [int(c.name) for c in node.get_children()]
            edge_index[node_id] = [int(node.up.name), children[0], children[1]]
    return edge_index


def prune_tree(full_tree: TreeNode, keep_leaves: set[int]) -> TreeNode:
    t = full_tree.copy()
    t.prune(
        [t.search_nodes(name=x)[0] for x in keep_leaves], preserve_branch_length=True
    )
    return t


def compute_node_leaf_masks(tree: TreeNode) -> dict[TreeNode, int]:
    masks: dict[TreeNode, int] = {}
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            masks[node] = 1 << int(node.name)
        else:
            mask = 0
            for child in node.get_children():
                mask |= masks[child]
            masks[node] = mask
    return masks


def build_leafmask_lookup(tree: TreeNode) -> dict[int, int]:
    masks = compute_node_leaf_masks(tree)
    lookup: dict[int, int] = {}
    for node in tree.traverse():
        if node.is_root():
            continue
        lookup[masks[node]] = int(node.name)
    return lookup


def resolve_builder_edge_child(
    builder_tree: TreeNode,
    remaining_leaves: list[int],
    target_mask: int,
) -> int:
    lookup = build_leafmask_lookup(builder_tree)
    remaining_mask = sum(1 << leaf_id for leaf_id in remaining_leaves)

    edge_child = lookup.get(target_mask)
    if edge_child is not None:
        return edge_child

    complement_mask = remaining_mask & ~target_mask
    edge_child = lookup.get(complement_mask)
    if edge_child is not None:
        return edge_child

    raise KeyError("Could not map sampled unrooted edge back to a rooted builder edge")


def precompute_leaf_sibling_masks(tree: TreeNode, n_leaves: int) -> list[list[int]]:
    masks = compute_node_leaf_masks(tree)
    leaf_nodes = {int(x.name): x for x in tree.get_leaves()}
    sibling_masks: list[list[int]] = [[] for _ in range(n_leaves)]
    for leaf_id, leaf in leaf_nodes.items():
        cur = leaf
        parent = cur.up
        while parent is not None:
            for sibling in parent.get_children():
                if sibling is cur:
                    continue
                sibling_masks[leaf_id].append(masks[sibling])
            cur = parent
            parent = cur.up
    return sibling_masks


def compute_dense_targets(
    full_tree: TreeNode,
    remaining_leaves: list[int],
    missing_leaves: list[int],
    remaining_mask: int,
    sibling_masks: list[list[int]],
):
    triplet = sorted(remaining_leaves)[:3]
    tree_cur = prune_tree(full_tree, set(remaining_leaves))
    tree_cur, _ = canonical_unroot(tree_cur, triplet)
    tree_cur = ensure_numeric_node_names(tree_cur)
    leafmask_lookup = build_leafmask_lookup(tree_cur)

    targets = []
    for leaf_id in missing_leaves:
        sib_mask = None
        for mask in sibling_masks[leaf_id]:
            if mask & remaining_mask:
                sib_mask = mask & remaining_mask
                break
        if sib_mask is None:
            raise RuntimeError("No sibling mask intersects remaining leaves")

        candidate_upside = remaining_mask & ~sib_mask
        edge_child = leafmask_lookup.get(candidate_upside)
        if edge_child is None:
            edge_child = leafmask_lookup.get(sib_mask)
        if edge_child is None:
            tree_plus = prune_tree(full_tree, set(remaining_leaves + [leaf_id]))
            tree_plus, _ = canonical_unroot(tree_plus, triplet)
            tree_plus = ensure_numeric_node_names(tree_plus)
            leaf_node = tree_plus.search_nodes(name=leaf_id)[0]
            parent = leaf_node.up
            if parent is None:
                raise RuntimeError("Leaf has no parent in tree_plus")
            all_leaves = set(int(x) for x in tree_plus.get_leaf_names())
            parent_leaves = set(int(x) for x in parent.get_leaf_names())
            neighbor_sets = []
            if parent.up is not None:
                neighbor_sets.append(all_leaves - parent_leaves)
            for child in parent.get_children():
                if child is not leaf_node:
                    neighbor_sets.append(set(int(x) for x in child.get_leaf_names()))
            neighbor_sets = [s - {leaf_id} for s in neighbor_sets]
            edge_child = None
            for neighbor in neighbor_sets:
                edge_child = leafmask_lookup.get(sum(1 << x for x in neighbor))
                if edge_child is not None:
                    break
            if edge_child is None:
                raise RuntimeError("Could not map edge by leaf masks")
        targets.append(edge_child)

    tree_cur, leaf_map, internal_map = remap_tree_ids(tree_cur)
    edge_index = torch.as_tensor(tree_to_edge_index(tree_cur), dtype=torch.long)
    remapped_targets = torch.as_tensor(
        [leaf_map[t] if t in leaf_map else internal_map[t] for t in targets],
        dtype=torch.long,
    )
    return tree_cur, edge_index, remapped_targets


def _squaretrick_core(A, n_leaves, nf, tol=1e-5, max_iters=10000):
    bs, n_nodes, _ = A.shape
    device = A.device
    X = torch.full((bs, n_nodes, nf), 1.0 / nf, device=device)
    leaf_idx = torch.arange(nf, device=device)
    X[:, leaf_idx, :] = 0.0
    X[:, leaf_idx, leaf_idx] = 1.0

    for _ in range(max_iters):
        X_old = X
        A = A @ A
        X = A @ X_old
        lnorm = torch.mean(torch.abs(X - X_old), dim=(1, 2))
        if not torch.any(lnorm > tol):
            break

    if nf != n_leaves:
        X = torch.cat(
            [X, torch.zeros((bs, n_nodes, n_leaves - nf), device=device)], dim=-1
        )
    return X


def squaretrick(n_leaves: int, edge_index: torch.LongTensor) -> torch.Tensor:
    bs, n_nodes = edge_index.shape[0], edge_index.shape[1]
    nf = (n_nodes + 2) // 2
    device = edge_index.device
    A = torch.zeros((bs, n_nodes, n_nodes), device=device)
    if n_nodes > nf:
        A[:, nf:].scatter_(dim=-1, index=edge_index[:, nf:], value=1 / 3)
    A[:, :nf, :nf] = torch.eye(nf, device=device).unsqueeze(0).expand(bs, -1, -1)
    return _squaretrick_core(A, n_leaves, nf)


def edge_index_to_spd(edge_index: torch.LongTensor) -> torch.LongTensor:
    # Build an undirected DGL graph from the [N, fanout] edge_index (entries ≥ 0
    # are neighbor ids; -1 slots are unused) and use DGL's C++ shortest_dist.
    n, fanout = edge_index.shape
    src = torch.arange(n, device=edge_index.device).unsqueeze(1).expand(-1, fanout).reshape(-1)
    dst = edge_index.reshape(-1)
    mask = dst >= 0
    s, d = src[mask], dst[mask]
    g = dgl.to_simple(dgl.graph((torch.cat([s, d]), torch.cat([d, s])), num_nodes=n))
    return dgl.shortest_dist(g)


def build_global_node_basis(
    edge_index: torch.LongTensor, remaining_leaves: list[int], total_leaves: int
) -> torch.Tensor:
    local_basis = squaretrick(len(remaining_leaves), edge_index.unsqueeze(0)).squeeze(0)
    global_basis = torch.zeros(
        edge_index.shape[0], total_leaves, dtype=local_basis.dtype
    )
    global_basis[:, remaining_leaves] = local_basis[:, : len(remaining_leaves)]
    return global_basis


def _build_train_step(
    tree: TreeNode,
    *,
    n_leaves: int,
    sibling_masks: list[list[int]],
    full_mask: int,
    leaf_removal_order: list[int],
    step_idx: int,
    attn_head: bool,
) -> MiddleOutTrainStep:
    missing = leaf_removal_order[:step_idx]
    missing_mask = 0
    for leaf_id in missing:
        missing_mask |= 1 << leaf_id
    remaining = sorted(set(range(n_leaves)) - set(missing))
    remaining_mask = full_mask ^ missing_mask

    _, edge_index, dense_targets = compute_dense_targets(
        tree,
        remaining,
        missing,
        remaining_mask,
        sibling_masks,
    )
    # Root token is always in remaining; find its remapped leaf ID
    root_token_global = n_leaves - 1
    root_token_local = remaining.index(root_token_global)

    depths, subtree_counts, local_basis = compute_rooted_features(
        edge_index, root_token_local, len(remaining)
    )
    # Map local basis [n_nodes, n_remaining] to global [n_nodes, n_total_leaves]
    global_basis = torch.zeros(edge_index.shape[0], n_leaves, dtype=local_basis.dtype)
    global_basis[:, remaining] = local_basis

    # Squaretrick absorption basis (unrooted topological coord), same shape as rooted.
    # Cheap CPU compute (runs in dataloader workers); no grad needed.
    global_topo = None
    if attn_head:
        local_topo = squaretrick_basis_from_edge_index(edge_index, len(remaining))
        global_topo = torch.zeros(edge_index.shape[0], n_leaves, dtype=local_topo.dtype)
        global_topo[:, remaining] = local_topo

    return MiddleOutTrainStep(
        missing_leaves=torch.as_tensor(missing, dtype=torch.long),
        node_basis=global_basis,
        edge_parents=edge_index[:-1, 0].clone(),
        dense_target_edges=dense_targets,
        sparse_target_edge=dense_targets[-1].item(),
        time=len(remaining) + 1,
        spd=edge_index_to_spd(edge_index) if attn_head else None,
        depths=depths,
        subtree_counts=subtree_counts,
        node_basis_topo=global_topo,
    )


def preprocess_train_example(
    tree: TreeNode,
    seqs: torch.Tensor,
    *,
    rb: bool,
    attn_head: bool,
    full_traj: bool,
    importance_sample: bool | float = False,
    n_is_weight: float = 1.0,
    leaf_removal_order: list[int] | None = None,
    generator: torch.Generator | None = None,
) -> MiddleOutTrainExample:
    rooted_tree = normalize_rooted_tree_ids(tree.copy(), seqs.shape[0])
    tree, _ = augment_rooted_tree_with_token(rooted_tree, seqs.shape[0])
    n_leaves = seqs.shape[0] + 1
    sibling_masks = precompute_leaf_sibling_masks(tree, n_leaves)
    full_mask = (1 << n_leaves) - 1

    root_token_id = n_leaves - 1  # last leaf id = seqs.shape[0]
    if leaf_removal_order is None:
        # Root token must stay in the seed triplet (last 3 of removal order).
        # Permute only the real leaves, then append root_token_id at the end.
        real_perm = torch.randperm(n_leaves - 1, generator=generator).tolist()
        leaf_removal_order = real_perm + [root_token_id]
    elif sorted(leaf_removal_order) != list(range(n_leaves)):
        raise ValueError("leaf_removal_order must be a permutation of all token ids")
    elif root_token_id not in leaf_removal_order[-3:]:
        raise ValueError("root token must be in the seed triplet (last 3 of removal order)")

    T = n_leaves - 3
    if full_traj:
        steps = [
            _build_train_step(
                tree,
                n_leaves=n_leaves,
                sibling_masks=sibling_masks,
                full_mask=full_mask,
                leaf_removal_order=leaf_removal_order,
                step_idx=step_idx,
                attn_head=attn_head,
            )
            for step_idx in range(1, T + 1)
        ]
        hamming_leaf = pairwise_hamming_frac(seqs.long())
        return MiddleOutTrainExample(seqs=seqs, T=T, steps=steps, is_weight=n_is_weight, hamming_leaf=hamming_leaf)

    # importance_sample: bool or float in [0, 1].
    # Float mixes IS (prob=importance_sample) with uniform (prob=1-importance_sample),
    # using the balance-heuristic weight: 1 / (π * probs[m] + (1-π) / T).
    if isinstance(importance_sample, bool) and not importance_sample:
        step_idx = torch.randint(1, T + 1, (1,), generator=generator).item()
        is_weight = float(T)
    else:
        # IS weights: log edge count per step.
        weights = torch.tensor(
            [math.log(2 * (n_leaves - i) - 3) for i in range(1, T + 1)]
        )
        probs = weights / weights.sum()
        if isinstance(importance_sample, bool):
            # Pure IS.
            step_idx = torch.multinomial(probs, 1, generator=generator).item() + 1
            is_weight = 1.0 / probs[step_idx - 1].item()
        else:
            pi_mix = float(importance_sample)
            if torch.rand(1).item() < pi_mix:
                step_idx = torch.multinomial(probs, 1, generator=generator).item() + 1
            else:
                step_idx = torch.randint(1, T + 1, (1,), generator=generator).item()
            is_weight = 1.0 / (pi_mix * probs[step_idx - 1].item() + (1.0 - pi_mix) / T)
    step = _build_train_step(
        tree,
        n_leaves=n_leaves,
        sibling_masks=sibling_masks,
        full_mask=full_mask,
        leaf_removal_order=leaf_removal_order,
        step_idx=step_idx,
        attn_head=attn_head,
    )
    hamming_leaf = pairwise_hamming_frac(seqs.long())
    return MiddleOutTrainExample(seqs=seqs, T=T, step=step, is_weight=is_weight * n_is_weight, hamming_leaf=hamming_leaf)


def train_map_fn(
    sample,
    *,
    rb: bool = True,
    attn_head: bool = True,
    full_traj: bool = False,
    importance_sample: bool | float = False,
    idx: int | None = None,
):
    tree, seqs, n_is_weight = sample
    generator = None
    if idx is not None:
        generator = torch.Generator()
        generator.manual_seed(int(idx))
    return preprocess_train_example(
        tree,
        seqs,
        rb=rb,
        attn_head=attn_head,
        full_traj=full_traj,
        importance_sample=importance_sample,
        n_is_weight=n_is_weight,
        generator=generator,
    )


def train_collate_fn(batch):
    return batch
