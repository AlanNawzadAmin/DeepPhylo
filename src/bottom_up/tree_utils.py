import heapq
import math
from itertools import combinations

import dgl
import torch
from ete3 import Tree


def _aldous_log_prior(leaf_counts: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Log-prior for all upper-triangular pairs under the Aldous beta-splitting model.

    For two subtrees with leaf counts a and b, the merge rate is proportional to
    gamma * (a + b - 2) + 1.  gamma=0 gives Yule, gamma=1 gives PDA (uniform
    random tree), and gamma -> inf gives the comb (caterpillar) tree.
    """
    n = leaf_counts.shape[0]
    if n < 2:
        return torch.empty(0, device=leaf_counts.device)
    i, j = torch.triu_indices(n, n, 1, device=leaf_counts.device)
    a = leaf_counts[i].float()
    b = leaf_counts[j].float()
    weights = gamma * (a + b - 2) + 1
    return torch.log(weights.clamp(min=1e-30))


def _leaf_counts_from_n(n_data: torch.Tensor) -> torch.Tensor:
    """Convert total-node counts (2L-1 for L leaves) to leaf counts."""
    return ((n_data.float() + 1) / 2).long()


def compute_step_kl(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    # Compute the loss in fp32: under bf16-true, logits/targets arrive as bf16,
    # but log_softmax on small probabilities can underflow in bf16's narrow
    # range. Casting up before the reduction is a few extra elementwise copies
    # but eliminates a real instability mode.
    model_log_probs = torch.log_softmax(logits.float(), dim=0)
    return torch.nn.functional.kl_div(model_log_probs, target_probs.float(), reduction="sum")


def compute_log_num_linear_extensions(ete_tree: Tree) -> float:
    for node in ete_tree.traverse("postorder"):
        if node.is_leaf():
            node._subtree_internal_size = 0
        else:
            node._subtree_internal_size = 1 + sum(
                child._subtree_internal_size for child in node.children
            )

    n_internal = ete_tree._subtree_internal_size
    if n_internal == 0:
        return 0.0

    log_num_ext = math.lgamma(n_internal + 1)
    for node in ete_tree.traverse("postorder"):
        if not node.is_leaf():
            log_num_ext -= math.log(node._subtree_internal_size)
    return log_num_ext


def ete3_to_dgl_preserve_ids(ete_tree: Tree, seqs=None) -> dgl.DGLGraph:
    """
    Build a DGL tree with node IDs taken directly from ete3 node.name.
    Edges are directed child -> parent.
    Branch lengths (ete3 node.dist) are stored as edge data "bl".
    """
    leaves = ete_tree.get_leaves()
    n_leaves = len(leaves)
    n_nodes = 2 * n_leaves - 1

    edges = []
    branch_lengths = []
    for node in ete_tree.traverse("postorder"):
        parent_id = int(node.name)
        for child in node.children:
            child_id = int(child.name)
            edges.append((child_id, parent_id))
            branch_lengths.append(child.dist if child.dist is not None else 0.0)

    g = dgl.graph(edges, num_nodes=n_nodes)
    g.edata["bl"] = torch.tensor(branch_lengths, dtype=torch.float32)
    if seqs is None:
        g.ndata["x"] = torch.zeros((n_nodes, 1), dtype=torch.long)
    else:
        g.ndata["x"] = torch.zeros((n_nodes, seqs.shape[1]), dtype=torch.long)
        g.ndata["x"][:n_leaves] = seqs
    return g


def calculate_subtree_sizes(graph: dgl.DGLGraph) -> None:
    graph.ndata["n"] = torch.ones(graph.num_nodes(), 1, device=graph.device)
    graph.ndata["internal_n"] = torch.zeros(graph.num_nodes(), 1, device=graph.device)
    dgl.prop_nodes_topo(
        graph,
        message_func=lambda e: {
            "n_m": e.src["n"],
            "internal_n_m": e.src["internal_n"],
        },
        reduce_func=lambda n: {
            "n": torch.sum(n.mailbox["n_m"], dim=1) + 1,
            "internal_n": torch.sum(n.mailbox["internal_n_m"], dim=1) + 1,
        },
    )


_MAX_VOCAB_SIZE = 22  # aa alphabet size (dna=6); unused values are no-ops


def _pairwise_match_counts(seqs: torch.Tensor):
    """Count pairwise matches from integer-coded sequences.

    Args:
        seqs: (n, s) integer tensor.
    Returns:
        matches: (n, n) tensor, s: sequence length.
    """
    n, s = seqs.shape
    matches = torch.zeros(n, n, device=seqs.device)
    for c in range(_MAX_VOCAB_SIZE):
        indicator = (seqs == c).float()  # (n, s)
        matches += indicator @ indicator.T  # (n, n)
    return matches, s


def _pairwise_jc_distances(seqs: torch.Tensor) -> torch.Tensor:
    """Pairwise JC-corrected distances from integer-coded sequences.

    Args:
        seqs: (n, s) integer tensor.
    Returns:
        (n, n) distance tensor.
    """
    matches, s = _pairwise_match_counts(seqs)
    k_mat = s - matches
    p_hat = ((k_mat + 1) / (s + 2)).clamp(max=0.749)
    return -0.75 * torch.log(1 - 4 / 3 * p_hat)


def pairwise_hamming_frac(seqs: torch.Tensor) -> torch.Tensor:
    """Pairwise similarity (match fraction) from integer-coded sequences.

    Args:
        seqs: (n, s) integer tensor.
    Returns:
        (n, n) tensor with values in [0, 1]: 1 = identical, 0 = no matches.
    """
    matches, s = _pairwise_match_counts(seqs)
    return matches / s


def _estimate_bl_forced_nj(ete_tree: Tree, seqs: torch.Tensor) -> dict:
    """Estimate per-edge branch lengths via forced NJ on JC-corrected Hamming distances.

    Returns a dict mapping (child_id, parent_id) -> estimated branch length.
    """
    leaves = ete_tree.get_leaves()
    n_leaves = len(leaves)
    leaf_ids = [int(l.name) for l in leaves]

    jc_dist = _pairwise_jc_distances(seqs[leaf_ids])

    n_total = 2 * n_leaves - 1
    id_to_idx = torch.full((n_total,), -1, dtype=torch.long)
    for idx, lid in enumerate(leaf_ids):
        id_to_idx[lid] = idx

    # Padded distance matrix; grows as parent nodes are added
    D = torch.zeros(n_total, n_total)
    D[:n_leaves, :n_leaves] = jc_dist

    active_mask = torch.zeros(n_total, dtype=torch.bool)
    active_mask[:n_leaves] = True
    next_idx = n_leaves

    edge_bls = {}

    for node in ete_tree.traverse("postorder"):
        if node.is_leaf():
            continue
        nid = int(node.name)
        children = [int(c.name) for c in node.children]
        ci_idx = id_to_idx[children[0]].item()
        cj_idx = id_to_idx[children[1]].item()

        n_active = active_mask.sum().item()
        d_ij = D[ci_idx, cj_idx].item()

        if n_active > 2:
            sum_ci = D[ci_idx][active_mask].sum().item() - d_ij
            sum_cj = D[cj_idx][active_mask].sum().item() - d_ij
            bl_ci = d_ij / 2 + (sum_ci - sum_cj) / (2 * (n_active - 2))
            bl_cj = d_ij - bl_ci
        else:
            bl_ci = d_ij / 2
            bl_cj = d_ij / 2

        edge_bls[(children[0], nid)] = max(bl_ci, 0.0)
        edge_bls[(children[1], nid)] = max(bl_cj, 0.0)

        parent_idx = next_idx
        id_to_idx[nid] = parent_idx
        next_idx += 1

        # d(parent, k) = (d(ci,k) + d(cj,k) - d_ij) / 2
        D[parent_idx] = (D[ci_idx] + D[cj_idx] - d_ij) / 2
        D[:, parent_idx] = D[parent_idx]
        D[parent_idx, parent_idx] = 0.0

        active_mask[ci_idx] = False
        active_mask[cj_idx] = False
        active_mask[parent_idx] = True

    return edge_bls


def _canonical_merge_order(graph: dgl.DGLGraph) -> list:
    """Phyloformer 2's canonical merge order (arXiv:2510.12976, Sec. 3.2): at each
    step merge the two closest available nodes, i.e. take the mergeable internal
    node with the smallest cherry length. Returned bottom-up (children first)."""
    bl_sums = torch.zeros(graph.num_nodes())
    _, dst = graph.edges()
    bl_sums.index_add_(0, dst.cpu(), graph.edata["bl"].cpu().float())

    internal = graph.in_degrees().eq(2).nonzero().squeeze(-1).tolist()
    children = {v: [int(c) for c in graph.predecessors(v).tolist()] for v in internal}
    internal_set = set(internal)
    parent = {c: v for v, cs in children.items() for c in cs}
    # A node is mergeable once its internal children have merged; leaves are ready.
    pending = {v: sum(1 for c in children[v] if c in internal_set) for v in internal}

    heap = [(float(bl_sums[v]), v) for v in internal if pending[v] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, v = heapq.heappop(heap)
        order.append(v)
        p = parent.get(v)
        if p is not None:
            pending[p] -= 1
            if pending[p] == 0:
                heapq.heappush(heap, (float(bl_sums[p]), p))
    return order


def run_forward_noising_process(
    graph: dgl.DGLGraph, num_steps=None, canonicalize_bl: bool = False
):
    n_leaves = int(graph.in_degrees().eq(0).sum().item())
    num_nodes = graph.num_nodes()
    M = num_steps if num_steps is not None else n_leaves - 1

    removed_t = torch.zeros(num_nodes, M, device=graph.device, dtype=torch.int64)
    roots = graph.out_degrees().eq(0).nonzero().squeeze(-1).tolist()

    exposed_nodes = roots[:]
    exposed_roots = [r for r in roots if graph.in_degrees(r) == 2]

    # This loop walks the REMOVAL sequence, which is the reverse of the merge
    # sequence, so the greedy criterion cannot be applied here directly: taking
    # the smallest cherry first on removal merges it last.
    removal_plan = (
        list(reversed(_canonical_merge_order(graph))) if canonicalize_bl else None
    )

    candidate_pairs = []
    removed_order = []
    for m in range(M):
        if canonicalize_bl:
            root = removal_plan[m]
        else:
            weights = graph.ndata["internal_n"][exposed_roots].squeeze(-1)
            picked_idx = torch.multinomial(weights, 1).item()
            root = exposed_roots[picked_idx]

        predecessors = graph.predecessors(root)

        exposed_nodes.remove(root)
        exposed_nodes.extend(predecessors.tolist())

        exposed_roots.remove(root)
        exposed_roots.extend(
            [v.item() for v in predecessors if graph.in_degrees(v).item() == 2]
        )

        candidate_pairs.append(list(combinations(sorted(exposed_nodes), 2)))
        removed_t[root, m] = 1
        removed_order.append(root)

    graph.ndata["removed_t"] = removed_t
    return candidate_pairs, graph, removed_order


def calculate_rb_target_probs(graph: dgl.DGLGraph, candidate_pairs):
    M = len(candidate_pairs)
    removed_indicator = graph.ndata["removed_t"].clone()

    dgl.prop_nodes_topo(
        graph,
        message_func=lambda e: {"m": e.src["removed_t"]},
        reduce_func=lambda n: {
            "removed_t": torch.sum(n.mailbox["m"], dim=1) + n.data["removed_t"]
        },
    )
    removed_n = graph.ndata["removed_t"].cumsum(dim=1)

    w = torch.where(removed_n == 1, 1, removed_n / (removed_n - 1))
    rev_graph = dgl.reverse(graph)
    rev_graph.ndata["c"] = w.clone()
    dgl.prop_nodes_topo(
        rev_graph,
        message_func=lambda edges: {"m": edges.src["c"]},
        reduce_func=lambda nodes: {
            "c": torch.prod(nodes.mailbox["m"], dim=1) * nodes.data["c"]
        },
    )

    valid_p, valid_m = torch.where(removed_n == 1)
    valid_pairs = torch.stack([graph.predecessors(int(v)) for v in valid_p])
    target_probs = rev_graph.ndata["c"][valid_p, valid_m] / (valid_m + 1)

    target_probs_lookup = [dict() for _ in range(M)]
    for pair, prob, m in zip(valid_pairs, target_probs, valid_m):
        key = tuple(sorted(pair.tolist()))
        target_probs_lookup[m.item()][key] = prob.item()

    target_pair_probs = []
    for m, pairs in enumerate(candidate_pairs):
        weights = torch.tensor(
            [target_probs_lookup[m].get(tuple(pair), 0.0) for pair in pairs],
            device=graph.device,
        )
        if weights.numel() > 0:
            z = weights.sum()
            if z > 0:
                weights = weights / z
        target_pair_probs.append(weights)

    graph.ndata["removed_t"] = removed_indicator
    return target_pair_probs
