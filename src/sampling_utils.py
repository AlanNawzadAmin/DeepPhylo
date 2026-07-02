import math
from functools import lru_cache

import torch


@lru_cache(maxsize=2000)
def _log_double_factorial(n: int) -> float:
    """Compute ln((2N-3)!!) = sum_{k=4}^{N} ln(2k-5). Returns 0 for N <= 3."""
    if n <= 3:
        return 0.0
    return sum(math.log(2 * k - 5) for k in range(4, n + 1))


def _log_double_factorial_weights(n_min: int, n_max: int) -> torch.Tensor:
    """Compute ln((2N-3)!!) for each N in [n_min, n_max] as a weight vector."""
    return torch.tensor([_log_double_factorial(n) for n in range(n_min, n_max + 1)])


def sample_n_leaves_weighted(
    n_min: int,
    n_max: int,
    generator: torch.Generator | None = None,
) -> tuple[int, float]:
    """Sample n_leaves ~ ln((2N-3)!!) and return (n_leaves, is_weight).

    The IS weight corrects from the biased distribution back to uniform,
    so that E_weighted[is_weight * f(n)] = E_uniform[f(n)].
    Falls back to uniform if n_min == n_max or all weights are zero.
    """
    K = n_max - n_min + 1
    if K <= 1:
        return n_min, 1.0
    weights = _log_double_factorial_weights(n_min, n_max)
    total = weights.sum().item()
    if total == 0:
        n_leaves = torch.randint(n_min, n_max + 1, (1,), generator=generator).item()
        return n_leaves, 1.0
    probs = weights / total
    idx = torch.multinomial(probs, 1, generator=generator).item()
    n_leaves = n_min + idx
    # p_uniform / p_weighted = (1/K) / probs[idx]
    is_weight = 1.0 / (K * probs[idx].item())
    return n_leaves, is_weight


def ensure_newick_semicolon(newick: str) -> str:
    return newick if newick.endswith(";") else f"{newick};"


def tuple_tree_to_newick(tree, names=None) -> str:
    if isinstance(tree, int):
        return names[tree] if names else str(tree)
    return f"({tuple_tree_to_newick(tree[0], names)},{tuple_tree_to_newick(tree[1], names)})"


def build_tree_structure(positions, children):
    def build(node):
        if node not in children:
            leaves = [i for i, p in enumerate(positions) if p.item() == node]
            return leaves[0]
        return (build(children[node][0]), build(children[node][1]))

    return build(0)


def child_lookup_to_tree(positions, left_child, right_child):
    children = {}
    for i, (l, r) in enumerate(zip(left_child, right_child)):
        l = l.item()
        r = r.item()
        if l == -1:
            assert r == -1
        else:
            assert r != -1
            children[i] = (l, r)
    return build_tree_structure(positions, children)
