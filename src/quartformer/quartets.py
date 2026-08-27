"""Quartet enumeration, true-tree labeling, and block sampling for QuartFormer.

Conventions (matching the released QuartFormer code, utils._extract_quartet /
_canonical_split_from_class): a quartet is a tuple of taxa in *sorted* order
(a, b, c, d), and the 3 unrooted resolutions are indexed

    class 0 -> (a,b) | (c,d)
    class 1 -> (a,c) | (b,d)
    class 2 -> (a,d) | (b,c)

The true label for a quartet is the resolution whose two within-pair path
lengths sum to the minimum (the four-point condition). We use *topological*
(unit-weight) path lengths, which recover the true topology exactly for any
resolved binary tree regardless of branch-length/rate heterogeneity.
"""

from itertools import combinations

import numpy as np
import torch

# The three unrooted resolutions of the ordered quartet (0,1,2,3), as the two
# pairs of within-pair positions. class c pairs position 0 with PAIR_PARTNER[c].
QUARTET_PAIRS = (
    ((0, 1), (2, 3)),  # class 0
    ((0, 2), (1, 3)),  # class 1
    ((0, 3), (1, 2)),  # class 2
)


def all_pairs_topo_distance(ete_tree, n_leaves):
    """Return an (n_leaves, n_leaves) int matrix of topological (edge-count)
    distances between leaves. Leaf names are assumed to be ints 0..n_leaves-1
    (as produced by src.simulation.dataset.label_tree_nodes).

    Runs a BFS from every leaf over the tree's node graph. O(n_leaves * nodes).
    """
    # Build adjacency over ete3 node objects (children + parent), unit weights.
    nodes = list(ete_tree.traverse())
    idx_of = {id(node): i for i, node in enumerate(nodes)}
    adj = [[] for _ in nodes]
    for node in nodes:
        i = idx_of[id(node)]
        for child in node.children:
            j = idx_of[id(child)]
            adj[i].append(j)
            adj[j].append(i)

    # Node index of each leaf name.
    leaf_node = [-1] * n_leaves
    for node in nodes:
        if node.is_leaf():
            leaf_node[int(node.name)] = idx_of[id(node)]

    dist = np.zeros((n_leaves, n_leaves), dtype=np.int64)
    n_nodes = len(nodes)
    for a in range(n_leaves):
        src = leaf_node[a]
        d = np.full(n_nodes, -1, dtype=np.int64)
        d[src] = 0
        frontier = [src]
        while frontier:
            nxt = []
            for u in frontier:
                du = d[u]
                for v in adj[u]:
                    if d[v] < 0:
                        d[v] = du + 1
                        nxt.append(v)
            frontier = nxt
        for b in range(n_leaves):
            dist[a, b] = d[leaf_node[b]]
    return dist


def label_quartets(quartets, dist):
    """Label an (Q, 4) int array of quartets (taxon ids) using distance matrix
    `dist`. Returns an (Q,) int64 tensor of classes in {0,1,2}."""
    q = np.asarray(quartets)
    a, b, c, d = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    sum0 = dist[a, b] + dist[c, d]
    sum1 = dist[a, c] + dist[b, d]
    sum2 = dist[a, d] + dist[b, c]
    sums = np.stack([sum0, sum1, sum2], axis=1)
    return torch.from_numpy(np.argmin(sums, axis=1)).long()


def enumerate_local_quartets(k):
    """All C(k,4) quartets over local indices 0..k-1 as an (Q,4) int64 tensor
    (each row sorted ascending). Deterministic order."""
    return torch.tensor(list(combinations(range(k), 4)), dtype=torch.long)


def shared_taxa_mask(local_quartets):
    """Boolean (Q, Q) mask: True where two quartets share >= 3 taxa (the
    QuartFormer sparse-attention neighborhood; the diagonal is always True since
    a quartet shares 4 taxa with itself)."""
    Q = local_quartets.shape[0]
    k = int(local_quartets.max().item()) + 1
    # Multi-hot membership (Q, k); shared count = onehot @ onehot.T.
    membership = torch.zeros(Q, k)
    membership.scatter_(1, local_quartets, 1.0)
    shared = membership @ membership.T
    return shared >= 3


def multihot_species(local_quartets, k):
    """Multi-hot (Q, k) indicator of the 4 taxa in each quartet."""
    Q = local_quartets.shape[0]
    m = torch.zeros(Q, k)
    m.scatter_(1, local_quartets, 1.0)
    return m


def sample_blocks(n_leaves, block_size, n_blocks, generator=None):
    """Sample `n_blocks` blocks, each `block_size` distinct taxon ids drawn
    without replacement from range(n_leaves). Returns a (n_blocks, block_size)
    long tensor. If block_size >= n_leaves every block is a permutation of all
    taxa (with wrap padding disallowed -> block_size is clamped)."""
    k = min(block_size, n_leaves)
    blocks = torch.empty(n_blocks, k, dtype=torch.long)
    for i in range(n_blocks):
        perm = torch.randperm(n_leaves, generator=generator)[:k]
        blocks[i] = perm
    return blocks


def coverage_blocks(n_leaves, block_size, min_per_taxon, generator=None):
    """Blocks for inference: keep sampling random `block_size` subsets until
    every taxon has appeared at least `min_per_taxon` times. Returns a
    (n_blocks, k) long tensor. Guarantees full taxon coverage so quartet
    puzzling has quartets touching every taxon."""
    k = min(block_size, n_leaves)
    counts = torch.zeros(n_leaves, dtype=torch.long)
    blocks = []
    while (counts < min_per_taxon).any():
        perm = torch.randperm(n_leaves, generator=generator)[:k]
        blocks.append(perm)
        counts[perm] += 1
    return torch.stack(blocks)
