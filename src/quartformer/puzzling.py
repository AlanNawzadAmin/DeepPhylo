"""Weighted quartet puzzling: assemble a full unrooted tree from per-quartet
topology predictions (Strimmer & von Haeseler, 1996), plus a majority consensus
over random insertion orders.

The model gives, for many 4-taxon subsets, a distribution over the three
resolutions. `QuartetOracle` stores the (averaged) predictions. `puzzle_once`
inserts taxa one at a time in a random order: to place taxon k it scores every
edge of the current tree by the (confidence-weighted) number of quartets
{k,x,y,z} whose predicted resolution disagrees with attaching k at that edge, and
inserts k on the minimum-penalty edge. `quartet_puzzling` repeats this over many
orders and returns the majority-rule consensus Newick.

Edge scoring uses the exact induced-pairing rule: attaching k at edge e=(u,v)
subdivides it with a node s; the induced resolution of {k,x,y,z} pairs k with the
leaf w in {x,y,z} minimizing g_e(w) + d(other two), where g_e(w) =
min(dist(u,w), dist(v,w)) and d(.,.) are current-tree topological distances (the
shared +1 from the s-k edge cancels across the three options).
"""

from collections import deque

import numpy as np

from .quartets import QUARTET_PAIRS


def _partner_in_class(sorted4, cls, k):
    """Given a sorted 4-tuple, a resolution class, and a taxon k in it, return
    the taxon paired with k under that resolution."""
    for i, j in QUARTET_PAIRS[cls]:
        if sorted4[i] == k:
            return sorted4[j]
        if sorted4[j] == k:
            return sorted4[i]
    raise ValueError("k not in quartet")


class QuartetOracle:
    """Accumulates per-quartet class probabilities keyed by the sorted 4-tuple
    of taxa, averaging duplicate observations across blocks."""

    def __init__(self):
        self._sum = {}
        self._cnt = {}

    def add(self, quartet_taxa, probs):
        """quartet_taxa: sorted tuple of 4 taxon ids. probs: length-3 array."""
        key = tuple(int(t) for t in quartet_taxa)
        probs = np.asarray(probs, dtype=np.float64)
        if key in self._sum:
            self._sum[key] += probs
            self._cnt[key] += 1
        else:
            self._sum[key] = probs.copy()
            self._cnt[key] = 1

    def add_block(self, block_taxa, local_quartets, probs):
        """Vectorized ingest of one block. block_taxa maps local index -> taxon
        id; local_quartets is (Q,4) local indices; probs is (Q,3)."""
        block_taxa = np.asarray(block_taxa)
        lq = np.asarray(local_quartets)
        gq = block_taxa[lq]  # (Q,4) global taxa (already ascending: block-local
        # quartets are combinations of ascending local idx, but global mapping
        # need not be monotone, so re-sort each row and permute probs? The class
        # convention depends on the sorted-taxa order, and add() keys by sorted
        # tuple while the model predicted classes w.r.t. the block-local order.)
        probs = np.asarray(probs, dtype=np.float64)
        for row, p in zip(gq, probs):
            order = np.argsort(row)
            sorted_taxa = row[order]
            self.add(tuple(sorted_taxa), _reorder_probs(p, order))

    def mean(self, quartet_taxa):
        key = tuple(sorted(int(t) for t in quartet_taxa))
        if key not in self._sum:
            return None
        return self._sum[key] / self._cnt[key]

    def neighbors_index(self):
        """Build, for each taxon, the list of (sorted4, meanprobs) it appears in.
        Used to fetch quartets {k, *} quickly during insertion."""
        idx = {}
        for key, s in self._sum.items():
            mp = s / self._cnt[key]
            for t in key:
                idx.setdefault(t, []).append((key, mp))
        return idx


def _reorder_probs(probs, order):
    """Remap a length-3 class distribution defined over taxa order `orig` to the
    class distribution over the permuted (sorted) taxa order `order`.

    Class c corresponds to a split (pairing) of the 4 positions. Permuting the
    positions maps each split to another split; we move probability mass
    accordingly. `order` is a permutation of (0,1,2,3): new position p holds old
    position order[p]."""
    # split-set representation of each class over ORIGINAL positions
    orig_splits = [frozenset(map(frozenset, QUARTET_PAIRS[c])) for c in range(3)]
    inv = np.argsort(order)  # old position -> new position
    out = np.zeros(3, dtype=np.float64)
    for c in range(3):
        # remap old positions to new positions
        remapped = frozenset(
            frozenset(int(inv[pos]) for pos in pair) for pair in QUARTET_PAIRS[c]
        )
        new_c = orig_splits.index(remapped)
        out[new_c] += probs[c]
    return out


class _PuzzleTree:
    """Minimal unrooted tree as an adjacency map for incremental insertion.
    Leaf node ids are taxon ids (>= 0); internal node ids are >= INTERNAL_BASE."""

    INTERNAL_BASE = 1_000_000

    def __init__(self):
        self.adj = {}
        self._next_internal = self.INTERNAL_BASE
        self.leaves = []

    def _new_internal(self):
        nid = self._next_internal
        self._next_internal += 1
        self.adj[nid] = set()
        return nid

    def _add_edge(self, a, b):
        self.adj.setdefault(a, set()).add(b)
        self.adj.setdefault(b, set()).add(a)

    def _rm_edge(self, a, b):
        self.adj[a].discard(b)
        self.adj[b].discard(a)

    def init_quartet(self, a, b, c, d):
        """Initialize with resolution (a,b)|(c,d)."""
        m1, m2 = self._new_internal(), self._new_internal()
        for leaf in (a, b, c, d):
            self.adj.setdefault(leaf, set())
        self._add_edge(a, m1)
        self._add_edge(b, m1)
        self._add_edge(m1, m2)
        self._add_edge(c, m2)
        self._add_edge(d, m2)
        self.leaves = [a, b, c, d]

    def edges(self):
        seen = set()
        out = []
        for u, nbrs in self.adj.items():
            for v in nbrs:
                key = (u, v) if u < v else (v, u)
                if key not in seen:
                    seen.add(key)
                    out.append((u, v))
        return out

    def node_leaf_distances(self):
        """Return {node: {leaf: dist}} topological distances to current leaves,
        via BFS from each leaf."""
        dist = {}
        for leaf in self.leaves:
            d = {leaf: 0}
            dq = deque([leaf])
            while dq:
                u = dq.popleft()
                for v in self.adj[u]:
                    if v not in d:
                        d[v] = d[u] + 1
                        dq.append(v)
            for node, dd in d.items():
                dist.setdefault(node, {})[leaf] = dd
        return dist

    def insert_at_edge(self, edge, k):
        u, v = edge
        self._rm_edge(u, v)
        s = self._new_internal()
        self._add_edge(u, s)
        self._add_edge(v, s)
        self._add_edge(s, k)
        self.leaves.append(k)

    def to_newick(self):
        # Root at any internal node; DFS.
        root = next(n for n in self.adj if n >= self.INTERNAL_BASE)

        def rec(node, parent):
            children = [c for c in self.adj[node] if c != parent]
            if not children:  # leaf
                return str(node)
            inner = ",".join(rec(c, node) for c in children)
            return f"({inner})"

        return rec(root, None) + ";"


def _sample_class(mp, temperature, rng):
    """Pick a resolution class (0/1/2) from mean probs `mp` (length-3).

    temperature == 0 -> argmax (deterministic, the point-estimate behavior).
    temperature > 0  -> draw from the temperature-scaled distribution
    p_c^(1/T) / sum_c p_c^(1/T) (equivalently softmax(log p / T)). T -> 1
    samples proportional to the model's own confidence; larger T flattens
    toward uniform, injecting more topological diversity across draws.
    Returns 0 when `mp` is missing (unseen quartet), matching the old default."""
    if mp is None:
        return 0
    if temperature <= 0:
        return int(np.argmax(mp))
    logp = np.log(np.clip(mp, 1e-12, None)) / temperature
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    return int(rng.choice(3, p=p))


def puzzle_once(taxa, oracle_index, rng, max_quartets_per_insertion=400,
                temperature=0.0):
    """One puzzling pass over a random insertion order. `oracle_index` maps taxon
    -> list of (sorted4, meanprobs). Returns a Newick string.

    `temperature` controls how each quartet's resolution is chosen: 0 uses the
    argmax (point estimate); >0 samples it, turning each pass into a draw whose
    topology is stochastic (see `_sample_class`)."""
    order = list(taxa)
    rng.shuffle(order)
    tree = _PuzzleTree()

    # Seed with the first four taxa using their predicted resolution.
    first4 = tuple(sorted(order[:4]))
    mp = _lookup(oracle_index, first4)
    cls = _sample_class(mp, temperature, rng)
    (i, j), (p, q) = QUARTET_PAIRS[cls]
    tree.init_quartet(first4[i], first4[j], first4[p], first4[q])
    inserted = set(order[:4])

    for k in order[4:]:
        # Candidate quartets {k, x, y, z} with x,y,z already inserted.
        cand = []
        for key, mp in oracle_index.get(k, ()):
            others = [t for t in key if t != k]
            if all(t in inserted for t in others):
                cand.append((key, mp))
        if len(cand) > max_quartets_per_insertion:
            sel = rng.choice(len(cand), size=max_quartets_per_insertion, replace=False)
            cand = [cand[s] for s in sel]

        edges = tree.edges()
        dist = tree.node_leaf_distances()
        penalty = np.zeros(len(edges), dtype=np.float64)

        if cand:
            # Precompute g_e(w) = min(dist(u,w), dist(v,w)) for edges x leaves.
            leaf_pos = {leaf: p for p, leaf in enumerate(tree.leaves)}
            L = len(tree.leaves)
            du = np.array([[dist[u].get(leaf, 10**6) for leaf in tree.leaves]
                           for (u, _) in edges])  # (E, L)
            dv = np.array([[dist[v].get(leaf, 10**6) for leaf in tree.leaves]
                           for (_, v) in edges])
            g = np.minimum(du, dv)  # (E, L)
            for key, mp in cand:
                others = [t for t in key if t != k]
                x, y, z = others
                px, py, pz = leaf_pos[x], leaf_pos[y], leaf_pos[z]
                dyz = _cur_dist(dist, y, z)
                dxz = _cur_dist(dist, x, z)
                dxy = _cur_dist(dist, x, y)
                opts = np.stack([
                    g[:, px] + dyz,   # pair k-x
                    g[:, py] + dxz,   # pair k-y
                    g[:, pz] + dxy,   # pair k-z
                ], axis=1)  # (E,3)
                induced = np.argmin(opts, axis=1)  # 0->x,1->y,2->z
                induced_partner = np.array([x, y, z])[induced]  # (E,)
                cls = _sample_class(mp, temperature, rng)
                weight = float(mp[cls])
                pred_partner = _partner_in_class(key, cls, k)
                penalty += weight * (induced_partner != pred_partner)

        best = int(np.argmin(penalty)) if len(edges) else 0
        tree.insert_at_edge(edges[best], k)
        inserted.add(k)

    return tree.to_newick()


def _cur_dist(dist, a, b):
    return dist[a][b]


def _lookup(oracle_index, sorted4):
    for key, mp in oracle_index.get(sorted4[0], ()):
        if key == sorted4:
            return mp
    return None


def quartet_puzzling(taxa, oracle, n_orderings=50, seed=0,
                     max_quartets_per_insertion=400, min_freq=0.5,
                     temperature=0.0, return_trees=False):
    """Run `n_orderings` puzzling passes and return the majority-rule consensus
    Newick over the resulting trees.

    `temperature` is forwarded to `puzzle_once`: 0 keeps the deterministic
    point-estimate decoding; >0 makes each pass a stochastic draw from the
    model's (temperature-scaled) per-quartet distribution, so the passes become
    approximate posterior samples and the consensus summarizes them.

    If `return_trees` is True, return `(consensus_newick, [per_ordering_newick,
    ...])` so callers can inspect the individual draws (e.g. sample diversity)."""
    from sample import _majority_consensus_newick

    index = oracle.neighbors_index()
    newicks = []
    for i in range(n_orderings):
        sub_rng = np.random.default_rng(seed + 1 + i)
        newicks.append(
            puzzle_once(taxa, index, sub_rng,
                        max_quartets_per_insertion=max_quartets_per_insertion,
                        temperature=temperature)
        )
    consensus = (newicks[0] if n_orderings == 1
                 else _majority_consensus_newick(newicks, min_freq=min_freq))
    if return_trees:
        return consensus, newicks
    return consensus
