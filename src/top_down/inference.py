"""Inference-time autoregressive tree sampling for the top down diffusion model.

Extracted from model.py. ``TopDownDiffusionModel`` mixes in ``TreeSamplingMixin``
for every tree-generation path (single-sample, temperature, corrector,
layer-corrector, beam). The training/loss logic stays in model.py; the
training-time *noise* sampling lives in sampling.py. Separate concerns by design.
"""

import dgl
import torch

from ..bottom_up.tree_utils import pairwise_hamming_frac
from ..sampling_utils import (
    child_lookup_to_tree,
    ensure_newick_semicolon,
    tuple_tree_to_newick,
)
from .graph_ops import build_obs_graph_aux
from .pda import _pda_left_prob_from_counts
from .percolation import (
    LeafConstraint,
    build_obs_basis,
    compute_constraints,
)


class TreeSamplingMixin:
    """Autoregressive percolation tree sampler, mixed into TopDownDiffusionModel.

    Expects the host to provide ``self.species_encoder``, ``self.rate_predictor``
    and ``self.offset`` (read by ``_apply_logit_offset``).
    """

    def _apply_logit_offset(
        self,
        logits,
        host_idx_in_obs,
        left_child,
        right_child,
        subtree_counts,
        host_counts,
    ):
        if self.offset is None:
            return logits
        if self.offset == "pda":
            n_leaves = host_idx_in_obs.numel()
            L = torch.zeros(n_leaves, device=host_idx_in_obs.device, dtype=torch.long)
            R = torch.zeros(n_leaves, device=host_idx_in_obs.device, dtype=torch.long)

            left_at_host = left_child[host_idx_in_obs]
            has_left = left_at_host.ne(-1)
            if has_left.any():
                L[has_left] = subtree_counts[left_at_host[has_left]].long()

            right_at_host = right_child[host_idx_in_obs]
            has_right = right_at_host.ne(-1)
            if has_right.any():
                R[has_right] = subtree_counts[right_at_host[has_right]].long()

            M = host_counts[host_idx_in_obs].long()
            p_left = _pda_left_prob_from_counts(L, R, M).to(subtree_counts.device)
            p_left = p_left.clamp(min=1e-6, max=1 - 1e-6)
            log_prior = torch.stack((torch.log(p_left), torch.log1p(-p_left)), dim=1)
            return logits + log_prior.to(logits.dtype)
        raise ValueError(
            f"Unsupported top down offset '{self.offset}'. "
            "Supported offsets are 'pda' and null (set model.offset=null)."
        )

    @staticmethod
    def _init_sampling_state(n_leaves, device):
        graph = dgl.graph(([], []), num_nodes=1, device=device)
        graph.ndata["d"] = torch.zeros(1, dtype=torch.long, device=device)
        graph.ndata["have"] = torch.tensor([n_leaves], dtype=torch.long, device=device)
        graph.ndata["subtree_count"] = torch.ones(1, dtype=torch.float, device=device)
        # basis_raw[v, i] = 1 iff taxon i's host is in subtree(v). Initially all
        # n_leaves taxa live at node 0, which is its own only subtree.
        graph.ndata["basis_raw"] = torch.ones(1, n_leaves, device=device)
        graph.edata["direction"] = torch.empty((0,), dtype=torch.bool, device=device)
        positions = torch.zeros(n_leaves, dtype=torch.long, device=device)
        return graph, positions

    @staticmethod
    def _percolate_leaf(
        graph,
        host_idx_in_obs,
        leaf_idx,
        direction,
        left_child,
        right_child,
        n_obs_per_graph,
    ):
        current_node = host_idx_in_obs[leaf_idx].item()
        next_node = (
            left_child[current_node] if direction == 0 else right_child[current_node]
        )

        if next_node == -1:
            graph.add_nodes(1)
            n_obs_per_graph = n_obs_per_graph + 1
            next_node = graph.num_nodes() - 1
            graph.ndata["have"][next_node] = 1
            graph.ndata["d"][next_node] = graph.ndata["d"][current_node] + 1
            graph.add_edges(next_node, current_node)
            graph.edata["direction"][-1] = direction
            graph.ndata["subtree_count"][next_node] = 1.0
        else:
            graph.ndata["have"][next_node] += 1
            graph.ndata["subtree_count"][next_node] += 1.0

        graph.ndata["have"][current_node] -= 1
        # Incremental basis update: only next_node's row flips (0 -> 1 for this taxon).
        # Ancestors' membership was already 1 via current_node; still 1 via next_node.
        graph.ndata["basis_raw"][next_node, leaf_idx] += 1.0
        host_idx_in_obs[leaf_idx] = next_node

    @torch.no_grad()
    def sample_tree(self, seqs, *, temperature=1.0, beam_size=1,
                    corrector_steps=0, corrector_drop_k=5,
                    layer_corrector=False, layer_drop_k=5, layer_lift_k=4,
                    layer_seed=None):
        if beam_size > 1:
            return self._sample_tree_beam(seqs, beam_size=beam_size)
        if layer_corrector:
            return self._sample_tree_layer_corrector(
                seqs, temperature=temperature,
                drop_k=layer_drop_k, lift_k=layer_lift_k, seed=layer_seed,
            )

        device = seqs.device
        n_leaves = seqs.shape[0]
        tree_to_n_leaves = torch.tensor([n_leaves], device=device)
        leaf_to_tree_id = torch.zeros(n_leaves, dtype=torch.long, device=device)

        pool = not self.rate_predictor.unpooled
        leaf_features = self.species_encoder(seqs, tree_to_n_leaves, pool=pool)

        # Precompute Hamming distances once (doesn't change during sampling)
        hamming_leaf = pairwise_hamming_frac(seqs.long())  # (n_leaves, n_leaves)

        paths = [[] for _ in range(n_leaves)]
        graph, host_idx_in_obs = self._init_sampling_state(n_leaves, device)
        n_obs_per_graph = self._sample_until_done(
            graph, host_idx_in_obs, leaf_features, hamming_leaf,
            tree_to_n_leaves, leaf_to_tree_id, n_leaves, temperature, paths,
        )

        if corrector_steps > 0:
            import random
            for _ in range(corrector_steps):
                K = min(corrector_drop_k, n_leaves)
                dropped = set(random.sample(range(n_leaves), K))
                kept_paths = [p if i not in dropped else [] for i, p in enumerate(paths)]
                # Rebuild graph: replay kept percolations. Order matters because
                # node indices are assigned as nodes are created — so iterate
                # round-robin (one step per leaf at a time) to roughly match the
                # original creation order. Order-within-replay doesn't change
                # final topology, only which graph node-id ends up where.
                graph, host_idx_in_obs = self._init_sampling_state(n_leaves, device)
                paths = [[] for _ in range(n_leaves)]
                max_len = max((len(p) for p in kept_paths), default=0)
                for step_i in range(max_len):
                    for leaf, dirs in enumerate(kept_paths):
                        if step_i < len(dirs):
                            self._replay_percolate(
                                graph, host_idx_in_obs, leaf, dirs[step_i],
                            )
                            paths[leaf].append(dirs[step_i])
                # Re-percolate dropped leaves at T=0 (argmax of corrected posterior)
                self._sample_until_done(
                    graph, host_idx_in_obs, leaf_features, hamming_leaf,
                    tree_to_n_leaves, leaf_to_tree_id, n_leaves, 0.0, paths,
                )

        # Get final left/right_child from the graph for tree reconstruction
        n_obs_per_graph = torch.tensor([graph.num_nodes()], device=device)
        spd, no_obs_children, left_child, right_child = build_obs_graph_aux(
            [graph], tree_to_n_leaves, n_obs_per_graph, host_idx_in_obs, leaf_to_tree_id,
        )
        if corrector_steps > 0:
            # Dropping leaves can leave single-child internal nodes ("kinks")
            # behind — they're nodes the dropped leaves used to branch off but
            # the remaining percolations don't fill. Splice them out so the
            # final binary tree is well-formed.
            left_child, right_child, host_idx_in_obs = self._collapse_kinks(
                left_child.clone(), right_child.clone(), host_idx_in_obs.clone(), n_leaves,
            )
        return child_lookup_to_tree(host_idx_in_obs, left_child, right_child)

    @staticmethod
    def _collapse_kinks(left_child, right_child, host_idx_in_obs, n_leaves):
        """Splice out internal nodes with exactly one child. A leaf parked at a
        kink moves to the kink's only child."""
        n_nodes = left_child.numel()
        parent = [-1] * n_nodes
        parent_dir = [-1] * n_nodes
        for i in range(n_nodes):
            l, r = int(left_child[i].item()), int(right_child[i].item())
            if l != -1:
                parent[l] = i
                parent_dir[l] = 0
            if r != -1:
                parent[r] = i
                parent_dir[r] = 1

        changed = True
        while changed:
            changed = False
            for i in range(n_nodes):
                l, r = int(left_child[i].item()), int(right_child[i].item())
                if (l == -1) == (r == -1):
                    continue  # both set or both -1, not a kink
                child = l if l != -1 else r
                p = parent[i]
                if p == -1:
                    continue  # root with single child — skip (rare; caller can re-root)
                # rewire parent → child, bypassing the kink
                if parent_dir[i] == 0:
                    left_child[p] = child
                else:
                    right_child[p] = child
                parent[child] = p
                parent_dir[child] = parent_dir[i]
                # any leaf parked at i moves to its surviving child
                for leaf in range(n_leaves):
                    if int(host_idx_in_obs[leaf].item()) == i:
                        host_idx_in_obs[leaf] = child
                left_child[i] = -1
                right_child[i] = -1
                changed = True
        return left_child, right_child, host_idx_in_obs

    @torch.no_grad()
    def _replay_percolate(self, graph, host_idx_in_obs, leaf, direction):
        """Same effect as _percolate_leaf but reads the existing graph for the
        next-node lookup, so it doesn't depend on a fresh build_obs_graph_aux."""
        current_node = host_idx_in_obs[leaf].item()
        # Find existing child of current_node along `direction`, if any.
        in_src, in_dst = graph.in_edges(current_node)
        next_node = -1
        if in_src.numel() > 0:
            edge_ids = graph.edge_ids(in_src, in_dst)
            edge_dirs = graph.edata["direction"][edge_ids]
            for i in range(in_src.numel()):
                if int(edge_dirs[i].item()) == direction:
                    next_node = int(in_src[i].item())
                    break
        if next_node == -1:
            graph.add_nodes(1)
            next_node = graph.num_nodes() - 1
            graph.ndata["have"][next_node] = 1
            graph.ndata["d"][next_node] = graph.ndata["d"][current_node] + 1
            graph.add_edges(next_node, current_node)
            graph.edata["direction"][-1] = bool(direction)
            graph.ndata["subtree_count"][next_node] = 1.0
        else:
            graph.ndata["have"][next_node] += 1
            graph.ndata["subtree_count"][next_node] += 1.0
        graph.ndata["have"][current_node] -= 1
        graph.ndata["basis_raw"][next_node, leaf] += 1.0
        host_idx_in_obs[leaf] = next_node

    @torch.no_grad()
    def _sample_until_done(
        self, graph, host_idx_in_obs, leaf_features, hamming_leaf,
        tree_to_n_leaves, leaf_to_tree_id, n_leaves, temperature, paths,
    ):
        """Run the sampling loop until no leaves remain eligible. Records each
        chosen direction in `paths[leaf]`."""
        device = host_idx_in_obs.device
        while True:
            n_obs_per_graph = torch.tensor([graph.num_nodes()], device=device)

            spd, no_observed_children, left_child, right_child = build_obs_graph_aux(
                [graph],
                tree_to_n_leaves,
                n_obs_per_graph,
                host_idx_in_obs,
                leaf_to_tree_id,
            )

            constraints = compute_constraints(
                graph,
                host_idx_in_obs,
                no_observed_children,
                left_child,
                right_child,
            )

            eligible_mask = ~torch.isin(
                constraints,
                torch.tensor(
                    [LeafConstraint.FINAL.value, LeafConstraint.WAITING.value],
                    device=constraints.device,
                ),
            )

            if not eligible_mask.any():
                break

            obs_basis = build_obs_basis(graph, n_leaves, host_idx_in_obs)

            logits = self.rate_predictor(
                leaf_features,
                spd,
                tree_to_n_leaves.tolist(),
                host_idx_in_obs,
                n_obs_per_graph.tolist(),
                graph.ndata["d"],
                graph.ndata["subtree_count"],
                left_child,
                right_child,
                hamming_leaf=hamming_leaf,
                obs_basis=obs_basis,
            )
            logits = self._apply_logit_offset(
                logits,
                host_idx_in_obs,
                left_child,
                right_child,
                graph.ndata["subtree_count"],
                graph.ndata["have"],
            )

            # Jointly pick (taxon, direction) from the full logits matrix.
            # Mask invalid cells: ineligible taxa, and directions forbidden by constraints.
            valid = eligible_mask.unsqueeze(1).expand_as(logits).clone()
            valid[constraints == LeafConstraint.MUST_LEFT.value, 1] = False
            valid[constraints == LeafConstraint.MUST_RIGHT.value, 0] = False
            # Row-normalize so each eligible taxon contributes equal mass at T=1:
            # training only normalizes within rows (per-taxon direction softmax),
            # so cross-row scale carries no signal — flat-sampling raw logits
            # would marginalize over an untrained P(taxon). After this, each
            # row sums to 1 in prob space, so flat sampling at T=1 is uniform
            # over eligible taxa with the trained per-taxon direction.
            # fp32 cast also avoids bf16 underflow inside Categorical.
            masked = logits.float().masked_fill(~valid, float("-inf"))
            lse = torch.logsumexp(masked, dim=-1, keepdim=True)
            row_finite = lse.isfinite()
            flat_logits = torch.where(row_finite, masked - lse, masked).flatten()
            if temperature <= 0:
                selected = flat_logits.argmax().item()
            else:
                selected = torch.distributions.Categorical(
                    logits=flat_logits / temperature
                ).sample().item()
            selected_taxon = selected // 2
            selected_direction = selected % 2

            # Percolate taxon and update graph
            self._percolate_leaf(
                graph,
                host_idx_in_obs,
                selected_taxon,
                selected_direction,
                left_child,
                right_child,
                n_obs_per_graph,
            )
            paths[selected_taxon].append(int(selected_direction))

    @torch.no_grad()
    def _rebuild_graph_from_paths(self, paths, n_leaves, device):
        """Vectorized graph reconstruction from per-leaf direction paths.

        The slow path (sequential _replay_percolate calls) is O(n^3) due to
        repeated DGL in_edges lookups + per-call mutations + GPU-CPU sync per
        item(). This builds the entire DGL graph in one shot:
          - enumerate every prefix any leaf has traversed (these are graph
            nodes), assign IDs depth-first so parents precede children,
          - tally `have`, `subtree_count`, `basis_raw` and `host_idx_in_obs`
            on CPU in O(n * max_depth),
          - construct the DGL graph once with bulk add and move all attribute
            tensors to device in a single transfer.

        Equivalent output to the loop version up to node-id permutation; both
        downstream consumers index by host_idx_in_obs so the relabeling is
        invisible.
        """
        prefix_set = {()}
        for path in paths:
            for k in range(1, len(path) + 1):
                prefix_set.add(tuple(path[:k]))
        sorted_prefixes = sorted(prefix_set, key=lambda p: (len(p), p))
        prefix_to_id = {p: i for i, p in enumerate(sorted_prefixes)}
        n_nodes = len(sorted_prefixes)

        depths_cpu = torch.tensor(
            [len(p) for p in sorted_prefixes], dtype=torch.long,
        )
        have_cpu = torch.zeros(n_nodes, dtype=torch.long)
        subtree_cpu = torch.zeros(n_nodes, dtype=torch.float)
        basis_cpu = torch.zeros(n_nodes, n_leaves)
        host_cpu = torch.empty(n_leaves, dtype=torch.long)

        for leaf, path in enumerate(paths):
            cur = ()
            cur_id = prefix_to_id[cur]
            basis_cpu[cur_id, leaf] = 1.0
            subtree_cpu[cur_id] += 1.0
            for direction in path:
                cur = cur + (direction,)
                cur_id = prefix_to_id[cur]
                basis_cpu[cur_id, leaf] = 1.0
                subtree_cpu[cur_id] += 1.0
            have_cpu[cur_id] += 1
            host_cpu[leaf] = cur_id

        src_edges, dst_edges, edge_dirs = [], [], []
        for prefix, parent_id in prefix_to_id.items():
            for direction in (0, 1):
                child_id = prefix_to_id.get(prefix + (direction,))
                if child_id is not None:
                    src_edges.append(child_id)
                    dst_edges.append(parent_id)
                    edge_dirs.append(direction)

        if src_edges:
            src_t = torch.tensor(src_edges, dtype=torch.long)
            dst_t = torch.tensor(dst_edges, dtype=torch.long)
            graph = dgl.graph((src_t, dst_t), num_nodes=n_nodes, device=device)
            graph.edata["direction"] = torch.tensor(
                edge_dirs, dtype=torch.bool, device=device,
            )
        else:
            graph = dgl.graph(([], []), num_nodes=n_nodes, device=device)
            graph.edata["direction"] = torch.empty(
                (0,), dtype=torch.bool, device=device,
            )
        graph.ndata["d"] = depths_cpu.to(device, non_blocking=True)
        graph.ndata["have"] = have_cpu.to(device, non_blocking=True)
        graph.ndata["subtree_count"] = subtree_cpu.to(device, non_blocking=True)
        graph.ndata["basis_raw"] = basis_cpu.to(device, non_blocking=True)
        return graph, host_cpu.to(device, non_blocking=True)

    @torch.no_grad()
    def _sample_tree_layer_corrector(
        self, seqs, *, temperature=1.0, drop_k=5, lift_k=4, seed=None,
    ):
        """Layer-by-layer top-down sampler with predictor-corrector dance.

        At each tree layer, every internal cluster with 2+ leaves needs to split
        into a left and right subtree. Within a layer we:
          - randomly assign one leaf in each cluster to L (breaks L/R symmetry,
            since the two sides are exchangeable until the seed),
          - repeatedly: forward-pass for predictions on uncommitted leaves,
            drop up to drop_k random uncommitted leaves (sampling direction at T),
            lift up to lift_k random already-committed leaves (random selection).
          - terminate the layer when no uncommitted leaves remain.

        Net progress per iteration: drop_k - lift_k leaves settled.
        """
        import random
        rng = random.Random(seed)

        device = seqs.device
        n_leaves = seqs.shape[0]
        tree_to_n_leaves = torch.tensor([n_leaves], device=device)
        leaf_to_tree_id = torch.zeros(n_leaves, dtype=torch.long, device=device)

        pool = not self.rate_predictor.unpooled
        leaf_features = self.species_encoder(seqs, tree_to_n_leaves, pool=pool)
        hamming_leaf = pairwise_hamming_frac(seqs.long())

        paths = [[] for _ in range(n_leaves)]
        max_layers = 2 * n_leaves  # safety bound (depth <= n_leaves - 1)
        layer = 0

        while layer < max_layers:
            # Group leaves by their length-`layer` path prefix; only multi-leaf
            # prefixes are active clusters that still need a layer-`layer` decision.
            prefix_to_leaves = {}
            for leaf in range(n_leaves):
                if len(paths[leaf]) >= layer:
                    prefix = tuple(paths[leaf][:layer])
                    prefix_to_leaves.setdefault(prefix, []).append(leaf)
            active_clusters = {
                p: ls for p, ls in prefix_to_leaves.items() if len(ls) >= 2
            }
            if not active_clusters:
                break

            cluster_done = {p: False for p in active_clusters}
            max_iter = 6 * max(len(ls) for ls in active_clusters.values())
            for _ in range(max_iter):
                # Setup: every active+not-done cluster needs >=1 committed leaf
                # to break L/R symmetry. Seed direction is arbitrary (chosen 0).
                for prefix, leaves in active_clusters.items():
                    if cluster_done[prefix]:
                        continue
                    if not any(len(paths[l]) == layer + 1 for l in leaves):
                        seed_leaf = rng.choice(leaves)
                        paths[seed_leaf].append(0)

                if all(cluster_done.values()):
                    break

                # Build the union of uncommitted-at-this-layer leaves across all
                # not-done clusters — used only to mask the model's logits.
                pending_clusters = {
                    p: ls for p, ls in active_clusters.items() if not cluster_done[p]
                }
                uncommitted = [
                    l for leaves in pending_clusters.values()
                    for l in leaves if len(paths[l]) == layer
                ]
                if not uncommitted:
                    break

                graph, host_idx_in_obs = self._rebuild_graph_from_paths(
                    paths, n_leaves, device,
                )
                n_obs_per_graph = torch.tensor(
                    [graph.num_nodes()], device=device,
                )
                spd, no_obs_children, left_child, right_child = build_obs_graph_aux(
                    [graph], tree_to_n_leaves, n_obs_per_graph,
                    host_idx_in_obs, leaf_to_tree_id,
                )
                constraints = compute_constraints(
                    graph, host_idx_in_obs, no_obs_children,
                    left_child, right_child,
                )
                obs_basis = build_obs_basis(graph, n_leaves, host_idx_in_obs)
                logits = self.rate_predictor(
                    leaf_features, spd, tree_to_n_leaves.tolist(),
                    host_idx_in_obs, n_obs_per_graph.tolist(),
                    graph.ndata["d"], graph.ndata["subtree_count"],
                    left_child, right_child,
                    hamming_leaf=hamming_leaf, obs_basis=obs_basis,
                )
                logits = self._apply_logit_offset(
                    logits, host_idx_in_obs, left_child, right_child,
                    graph.ndata["subtree_count"], graph.ndata["have"],
                )

                # Mask: only sample from uncommitted-at-this-layer rows; respect
                # MUST_LEFT/RIGHT.
                valid = torch.zeros_like(logits, dtype=torch.bool)
                uncommitted_idx = torch.tensor(
                    uncommitted, dtype=torch.long, device=device,
                )
                valid[uncommitted_idx, 0] = True
                valid[uncommitted_idx, 1] = True
                valid[constraints == LeafConstraint.MUST_LEFT.value, 1] = False
                valid[constraints == LeafConstraint.MUST_RIGHT.value, 0] = False
                masked = logits.float().masked_fill(~valid, float("-inf"))
                row_lse = torch.logsumexp(masked, dim=-1, keepdim=True)

                # Per-cluster drop: each not-done cluster independently drops up
                # to drop_k of its own uncommitted leaves with model-sampled
                # directions at temperature T. We collect every cluster's drop
                # picks first, then do ONE batched Categorical sample so the
                # sync count stays at 1 per iter regardless of cluster count.
                drop_leaf_idx = []
                for prefix, leaves in pending_clusters.items():
                    uncommitted_in_cluster = [
                        l for l in leaves if len(paths[l]) == layer
                    ]
                    n_drop = min(drop_k, len(uncommitted_in_cluster))
                    drop_leaf_idx.extend(rng.sample(uncommitted_in_cluster, n_drop))

                if drop_leaf_idx:
                    drop_idx_t = torch.tensor(
                        drop_leaf_idx, dtype=torch.long, device=device,
                    )
                    rows = masked.index_select(0, drop_idx_t)            # (D, 2)
                    row_lse_d = row_lse.index_select(0, drop_idx_t)      # (D, 1)
                    log_probs = rows - row_lse_d                          # (D, 2)
                    if temperature <= 0:
                        directions_t = log_probs.argmax(dim=-1)
                    else:
                        directions_t = torch.distributions.Categorical(
                            logits=log_probs / temperature
                        ).sample()
                    # Encode "no valid direction" as -1 so a single .tolist()
                    # carries both the chosen direction and a validity flag —
                    # one GPU->CPU sync per iter regardless of drop count.
                    valid_drops = row_lse_d.squeeze(-1).isfinite()
                    directions_t = torch.where(
                        valid_drops, directions_t,
                        torch.full_like(directions_t, -1),
                    )
                    directions = directions_t.tolist()
                    for i, leaf in enumerate(drop_leaf_idx):
                        d = directions[i]
                        if d >= 0:
                            paths[leaf].append(d)

                # Per-cluster done-check + lift. A cluster terminates the moment
                # its drops empty its uncommitted set; on that iter we apply the
                # degenerate-split guard and skip its lift. Other clusters keep
                # iterating with lift_k random reverts.
                for prefix, leaves in pending_clusters.items():
                    uncommitted_after = [
                        l for l in leaves if len(paths[l]) == layer
                    ]
                    if not uncommitted_after:
                        # Degenerate-split guard: in parallel drops the per-leaf
                        # MUST_LEFT/MUST_RIGHT constraint can become stale, so a
                        # cluster's leaves can all end up on the seed's side.
                        # Flip one random leaf to balance the split.
                        dirs = [paths[l][layer] for l in leaves]
                        if all(d == 0 for d in dirs):
                            paths[rng.choice(leaves)][layer] = 1
                        elif all(d == 1 for d in dirs):
                            paths[rng.choice(leaves)][layer] = 0
                        cluster_done[prefix] = True
                    else:
                        committed_in_cluster = [
                            l for l in leaves if len(paths[l]) == layer + 1
                        ]
                        n_lift = min(lift_k, len(committed_in_cluster))
                        if n_lift > 0:
                            lift_leaves = rng.sample(
                                committed_in_cluster, n_lift,
                            )
                            for leaf in lift_leaves:
                                paths[leaf].pop()
            layer += 1

        graph, host_idx_in_obs = self._rebuild_graph_from_paths(
            paths, n_leaves, device,
        )
        n_obs_per_graph = torch.tensor([graph.num_nodes()], device=device)
        spd, no_obs_children, left_child, right_child = build_obs_graph_aux(
            [graph], tree_to_n_leaves, n_obs_per_graph,
            host_idx_in_obs, leaf_to_tree_id,
        )
        return child_lookup_to_tree(host_idx_in_obs, left_child, right_child)

    def _beam_step_eval(self, graph, host_idx_in_obs, n_leaves, leaf_features, hamming_leaf):
        """Score one beam: returns (flat_log_probs, n_valid, left_child, right_child).

        flat_log_probs is length 2*n_leaves with -inf on invalid (taxon, direction)
        actions, or None if the beam has no eligible leaves remaining.
        """
        device = graph.device
        tree_to_n_leaves = torch.tensor([n_leaves], device=device)
        leaf_to_tree_id = torch.zeros(n_leaves, dtype=torch.long, device=device)
        n_obs_per_graph = torch.tensor([graph.num_nodes()], device=device)

        spd, no_observed_children, left_child, right_child = build_obs_graph_aux(
            [graph], tree_to_n_leaves, n_obs_per_graph,
            host_idx_in_obs, leaf_to_tree_id,
        )
        constraints = compute_constraints(
            graph, host_idx_in_obs, no_observed_children, left_child, right_child,
        )
        eligible_mask = ~torch.isin(
            constraints,
            torch.tensor(
                [LeafConstraint.FINAL.value, LeafConstraint.WAITING.value],
                device=constraints.device,
            ),
        )
        if not eligible_mask.any():
            return None, 0, left_child, right_child

        obs_basis = build_obs_basis(graph, n_leaves, host_idx_in_obs)
        logits = self.rate_predictor(
            leaf_features, spd, tree_to_n_leaves.tolist(),
            host_idx_in_obs, n_obs_per_graph.tolist(),
            graph.ndata["d"], graph.ndata["subtree_count"],
            left_child, right_child,
            hamming_leaf=hamming_leaf, obs_basis=obs_basis,
        )
        logits = self._apply_logit_offset(
            logits, host_idx_in_obs, left_child, right_child,
            graph.ndata["subtree_count"], graph.ndata["have"],
        )

        valid = eligible_mask.unsqueeze(1).expand_as(logits).clone()
        valid[constraints == LeafConstraint.MUST_LEFT.value, 1] = False
        valid[constraints == LeafConstraint.MUST_RIGHT.value, 0] = False
        masked_logits = logits.masked_fill(~valid, float("-inf"))
        flat_log_probs = torch.log_softmax(masked_logits.flatten().float(), dim=0)
        return flat_log_probs, int(valid.sum().item()), left_child, right_child

    @torch.no_grad()
    def _sample_tree_beam(self, seqs, *, beam_size):
        device = seqs.device
        n_leaves = seqs.shape[0]
        tree_to_n_leaves = torch.tensor([n_leaves], device=device)

        pool = not self.rate_predictor.unpooled
        leaf_features = self.species_encoder(seqs, tree_to_n_leaves, pool=pool)
        hamming_leaf = pairwise_hamming_frac(seqs.long())

        g0, h0 = self._init_sampling_state(n_leaves, device)
        beams = [{
            "graph": g0, "host": h0,
            "log_prob": 0.0,
            "left_child": None, "right_child": None,
            "done": False,
        }]

        while True:
            any_active = False
            candidates = []  # (parent_idx, taxon, direction, new_log_prob)
            for parent_idx, b in enumerate(beams):
                if b["done"]:
                    continue
                flat_lp, n_valid, lc, rc = self._beam_step_eval(
                    b["graph"], b["host"], n_leaves, leaf_features, hamming_leaf,
                )
                if flat_lp is None:
                    b["done"] = True
                    b["left_child"], b["right_child"] = lc, rc
                    continue
                any_active = True
                b["left_child"], b["right_child"] = lc, rc
                k_take = min(beam_size, n_valid)
                topk_lp, topk_idx = flat_lp.topk(k_take)
                for lp_v, idx_v in zip(topk_lp.tolist(), topk_idx.tolist()):
                    candidates.append(
                        (parent_idx, idx_v // 2, idx_v % 2, b["log_prob"] + lp_v)
                    )

            if not any_active:
                break

            # Compete: carry-forward done beams vs new candidates, take top beam_size.
            scored = [(b["log_prob"], "done", i) for i, b in enumerate(beams) if b["done"]]
            scored += [(c[3], "new", c) for c in candidates]
            scored.sort(key=lambda x: x[0], reverse=True)
            scored = scored[:beam_size]

            new_beams = []
            for _, kind, payload in scored:
                if kind == "done":
                    new_beams.append(beams[payload])
                    continue
                parent_idx, taxon, direction, new_lp = payload
                parent = beams[parent_idx]
                new_graph = parent["graph"].clone()
                new_host = parent["host"].clone()
                n_obs = torch.tensor([new_graph.num_nodes()], device=device)
                self._percolate_leaf(
                    new_graph, new_host, taxon, direction,
                    parent["left_child"], parent["right_child"], n_obs,
                )
                new_beams.append({
                    "graph": new_graph, "host": new_host,
                    "log_prob": new_lp,
                    "left_child": None, "right_child": None,
                    "done": False,
                })
            beams = new_beams

        best = max(beams, key=lambda b: b["log_prob"])
        return child_lookup_to_tree(
            best["host"], best["left_child"], best["right_child"],
        )

    @torch.no_grad()
    def sample_from_batch(
        self, batch, *, batch_index=None, datamodule=None,
        temperature=1.0, beam_size=1, corrector_steps=0, corrector_drop_k=5,
        layer_corrector=False, layer_drop_k=5, layer_lift_k=4, layer_seed=None,
        **kwargs,
    ):
        seqs, trees = batch
        assert seqs.shape[0] == 1, "Batch size > 1 not supported"
        true_tree = trees[0]
        sampled_tree = self.sample_tree(
            seqs[0], temperature=temperature, beam_size=beam_size,
            corrector_steps=corrector_steps, corrector_drop_k=corrector_drop_k,
            layer_corrector=layer_corrector,
            layer_drop_k=layer_drop_k, layer_lift_k=layer_lift_k,
            layer_seed=layer_seed,
        )
        sampled_leaf_names = [str(i) for i in range(seqs.shape[1])]

        return {
            "true_tree_newick": ensure_newick_semicolon(true_tree.write(format=9)),
            "sampled_tree_newick": ensure_newick_semicolon(
                tuple_tree_to_newick(sampled_tree, names=sampled_leaf_names)
            ),
        }
