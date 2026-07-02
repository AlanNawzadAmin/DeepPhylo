from typing import List, Sequence

import dgl
import torch
import torch.nn.functional as F
from ete3 import Tree

from ..spd_attn import compute_spd_nodes
from .BayesNJ.core import batch_init_NJ, batch_mask_from_ignored, batch_sample_Q
from .BayesNJ.evopf import EvoPF
from .graph_utils import ordered_candidate_pairs, ordered_exposed_nodes
from .trajectory import TrajectoryStep
from .tree_utils import _aldous_log_prior, _leaf_counts_from_n


def build_all_embeddings(model: EvoPF, leaf_embs: torch.Tensor, graph_or_levels):
    n_leaves = leaf_embs.shape[-1]
    n_nodes = 2 * n_leaves - 1
    device = leaf_embs.device

    if isinstance(graph_or_levels, list):
        levels = graph_or_levels
    else:
        # Fallback: compute levels from graph (e.g. during sampling)
        from .tree_levels import precompute_tree_levels
        levels = precompute_tree_levels(graph_or_levels)

    levels_gpu = [(n.to(device), l.to(device), r.to(device)) for n, l, r in levels]

    embs = torch.zeros(n_nodes, leaf_embs.shape[0], device=device, dtype=leaf_embs.dtype)
    embs[:n_leaves] = leaf_embs.transpose(0, 1)

    for node_ids, left_ids, right_ids in levels_gpu:
        embs[node_ids] = model.e_norm(model.embed_parent_node(embs[left_ids], embs[right_ids]))

    return embs


def compute_q_logits(model: EvoPF, embs_all: torch.Tensor) -> torch.Tensor:
    embs = embs_all.unsqueeze(0)
    ignored = torch.zeros(1, embs.shape[1], dtype=torch.bool, device=embs.device)
    Q = batch_sample_Q(model, embs, ignored)
    return Q.squeeze(0)


def bayesnj_sampling_logits(Q, ignored, temp=None):
    batch_size, _, _ = Q.shape
    mask_2d = batch_mask_from_ignored(ignored)
    mask_tril = (
        torch.ones_like(Q[0, :], dtype=torch.bool)
        .tril(diagonal=-1)
        .expand(batch_size, -1, -1)
    )
    mask = mask_2d & mask_tril

    Q_masked = Q.masked_fill(mask, torch.inf)
    if temp is not None:
        Q_masked = temp * Q_masked

    return (-Q_masked.view(batch_size, -1)).log_softmax(dim=-1).view(Q.shape)


class BayesNJStepScorer:
    def __init__(self, species_encoder):
        self.model = species_encoder.model
        self.vocab_size = species_encoder.vocab_size

    def encode_leaf_features(self, seqs: torch.Tensor, tree_to_n_leaves: torch.Tensor):
        if not isinstance(seqs, torch.Tensor):
            seqs = torch.stack(list(seqs))
        # Cast one_hot's int64 output to the model's float dtype (bf16 under
        # bf16-true) so the conv2d bias dtype matches.
        model_dtype = next(self.model.parameters()).dtype
        msa = (
            torch.nn.functional.one_hot(seqs, num_classes=self.vocab_size)
            .permute(0, 3, 2, 1)
            .to(model_dtype)
        )
        _, embs = self.model(msa)
        return [embs[b] for b in range(embs.shape[0])]

    def prepare_example(
        self,
        leaf_features: torch.Tensor,
        full_graph: dgl.DGLGraph | None = None,
        tree_levels=None,
    ):
        if tree_levels is None and full_graph is None:
            raise ValueError("BayesNJ training expects full_graph or tree_levels.")
        graph_or_levels = tree_levels if tree_levels is not None else full_graph
        embs_all = build_all_embeddings(self.model, leaf_features, graph_or_levels)
        Q = compute_q_logits(self.model, embs_all)
        return {"Q": Q}

    def score_step(
        self,
        step: TrajectoryStep,
        leaf_features: torch.Tensor,
        prepared_example=None,
        hamming_dist=None,
    ) -> torch.Tensor:
        if prepared_example is None:
            raise ValueError("BayesNJ training requires a prepared example.")
        Q = prepared_example["Q"]
        if step.candidate_pairs_global is None:
            raise ValueError("BayesNJ training requires candidate pairs for each step.")
        pairs_t = step.candidate_pairs_global.to(leaf_features.device)
        return -Q[pairs_t[:, 0], pairs_t[:, 1]]

    @torch.no_grad()
    def sample_tree(self, features: torch.Tensor, *, temperature=1.0, offset=None, offset_gamma=1.0):
        batch_size = features.shape[0]
        assert batch_size == 1
        b_idx = torch.arange(batch_size, device=features.device)
        msa = torch.nn.functional.one_hot(
            features, num_classes=self.vocab_size
        ).permute(0, 3, 2, 1)
        # one_hot is int64; cast to the model's float dtype (bf16 under
        # bf16-true precision, fp32 otherwise).
        model_dtype = next(self.model.parameters()).dtype
        _, embs = self.model(msa.to(model_dtype))

        _, _, _, ignored_nodes, n_leaves, node_count = batch_init_NJ(embs, rooted=True)
        n_merges = n_leaves - 1
        embs = F.pad(
            embs, (0, node_count - n_leaves), mode="constant", value=0.0
        ).transpose(-1, -2)

        num_nodes = embs.shape[1]
        nodes = {i: Tree(name=str(i)) for i in range(num_nodes)}

        # Track per-node leaf counts for Aldous bias
        node_leaf_counts = torch.ones(num_nodes, device=features.device)

        for iter_ in range(n_merges):
            Q = batch_sample_Q(self.model, embs, ignored_nodes)
            logits = bayesnj_sampling_logits(Q, ignored_nodes)

            flat_logits = logits.view(batch_size, -1)

            if offset == "aldous":
                lc = node_leaf_counts.float()
                bias_matrix = offset_gamma * (lc.unsqueeze(1) + lc.unsqueeze(0) - 2) + 1
                log_bias = torch.log(bias_matrix.clamp(min=1e-30))
                flat_logits = flat_logits + log_bias.unsqueeze(0).view(1, -1)

            if temperature <= 0:
                pos = flat_logits.argmax(dim=1)
            else:
                scaled_logits = (
                    flat_logits if temperature == 1.0 else flat_logits / temperature
                )
                pos = torch.distributions.Categorical(logits=scaled_logits).sample()
            h_i = (pos // num_nodes).item()
            h_j = (pos % num_nodes).item()
            parent = n_leaves + iter_

            node_leaf_counts[parent] = node_leaf_counts[h_i] + node_leaf_counts[h_j]

            if iter_ != n_merges - 1:
                embs_i = embs[b_idx, h_i]
                embs_j = embs[b_idx, h_j]
                e_u = self.model.embed_parent_node(embs_i, embs_j)
                e_u = self.model.e_norm(e_u)
                embs[b_idx, parent] = e_u

                ignored_nodes[b_idx, h_i] = True
                ignored_nodes[b_idx, h_j] = True
                ignored_nodes[b_idx, parent] = False

            nodes[parent].add_child(nodes[h_i])
            nodes[parent].add_child(nodes[h_j])

        root = n_leaves + n_merges - 1
        return nodes[root]


class AttentionStepScorer:
    def __init__(self, species_encoder, rate_predictor, reuse_full_graph_embeddings):
        self.species_encoder = species_encoder
        self.rate_predictor = rate_predictor
        self.reuse_full_graph_embeddings = reuse_full_graph_embeddings

    def encode_leaf_features(self, seqs: torch.Tensor, tree_to_n_leaves: torch.Tensor):
        flat_seqs = torch.cat([tree_seqs for tree_seqs in seqs], dim=0)
        pool = not getattr(self.rate_predictor, 'unpooled', False)
        leaf_features = self.species_encoder(flat_seqs, tree_to_n_leaves.cpu(), pool=pool)
        return list(torch.split(leaf_features, tree_to_n_leaves.tolist()))

    def prepare_example(
        self,
        leaf_features: torch.Tensor,
        full_graph=None,
        tree_levels=None,
    ):
        return None

    def score_steps_batched(
        self,
        steps: Sequence[TrajectoryStep],
        prepared_example,
    ) -> List[torch.Tensor]:
        if prepared_example is None or "full_graph_embs" not in prepared_example:
            raise ValueError(
                "Batched attention scoring requires cached full-graph embeddings."
            )

        full_graph_embs = prepared_example["full_graph_embs"]
        device = full_graph_embs.device
        batch_size = len(steps)
        hidden_size = full_graph_embs.shape[1]
        seq_lens = [step.obs_node_ids.shape[0] for step in steps]
        max_len = max(seq_lens)

        x = torch.zeros(
            batch_size,
            max_len,
            hidden_size,
            device=device,
            dtype=full_graph_embs.dtype,
        )
        spd = torch.full(
            (batch_size, max_len, max_len),
            -1,
            device=device,
            dtype=torch.long,
        )
        padding_mask = torch.ones(batch_size, max_len, device=device, dtype=torch.bool)
        t = torch.tensor(
            [step.t for step in steps], device=device, dtype=full_graph_embs.dtype
        )

        for i, step in enumerate(steps):
            if (
                step.obs_node_ids is None
                or step.exposed_nodes_local is None
                or step.spd is None
            ):
                raise ValueError(
                    "Attention training requires obs_node_ids, exposed_nodes_local, and spd."
                )
            n_nodes = step.obs_node_ids.shape[0]
            node_ids = step.obs_node_ids.to(device)
            x[i, :n_nodes] = full_graph_embs[node_ids]
            spd[i, :n_nodes, :n_nodes] = step.spd.to(device)
            padding_mask[i, :n_nodes] = False

        out = self.rate_predictor.attn_module.forward_batched(
            t, x, spd, padding_mask=padding_mask
        )

        logits = []
        for i, step in enumerate(steps):
            exposed_nodes = step.exposed_nodes_local.to(device)
            if exposed_nodes.numel() < 2:
                logits.append(torch.empty(0, device=device, dtype=out.dtype))
                continue
            exposed_out = out[i, exposed_nodes]
            tri_i, tri_j = torch.triu_indices(
                exposed_out.shape[0], exposed_out.shape[0], 1, device=device
            )
            pair_sum = exposed_out[tri_i] + exposed_out[tri_j]
            pair_prod = exposed_out[tri_i] * exposed_out[tri_j]
            pair_features = torch.cat([pair_sum, pair_prod], dim=-1)
            logits.append(self.rate_predictor.pair_readout(pair_features).squeeze(-1))
        return logits

    def score_step(
        self,
        step: TrajectoryStep,
        leaf_features: torch.Tensor,
        prepared_example=None,
        hamming_dist=None,
    ) -> torch.Tensor:
        device = leaf_features.device
        return self.rate_predictor(
            leaf_features, step.spd.to(device), hamming_dist,
            step.leaf_idx.to(device), step.basis, step.exposed_nodes_local.to(device),
            step.depths, step.counts,
        )

    @staticmethod
    def _init_sampling_graph(n_leaves, device):
        graph = dgl.graph(([], []), num_nodes=n_leaves, device=device)
        graph.ndata["n"] = torch.ones(n_leaves, 1, device=device)
        graph.ndata[dgl.NID] = torch.arange(n_leaves, device=device)
        return graph

    @staticmethod
    def _merge_pair(graph, left_idx, right_idx, parent_idx):
        graph.add_nodes(1)
        parent = graph.num_nodes() - 1
        if parent != parent_idx:
            raise RuntimeError(
                f"Expected parent idx {parent_idx}, but DGL created node {parent}"
            )

        graph.add_edges(
            torch.tensor([left_idx, right_idx], device=graph.device),
            torch.tensor([parent, parent], device=graph.device),
        )
        graph.ndata["n"][parent] = (
            graph.ndata["n"][left_idx] + graph.ndata["n"][right_idx] + 1
        )
        graph.ndata[dgl.NID] = torch.arange(graph.num_nodes(), device=graph.device)

    @torch.no_grad()
    def sample_tree(
        self, seqs: torch.Tensor, *,
        temperature=1.0, offset=None, offset_gamma=1.0,
        beam_size=1, corrector_steps=0, n_samples=1,
    ):
        if beam_size > 1:
            return self._sample_tree_beam(
                seqs, beam_size=beam_size, offset=offset, offset_gamma=offset_gamma,
            )
        if n_samples > 1:
            return self._sample_tree_best_of_n(
                seqs, n_samples=n_samples, temperature=temperature,
                offset=offset, offset_gamma=offset_gamma,
            )
        if corrector_steps > 0:
            return self._sample_tree_with_corrector(
                seqs, temperature=temperature, offset=offset,
                offset_gamma=offset_gamma, corrector_steps=corrector_steps,
            )

        from ..spd_attn import build_basis_from_graph
        from ..tree_utils import squaretrick_basis_from_dgl
        from ..top_down.graph_ops import compute_depths
        from .tree_utils import pairwise_hamming_frac

        device = next(self.species_encoder.parameters()).device
        seqs = seqs.to(device)
        n_leaves = seqs.shape[0]

        tree_to_n_leaves = torch.tensor([n_leaves], dtype=torch.long)
        pool = not getattr(self.rate_predictor, 'unpooled', False)
        leaf_features = self.species_encoder(seqs, tree_to_n_leaves.cpu(), pool=pool)
        hamming_dist = pairwise_hamming_frac(seqs.long())

        graph = self._init_sampling_graph(n_leaves, device)
        tree_nodes = {i: Tree(name=str(i)) for i in range(n_leaves)}
        next_parent = n_leaves

        while next_parent < 2 * n_leaves - 1:
            candidate_pairs = ordered_candidate_pairs(graph)
            if not candidate_pairs:
                break

            spd = compute_spd_nodes(graph)
            exposed_nodes = ordered_exposed_nodes(graph)
            leaf_idx = torch.where(graph.in_degrees().eq(0))[0]
            rooted = build_basis_from_graph(graph, leaf_idx)
            topo = squaretrick_basis_from_dgl(graph, leaf_idx).to(rooted.device)
            basis = torch.stack([rooted, topo], dim=-1)
            depths = compute_depths(graph).float()
            counts = graph.ndata["n"].squeeze(-1).float()

            logits = self.rate_predictor(
                leaf_features, spd, hamming_dist,
                leaf_idx, basis, exposed_nodes, depths, counts,
            )

            logits = logits.float()
            if offset == "aldous":
                leaf_counts = _leaf_counts_from_n(
                    graph.ndata["n"][exposed_nodes].squeeze(-1)
                )
                log_prior = _aldous_log_prior(leaf_counts, gamma=offset_gamma)
                logits = logits + log_prior.float()

            if temperature <= 0:
                selected_pair = logits.argmax()
            else:
                scaled_logits = logits if temperature == 1.0 else logits / temperature
                selected_pair = torch.distributions.Categorical(
                    logits=scaled_logits
                ).sample()
            left_idx, right_idx = candidate_pairs[selected_pair.item()]

            parent_idx = next_parent
            self._merge_pair(graph, left_idx, right_idx, parent_idx)

            tree_nodes[parent_idx] = Tree(name=str(parent_idx))
            tree_nodes[parent_idx].add_child(tree_nodes[left_idx])
            tree_nodes[parent_idx].add_child(tree_nodes[right_idx])
            next_parent += 1

        root_idx = next_parent - 1
        return tree_nodes[root_idx]

    def _beam_step_eval(
        self, graph, leaf_features, hamming_dist, offset, offset_gamma,
    ):
        """Score one beam: returns (log_probs, candidate_pairs) or (None, None) if done."""
        from ..spd_attn import build_basis_from_graph
        from ..tree_utils import squaretrick_basis_from_dgl
        from ..top_down.graph_ops import compute_depths

        candidate_pairs = ordered_candidate_pairs(graph)
        if not candidate_pairs:
            return None, None

        spd = compute_spd_nodes(graph)
        exposed_nodes = ordered_exposed_nodes(graph)
        leaf_idx = torch.where(graph.in_degrees().eq(0))[0]
        rooted = build_basis_from_graph(graph, leaf_idx)
        topo = squaretrick_basis_from_dgl(graph, leaf_idx).to(rooted.device)
        basis = torch.stack([rooted, topo], dim=-1)
        depths = compute_depths(graph).float()
        counts = graph.ndata["n"].squeeze(-1).float()

        logits = self.rate_predictor(
            leaf_features, spd, hamming_dist,
            leaf_idx, basis, exposed_nodes, depths, counts,
        )

        if offset == "aldous":
            leaf_counts = _leaf_counts_from_n(
                graph.ndata["n"][exposed_nodes].squeeze(-1)
            )
            log_prior = _aldous_log_prior(leaf_counts, gamma=offset_gamma)
            logits = logits + log_prior.to(logits.dtype)

        log_probs = torch.log_softmax(logits.float(), dim=0)
        return log_probs, candidate_pairs

    @torch.no_grad()
    def _sample_tree_beam(self, seqs, *, beam_size, offset, offset_gamma):
        from .tree_utils import pairwise_hamming_frac

        device = next(self.species_encoder.parameters()).device
        seqs = seqs.to(device)
        n_leaves = seqs.shape[0]

        tree_to_n_leaves = torch.tensor([n_leaves], dtype=torch.long)
        pool = not getattr(self.rate_predictor, "unpooled", False)
        leaf_features = self.species_encoder(seqs, tree_to_n_leaves.cpu(), pool=pool)
        hamming_dist = pairwise_hamming_frac(seqs.long())

        g0 = self._init_sampling_graph(n_leaves, device)
        beams = [{
            "graph": g0,
            "merges": [],   # list of (left_idx, right_idx, parent_idx)
            "next_parent": n_leaves,
            "log_prob": 0.0,
            "done": False,
        }]

        while True:
            any_active = False
            candidates = []  # (parent_idx, pair_idx, new_log_prob)
            for parent_idx, b in enumerate(beams):
                if b["done"]:
                    continue
                log_probs, pairs = self._beam_step_eval(
                    b["graph"], leaf_features, hamming_dist, offset, offset_gamma,
                )
                if log_probs is None:
                    b["done"] = True
                    continue
                any_active = True
                k_take = min(beam_size, len(pairs))
                topk_lp, topk_idx = log_probs.topk(k_take)
                for lp_v, idx_v in zip(topk_lp.tolist(), topk_idx.tolist()):
                    candidates.append((parent_idx, idx_v, b["log_prob"] + lp_v))
                b["_pairs"] = pairs

            if not any_active:
                break

            scored = [(b["log_prob"], "done", i) for i, b in enumerate(beams) if b["done"]]
            scored += [(c[2], "new", c) for c in candidates]
            scored.sort(key=lambda x: x[0], reverse=True)
            scored = scored[:beam_size]

            new_beams = []
            for _, kind, payload in scored:
                if kind == "done":
                    new_beams.append(beams[payload])
                    continue
                pi, pair_idx, new_lp = payload
                parent = beams[pi]
                left_idx, right_idx = parent["_pairs"][pair_idx]
                new_graph = parent["graph"].clone()
                np_idx = parent["next_parent"]
                self._merge_pair(new_graph, left_idx, right_idx, np_idx)
                new_beams.append({
                    "graph": new_graph,
                    "merges": parent["merges"] + [(left_idx, right_idx, np_idx)],
                    "next_parent": np_idx + 1,
                    "log_prob": new_lp,
                    "done": False,
                })
            beams = new_beams

        best = max(beams, key=lambda b: b["log_prob"])
        nodes = {i: Tree(name=str(i)) for i in range(n_leaves)}
        for left_idx, right_idx, p_idx in best["merges"]:
            nodes[p_idx] = Tree(name=str(p_idx))
            nodes[p_idx].add_child(nodes[left_idx])
            nodes[p_idx].add_child(nodes[right_idx])
        root_idx = best["next_parent"] - 1
        return nodes[root_idx]

    @staticmethod
    def _random_undo_and_reindex(merges, K_target, n_leaves):
        """Undo K random merges; return (new_merges, n_undone).

        A merge is eligible to undo iff its parent_idx is not used as a child
        in any kept merge. Surviving merges are re-indexed so parent_idxs are
        contiguous starting at n_leaves.
        """
        n = len(merges)
        K = min(K_target, n)
        if K == 0:
            return list(merges), 0

        used_count = {}
        for l, r, _ in merges:
            used_count[l] = used_count.get(l, 0) + 1
            used_count[r] = used_count.get(r, 0) + 1

        kept = set(range(n))
        undone = 0
        while undone < K:
            eligible = [i for i in kept
                        if used_count.get(merges[i][2], 0) == 0]
            if not eligible:
                break
            pick = int(torch.randint(0, len(eligible), (1,)).item())
            i = eligible[pick]
            l, r, _ = merges[i]
            used_count[l] -= 1
            used_count[r] -= 1
            kept.remove(i)
            undone += 1

        if undone == 0:
            return list(merges), 0

        kept_sorted = sorted(kept)
        remap = {merges[old_idx][2]: n_leaves + new_pos
                 for new_pos, old_idx in enumerate(kept_sorted)}
        new_merges = []
        for new_pos, old_idx in enumerate(kept_sorted):
            l, r, _ = merges[old_idx]
            new_merges.append(
                (remap.get(l, l), remap.get(r, r), n_leaves + new_pos)
            )
        return new_merges, undone

    @torch.no_grad()
    def _sample_tree_with_corrector(
        self, seqs, *, temperature, offset, offset_gamma, corrector_steps,
    ):
        from .tree_utils import pairwise_hamming_frac

        device = next(self.species_encoder.parameters()).device
        seqs = seqs.to(device)
        n_leaves = seqs.shape[0]

        tree_to_n_leaves = torch.tensor([n_leaves], dtype=torch.long)
        pool = not getattr(self.rate_predictor, "unpooled", False)
        leaf_features = self.species_encoder(seqs, tree_to_n_leaves.cpu(), pool=pool)
        hamming_dist = pairwise_hamming_frac(seqs.long())

        graph = self._init_sampling_graph(n_leaves, device)
        merges = []
        next_parent = n_leaves

        def _step(g, m_list, np_, T):
            log_probs, pairs = self._beam_step_eval(
                g, leaf_features, hamming_dist, offset, offset_gamma,
            )
            if log_probs is None:
                return g, m_list, np_, False
            if T <= 0:
                idx = log_probs.argmax().item()
            else:
                idx = torch.distributions.Categorical(
                    logits=log_probs / T
                ).sample().item()
            l, r = pairs[idx]
            self._merge_pair(g, l, r, np_)
            return g, m_list + [(l, r, np_)], np_ + 1, True

        while next_parent < 2 * n_leaves - 1:
            graph, merges, next_parent, ok = _step(
                graph, merges, next_parent, temperature,
            )
            if not ok:
                break

            if corrector_steps > 0 and len(merges) > 0:
                new_merges, n_undone = self._random_undo_and_reindex(
                    merges, corrector_steps, n_leaves,
                )
                if n_undone > 0:
                    graph = self._init_sampling_graph(n_leaves, device)
                    for l, r, p in new_merges:
                        self._merge_pair(graph, l, r, p)
                    merges = new_merges
                    next_parent = n_leaves + len(merges)
                    for _ in range(n_undone):
                        graph, merges, next_parent, ok = _step(
                            graph, merges, next_parent, 0.0,
                        )
                        if not ok:
                            break

        nodes = {i: Tree(name=str(i)) for i in range(n_leaves)}
        for l, r, p in merges:
            nodes[p] = Tree(name=str(p))
            nodes[p].add_child(nodes[l])
            nodes[p].add_child(nodes[r])
        return nodes[next_parent - 1]

    def _sample_one_with_logprob(
        self, leaf_features, hamming_dist, n_leaves, device,
        *, temperature, offset, offset_gamma,
    ):
        """Sample one tree; return (tree, sum of T=1 log-probs of chosen actions)."""
        graph = self._init_sampling_graph(n_leaves, device)
        tree_nodes = {i: Tree(name=str(i)) for i in range(n_leaves)}
        next_parent = n_leaves
        total_log_prob = 0.0

        while next_parent < 2 * n_leaves - 1:
            log_probs, pairs = self._beam_step_eval(
                graph, leaf_features, hamming_dist, offset, offset_gamma,
            )
            if log_probs is None:
                break
            if temperature <= 0:
                idx = log_probs.argmax().item()
            else:
                idx = torch.distributions.Categorical(
                    logits=log_probs / temperature
                ).sample().item()
            total_log_prob += log_probs[idx].item()
            l, r = pairs[idx]
            self._merge_pair(graph, l, r, next_parent)
            tree_nodes[next_parent] = Tree(name=str(next_parent))
            tree_nodes[next_parent].add_child(tree_nodes[l])
            tree_nodes[next_parent].add_child(tree_nodes[r])
            next_parent += 1

        return tree_nodes[next_parent - 1], total_log_prob

    @torch.no_grad()
    def _sample_tree_best_of_n(
        self, seqs, *, n_samples, temperature, offset, offset_gamma,
    ):
        from .tree_utils import pairwise_hamming_frac

        device = next(self.species_encoder.parameters()).device
        seqs = seqs.to(device)
        n_leaves = seqs.shape[0]

        tree_to_n_leaves = torch.tensor([n_leaves], dtype=torch.long)
        pool = not getattr(self.rate_predictor, "unpooled", False)
        leaf_features = self.species_encoder(seqs, tree_to_n_leaves.cpu(), pool=pool)
        hamming_dist = pairwise_hamming_frac(seqs.long())

        best_tree, best_lp = None, float("-inf")
        for _ in range(n_samples):
            tree, lp = self._sample_one_with_logprob(
                leaf_features, hamming_dist, n_leaves, device,
                temperature=temperature, offset=offset, offset_gamma=offset_gamma,
            )
            if lp > best_lp:
                best_tree, best_lp = tree, lp
        return best_tree
