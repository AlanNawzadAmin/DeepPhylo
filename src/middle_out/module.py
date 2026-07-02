"""The :class:`UnifiedMiddleOutModule` LightningModule.

Split out of the former monolithic ``unified.py``."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..model import PhylogeneticTreeDiffusionModel
from ..bottom_up.tree_utils import pairwise_hamming_frac
from ..sampling_utils import ensure_newick_semicolon
from ..tree_utils import squaretrick_basis_from_edge_index
from .rooted_features import compute_rooted_features
from .data import MiddleOutStepExample, MiddleOutTrainExample, MiddleOutTrainStep, canonical_unroot, compute_node_leaf_masks, edge_index_to_spd, ensure_numeric_node_names, remap_tree_ids, resolve_builder_edge_child, tree_to_edge_index
from .sampling import _SamplingTreeBuilder, augmented_tree_to_rooted_tree


class UnifiedMiddleOutModule(PhylogeneticTreeDiffusionModel):
    def __init__(
        self,
        n_max,
        lr,
        rb,
        full_traj,
        attn_head,
        model_cfg,
        species_encoder,
        rate_predictor,
        ema_decay=0.9998,
        weight_decay=1e-5,
        warmup_steps=1000,
        mup_mha_fix=None,
        mup_optimizer=True,
        logit_norm_penalty=1e-4,
    ):
        super().__init__(lr=lr, ema_decay=ema_decay, weight_decay=weight_decay,
                         warmup_steps=warmup_steps, mup_mha_fix=mup_mha_fix,
                         mup_optimizer=mup_optimizer)
        self.save_hyperparameters(ignore=["species_encoder", "rate_predictor"])
        self.n_max = n_max + 1
        self.lr = lr
        self.rb = bool(rb)
        self.full_traj = bool(full_traj)
        self.attn_head = bool(attn_head)
        self.logit_norm_penalty = float(logit_norm_penalty)
        self.optimizer_cfg = model_cfg.optimizer
        self.model_cfg = model_cfg
        self.species_encoder = species_encoder
        self.rate_predictor = rate_predictor
        self.feature_proj = nn.Linear(
            model_cfg.transformer.hidden_dim, model_cfg.transformer.hidden_dim
        )
        # Non-zero init so root is distinguishable from mean(leaves) at step 0.
        self.root_token_offset = nn.Parameter(
            torch.randn(model_cfg.transformer.hidden_dim)
        )

    def encode_leaf_features(self, seqs: torch.Tensor, step=None, hamming_dist=None) -> torch.Tensor:
        tree_to_n_leaves = torch.tensor([seqs.shape[0]], device=seqs.device)
        pool = not getattr(self.rate_predictor, 'unpooled', False)
        # Pass step + hamming_dist to the species encoder. EvoPF accepts via
        # **kwargs and ignores; SPDAttnSpeciesEncoder uses them to mirror
        # score_step's tree-extended attention pass.
        encoded = self.species_encoder(
            seqs, tree_to_n_leaves, pool=pool,
            step=step, hamming_dist=hamming_dist,
        )
        leaf_features = self.feature_proj(encoded)
        if leaf_features.dim() == 3:
            # Unpooled: [n_leaves, L, h] — pool root feature over both leaves and positions
            root_feature = leaf_features.mean(dim=(0, 1)) + self.root_token_offset  # [h]
            return torch.cat([leaf_features, root_feature.unsqueeze(0).unsqueeze(0).expand(1, leaf_features.shape[1], -1)], dim=0)
        root_feature = leaf_features.mean(dim=0) + self.root_token_offset
        return torch.cat([leaf_features, root_feature.unsqueeze(0)], dim=0)

    def compute_step_loss(
        self,
        step: MiddleOutTrainStep,
        seqs: torch.Tensor,
        *,
        use_rb: bool | None = None,
        hamming_dist=None,
    ) -> torch.Tensor:
        leaf_features = self.encode_leaf_features(seqs, step=step, hamming_dist=hamming_dist)
        logits = self.rate_predictor.score_step(step, leaf_features, hamming_dist=hamming_dist)
        use_dense_targets = self.rb if use_rb is None else use_rb
        if use_dense_targets:
            targets = step.dense_target_edges.to(logits.device)
        else:
            targets = torch.as_tensor([step.sparse_target_edge], device=logits.device)
        if targets.ndim == 0:
            targets = targets.unsqueeze(0)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.shape[0] != targets.shape[0]:
            logits = logits[[-1]]
        # fp32 for the loss reduction so bf16 underflow on small log-probs
        # doesn't corrupt the gradient.
        logits_f = logits.float()
        log_probs = logits_f - torch.logsumexp(logits_f, dim=-1, keepdim=True)
        nll = -torch.gather(log_probs, dim=1, index=targets[:, None]).mean()
        if self.logit_norm_penalty > 0:
            nll = nll + self.logit_norm_penalty * logits_f.pow(2).mean()
        return nll

    def compute_example_loss(
        self,
        example: MiddleOutTrainExample,
        seqs: torch.Tensor,
        hamming_dist=None,
    ):
        if self.full_traj:
            if example.steps is None:
                raise ValueError("Expected full trajectory training payload")
            steps = example.steps
        else:
            if example.step is None:
                raise ValueError("Expected single-step training payload")
            steps = [example.step]
        step_losses = torch.stack(
            [self.compute_step_loss(step, seqs, hamming_dist=hamming_dist) for step in steps]
        )
        loss = step_losses.sum() * example.is_weight if self.full_traj else step_losses[0] * example.is_weight
        return loss, loss

    def _compute_batch_losses(self, batch):
        losses = []
        merge_losses = []
        for example in batch:
            seqs = example.seqs.to(self.device)
            hamming_dist = example.hamming_leaf.to(self.device)
            total_loss, merge_loss = self.compute_example_loss(example, seqs, hamming_dist=hamming_dist)
            losses.append(total_loss)
            merge_losses.append(merge_loss)
        return torch.stack(losses), torch.stack(merge_losses)

    def training_step(self, batch, batch_idx):
        losses, merge_losses = self._compute_batch_losses(batch)
        loss = losses.mean()
        self.log("train_loss", loss, prog_bar=True, batch_size=len(batch))
        self.log("train_merge_loss", merge_losses.mean(), batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        losses, merge_losses = self._compute_batch_losses(batch)
        loss = losses.mean()
        self.log("val_loss", loss, prog_bar=True, batch_size=len(batch), sync_dist=True)
        self.log("val_merge_loss", merge_losses.mean(), batch_size=len(batch), sync_dist=True)
        return loss

    @torch.no_grad()
    def sample_tree_from_seqs(
        self,
        seqs: torch.Tensor,
        temperature: float = 1.0,
        seed_triplet: list[int] | None = None,
        insertion_order: list[int] | None = None,
    ):
        seqs = seqs.to(self.device)
        # leaf_features is now built per-step (so the species encoder can use the
        # current tree state). Hamming over leaves is constant across steps.
        hamming_dist = pairwise_hamming_frac(seqs.long()).to(self.device)
        root_token_id = seqs.shape[0]
        n_leaves = seqs.shape[0] + 1

        if seed_triplet is None:
            # Root token always in seed triplet — pick 2 random real leaves to join it.
            real_perm = torch.randperm(n_leaves - 1, device=self.device)[:2].tolist()
            seed_triplet = real_perm + [root_token_id]
        elif len(seed_triplet) != 3:
            raise ValueError("seed_triplet must contain exactly 3 token ids")
        elif root_token_id not in seed_triplet:
            raise ValueError("root token must be in the seed triplet")

        seed_set = set(seed_triplet)
        missing = [leaf_id for leaf_id in range(n_leaves) if leaf_id not in seed_set]
        if insertion_order is not None:
            if sorted(insertion_order) != sorted(missing):
                raise ValueError("insertion_order must match the missing token set")
            remaining_insertion_order = list(insertion_order)
        else:
            remaining_insertion_order = None

        builder = _SamplingTreeBuilder()
        builder.init_triplet(seed_triplet, n_leaves - 1)

        while missing:
            current_tree = builder.root.copy()
            remaining = sorted(int(x.name) for x in current_tree.get_leaves())
            current_tree, _ = canonical_unroot(current_tree, sorted(remaining)[:3])
            current_tree = ensure_numeric_node_names(current_tree)
            node_masks = compute_node_leaf_masks(current_tree)
            old_id_to_mask = {
                int(node.name): node_masks[node]
                for node in current_tree.traverse()
                if not node.is_root()
            }
            current_tree, _, internal_map = remap_tree_ids(current_tree)
            edge_index = torch.as_tensor(
                tree_to_edge_index(current_tree), dtype=torch.long
            )
            root_token_local = remaining.index(root_token_id)
            depths, subtree_counts, local_basis = compute_rooted_features(
                edge_index, root_token_local, len(remaining)
            )
            node_basis = torch.zeros(edge_index.shape[0], n_leaves, dtype=local_basis.dtype)
            node_basis[:, remaining] = local_basis
            local_topo = squaretrick_basis_from_edge_index(edge_index, len(remaining))
            node_basis_topo = torch.zeros(edge_index.shape[0], n_leaves, dtype=local_topo.dtype)
            node_basis_topo[:, remaining] = local_topo
            step = MiddleOutStepExample(
                remaining_leaves=remaining,
                missing_leaves=torch.as_tensor(missing, dtype=torch.long),
                edge_index=edge_index,
                node_basis=node_basis,
                spd=edge_index_to_spd(edge_index),
                edge_children=torch.arange(edge_index.shape[0] - 1, dtype=torch.long),
                edge_parents=edge_index[:-1, 0].clone(),
                sparse_target_edge=0,
                dense_target_edges=torch.zeros(len(missing), dtype=torch.long),
                time=len(remaining) + 1,
                depths=depths,
                subtree_counts=subtree_counts,
                node_basis_topo=node_basis_topo,
            )
            leaf_features = self.encode_leaf_features(seqs, step=step, hamming_dist=hamming_dist)
            logits = self.rate_predictor.score_step(
                step, leaf_features, hamming_dist=hamming_dist
            )  # [n_missing, n_edges]

            n_edges = logits.shape[1]
            if remaining_insertion_order is not None:
                leaf_id = remaining_insertion_order.pop(0)
                missing_idx = missing.index(leaf_id)
                edge_logits = logits[missing_idx]
                if temperature <= 0:
                    sampled_edge = edge_logits.argmax().item()
                else:
                    sampled_edge = torch.distributions.Categorical(
                        logits=edge_logits / temperature
                    ).sample().item()
            else:
                # Row-normalize so each missing leaf contributes equal mass at
                # T=1: training only normalizes per-row (logsumexp over edges),
                # so cross-row scale is uncalibrated — flat-sampling raw logits
                # would marginalize over an untrained P(missing_leaf). After
                # this, each row sums to 1 in prob space, so flat sampling at
                # T=1 is uniform over missing leaves with the trained per-leaf
                # edge distribution. fp32 cast also avoids bf16 underflow.
                logits_f = logits.float()
                normalized = logits_f - torch.logsumexp(logits_f, dim=-1, keepdim=True)
                flat_logits = normalized.flatten()
                if temperature <= 0:
                    selected = flat_logits.argmax().item()
                else:
                    selected = torch.distributions.Categorical(
                        logits=flat_logits / temperature
                    ).sample().item()
                missing_idx = selected // n_edges
                sampled_edge = selected % n_edges
                leaf_id = missing[missing_idx]
            missing.pop(missing_idx)
            sampled_child = step.edge_children[sampled_edge].item()

            if sampled_child < len(remaining):
                global_child = remaining[sampled_child]
            else:
                inverse_internal_map = {v: k for k, v in internal_map.items()}
                global_child = inverse_internal_map[sampled_child]
                if global_child not in builder.name_dict:
                    global_child = resolve_builder_edge_child(
                        builder.root.copy(),
                        remaining,
                        old_id_to_mask[global_child],
                    )
            builder.add_leaf(global_child, leaf_id)

        return augmented_tree_to_rooted_tree(builder.root, root_token_id)

    @torch.no_grad()
    def sample_from_batch(
        self,
        batch,
        *,
        batch_index=None,
        datamodule=None,
        temperature=1.0,
        **kwargs,
    ):
        seqs, trees = batch
        sampled_tree = self.sample_tree_from_seqs(seqs[0], temperature=temperature)
        return {
            "true_tree_newick": ensure_newick_semicolon(trees[0].write(format=9)),
            "sampled_tree_newick": ensure_newick_semicolon(
                sampled_tree.write(format=9)
            ),
        }
