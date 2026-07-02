from typing import Sequence

import torch
from torch.optim.lr_scheduler import LambdaLR

from ..model import PhylogeneticTreeDiffusionModel
from ..sampling_utils import ensure_newick_semicolon
from .scorers import AttentionStepScorer, BayesNJStepScorer
from .trajectory import TrajectoryExample, TrajectoryStep
from .tree_utils import _aldous_log_prior, compute_step_kl


class UnifiedBottomUpModule(PhylogeneticTreeDiffusionModel):
    def __init__(
        self,
        *,
        lr,
        rb,
        attn_head,
        full_traj,
        weight_decay=1e-5,
        species_encoder=None,
        rate_predictor=None,
        offset=None,
        offset_gamma=1.0,
        ema_decay=0.9998,
        warmup_steps=1000,
        mup_mha_fix=None,
        mup_optimizer=True,
    ):
        super().__init__(lr, ema_decay=ema_decay, weight_decay=weight_decay,
                         warmup_steps=warmup_steps, mup_mha_fix=mup_mha_fix,
                         mup_optimizer=mup_optimizer)
        self.save_hyperparameters(ignore=["species_encoder", "rate_predictor"])
        self.rb = rb
        self.attn_head = attn_head
        self.full_traj = full_traj
        self.offset = offset.lower() if isinstance(offset, str) else offset
        self.offset_gamma = offset_gamma
        self.species_encoder = species_encoder

        if self.attn_head:
            self.rate_predictor = rate_predictor
            self._merge_scorer = AttentionStepScorer(
                species_encoder,
                rate_predictor,
                reuse_full_graph_embeddings=full_traj,
            )
        else:
            self.vocab_size = species_encoder.vocab_size
            self._merge_scorer = BayesNJStepScorer(species_encoder)

    def select_steps(self, example: TrajectoryExample) -> Sequence[TrajectoryStep]:
        if self.full_traj or example.preselected_steps:
            return example.steps
        step_idx = torch.randint(example.T, (1,)).item()
        return [example.steps[step_idx]]

    def _apply_offset(self, logits: torch.Tensor, step: TrajectoryStep) -> torch.Tensor:
        if self.offset is None:
            return logits
        if self.offset == "aldous":
            if step.exposed_leaf_counts is None:
                return logits
            lc = step.exposed_leaf_counts.to(logits.device)
            log_prior = _aldous_log_prior(lc, gamma=self.offset_gamma)
            return logits + log_prior.to(logits.dtype)
        raise ValueError(
            f"Unsupported bottom up offset '{self.offset}'. "
            "Supported: 'aldous' or None."
        )

    def compute_step_loss(self, logits: torch.Tensor, target_probs: torch.Tensor):
        target_probs = target_probs.to(logits.device)
        loss = compute_step_kl(logits, target_probs)
        trivial_loss = compute_step_kl(torch.zeros_like(logits), target_probs)
        return loss, trivial_loss

    def compute_example_losses(
        self,
        example: TrajectoryExample,
        leaf_features: torch.Tensor,
        *,
        selected_steps=None,
        hamming_dist=None,
    ):
        selected_steps = (
            self.select_steps(example) if selected_steps is None else selected_steps
        )
        prepared_example = self._merge_scorer.prepare_example(
            leaf_features, example.full_graph,
            tree_levels=getattr(example, 'tree_levels', None),
        )
        if (
            self.attn_head
            and prepared_example is not None
            and "full_graph_embs" in prepared_example
            and len(selected_steps) > 1
            and not getattr(self._merge_scorer.rate_predictor, 'unpooled', False)
        ):
            step_logits = self._merge_scorer.score_steps_batched(
                selected_steps, prepared_example
            )
        else:
            step_logits = [
                self._merge_scorer.score_step(
                    step, leaf_features, prepared_example=prepared_example,
                    hamming_dist=hamming_dist,
                )
                for step in selected_steps
            ]
        step_losses = []
        step_trivial_losses = []
        t_values = []
        for step, logits in zip(selected_steps, step_logits):
            logits = self._apply_offset(logits, step)
            loss, trivial_loss = self.compute_step_loss(logits, step.target_probs)
            step_losses.append(loss)
            step_trivial_losses.append(trivial_loss)
            t_values.append(step.t)

        loss = torch.stack(step_losses).sum()
        trivial_loss = torch.stack(step_trivial_losses).sum()
        if not self.rb:
            step_entropy = sum(s.rb_entropy for s in selected_steps)
            loss = loss - step_entropy
            trivial_loss = trivial_loss - step_entropy
        loss = loss * example.n_is_weight
        trivial_loss = trivial_loss * example.n_is_weight

        mean_t = loss.new_tensor(t_values).mean()
        return loss, trivial_loss, mean_t

    def _compute_batch_losses(self, batch):
        examples = batch
        seqs = [example.leaf_features_inputs for example in examples]
        tree_to_n_leaves = torch.tensor(
            [tree_seqs.shape[0] for tree_seqs in seqs], dtype=torch.long
        )
        leaf_features_list = self._merge_scorer.encode_leaf_features(
            seqs, tree_to_n_leaves
        )

        losses = []
        trivial_losses = []
        t_values = []
        for example, leaf_features in zip(examples, leaf_features_list):
            hamming_dist = (
                example.hamming_leaf.to(leaf_features.device)
                if example.hamming_leaf is not None
                else None
            )
            loss, trivial_loss, mean_t = self.compute_example_losses(
                example, leaf_features, hamming_dist=hamming_dist,
            )
            losses.append(loss)
            trivial_losses.append(trivial_loss)
            t_values.append(mean_t)

        return torch.stack(losses), torch.stack(trivial_losses), torch.stack(t_values)

    def training_step(self, batch, batch_idx):
        losses, trivial_losses, t_values = self._compute_batch_losses(batch)
        self.log("train_loss", losses.mean(), prog_bar=True)
        self.log("trivial_loss", trivial_losses.mean(), prog_bar=True)
        self.log("t", t_values.mean())
        return losses.mean()

    def validation_step(self, batch, batch_idx):
        losses, _, _ = self._compute_batch_losses(batch)
        self.log("val_loss", losses.mean(), prog_bar=True, sync_dist=True)
        return losses.mean()

    @torch.no_grad()
    def sample_from_batch(
        self, batch, *, batch_index=None, datamodule=None,
        temperature=1.0, beam_size=1, corrector_steps=0, n_samples=1, **kwargs,
    ):
        seqs, trees = batch
        assert seqs.shape[0] == 1, "Batch size > 1 not supported"

        extra = {}
        if beam_size > 1:
            extra["beam_size"] = beam_size
        if corrector_steps > 0:
            extra["corrector_steps"] = corrector_steps
        if n_samples > 1:
            extra["n_samples"] = n_samples
        if self.attn_head:
            sampled_tree = self._merge_scorer.sample_tree(
                seqs[0], temperature=temperature,
                offset=self.offset, offset_gamma=self.offset_gamma,
                **extra,
            )
        else:
            sampled_tree = self._merge_scorer.sample_tree(
                seqs, temperature=temperature,
                offset=self.offset, offset_gamma=self.offset_gamma,
                **extra,
            )

        return {
            "true_tree_newick": ensure_newick_semicolon(trees[0].write(format=9)),
            "sampled_tree_newick": ensure_newick_semicolon(
                sampled_tree.write(format=9)
            ),
        }

    def configure_optimizers(self):
        if self.attn_head:
            return super().configure_optimizers()

        q_params = list(self.species_encoder.model.q_embedder.parameters())
        q_param_ids = {id(p) for p in q_params}
        other_params = [
            p
            for p in self.species_encoder.model.parameters()
            if id(p) not in q_param_ids
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": self.lr},
                {"params": q_params, "lr": self.lr},
            ],
            weight_decay=self.weight_decay,
        )

        warmup_steps = 1_000
        final_step = 86_000

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if current_step < final_step:
                return float(final_step - current_step) / float(
                    max(1, final_step - warmup_steps)
                )
            return 0.0

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
