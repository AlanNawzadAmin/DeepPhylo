import torch

from ..model import PhylogeneticTreeDiffusionModel
from .inference import TreeSamplingMixin
from .percolation import LeafConstraint, PercolationDirection
from .sampling import compute_tau, compute_weight


class TopDownDiffusionModel(TreeSamplingMixin, PhylogeneticTreeDiffusionModel):
    def __init__(
        self,
        lr,
        species_encoder,
        rate_predictor,
        alpha=1,
        offset=None,
        rb=True,
        ema_decay=0.9998,
        weight_decay=1e-5,
        warmup_steps=1000,
        mup_mha_fix=None,
        mup_optimizer=True,
    ):
        super().__init__(lr, ema_decay=ema_decay, weight_decay=weight_decay,
                         warmup_steps=warmup_steps, mup_mha_fix=mup_mha_fix,
                         mup_optimizer=mup_optimizer)
        self.save_hyperparameters(ignore=["species_encoder", "rate_predictor"])
        self.species_encoder = species_encoder
        self.rate_predictor = rate_predictor
        self.alpha = alpha
        self.offset = offset.lower() if isinstance(offset, str) else offset
        self.rb = rb

    def _compute_tau(self, t):
        return compute_tau(t, self.alpha)

    def _compute_weight(self, t):
        return compute_weight(t, self.alpha)

    def get_metrics(self, batch, batch_idx=None, timer=False):
        (
            obs_graph,
            spd,
            t,
            t_depth,
            t_layer,
            target,
            T,
            host_idx_in_obs,
            tree_to_n_leaves,
            seqs,
            constraints,
            no_observed_children,
            n_obs_per_graph,
            left_child,
            right_child,
            importance_weight,
            obs_basis,
            hamming_leaf,
            pda_log_prior,
        ) = batch

        eligible_mask = ~torch.isin(
            constraints,
            torch.tensor(
                [LeafConstraint.FINAL.value, LeafConstraint.WAITING.value],
                device=constraints.device,
            ),
        )
        # assert eligible_mask.any(), "No eligible samples for loss computation"

        if not (eligible_mask & (constraints == LeafConstraint.FREE.value)).any():
            # No eligible FREE decision in this microbatch. Return a loss that is
            # exactly zero but CONNECTED to every trainable parameter (× 0.0) so
            # each one still receives a zero gradient. Under DDP with
            # find_unused_parameters=False a bare torch.zeros(()) here touches no
            # params, so this rank's gradient-reduction buckets never fire while the
            # other rank (normal path) waits → NCCL collective timeout. Connecting
            # to all params keeps both ranks in sync. (Triggered by the
            # importance_sample=false schedule, which can produce degenerate states
            # with no eligible FREE leaf.)
            zero_loss = sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0
            return {
                "loss": zero_loss,
                "t": t / T,
                "t_depth": t_depth,
                "t_layer": t_layer,
            }

        pool = not self.rate_predictor.unpooled
        leaf_features = self.species_encoder(seqs, tree_to_n_leaves, pool=pool)

        logits = self.rate_predictor(
            leaf_features,
            spd,
            tree_to_n_leaves.tolist(),
            host_idx_in_obs,
            n_obs_per_graph.tolist(),
            obs_graph.ndata["d"],
            obs_graph.ndata["subtree_count"],
            left_child,
            right_child,
            hamming_leaf=hamming_leaf,
            obs_basis=obs_basis,
        )
        if self.offset == "pda":
            logits = logits + pda_log_prior

        # fp32 for the log_softmax / NLL — bf16 has too narrow a range to
        # represent very small probabilities reliably.
        log_probs = torch.log_softmax(logits.float(), dim=1)
        loglik = torch.where(
            target == PercolationDirection.LEFT.value, log_probs[:, 0], log_probs[:, 1]
        )
        nll = torch.where(constraints == LeafConstraint.FREE.value, -loglik, 0 * loglik)

        if self.rb:
            loss = nll[eligible_mask].mean() * importance_weight
        else:
            idx = torch.multinomial(eligible_mask.float(), 1)
            loss = nll[idx] * importance_weight

        return {
            "loss": loss,
            "t": t / T,
            "t_depth": t_depth,
            "t_layer": t_layer,
        }

    def training_step(self, batch, batch_idx):
        metrics = self.get_metrics(batch, batch_idx)
        self.log("train_loss", metrics["loss"], prog_bar=True)
        self.log("t", metrics["t"])
        self.log("t_depth", metrics["t_depth"])
        self.log("t_layer", metrics["t_layer"])
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        metrics = self.get_metrics(batch, batch_idx)
        self.log("val_loss", metrics["loss"], prog_bar=True, sync_dist=True)
        return metrics["loss"]
