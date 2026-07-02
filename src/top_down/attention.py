import math

import torch
import torch.nn as nn

from ..spd_attn import SPDBiasedAttention


def build_attn_inputs(
    leaf_features, spd, n_leaves_per_tree, host_idx_in_obs,
    depths, subtree_counts, hamming_leaf, obs_basis, pos_type_emb,
    node_seed_per_position=False, basis_weights=None, seed_nodes=True,
):
    """Assemble the [N, L, h] node set + biases consumed by SPDBiasedAttention.

    Shared by every top down rate-predictor head so the trunk sees identical
    inputs regardless of how the final activations are read out. Pure (no module
    state beyond ``pos_type_emb``).

    ``node_seed_per_position`` controls the observed-node seed embedding:
      * False (default, parity with prior checkpoints): collapse each leaf to its
        position-mean, average over the subtree, broadcast back over all L columns
        — a position-CONSTANT seed (legacy from the pooled-encoder days).
      * True: per-position weighted average of subtree leaves (same formula the
        refresh step uses), keeping the per-column structure. Preferred for heads
        that read per-position activations (e.g. the covariance head), since a
        position-constant seed is removed by position-centering.

    ``basis_weights`` (only used when ``node_seed_per_position`` is True): an
    optional learnable [2]-vector mixing the two ``obs_basis`` channels
    (rooted, squaretrick) before contracting with the leaf features. None →
    use the rooted channel alone.

    ``seed_nodes`` (default True): ablation toggle for the leaf-averaging node
    seed. When False, internal nodes start at zero embeddings (regardless of
    ``node_seed_per_position``/``basis_weights``) so the only topology that
    enters them is the structural depth/count seed + per-layer refresh inside
    the trunk — i.e. the basis-averaged init is removed.

    Returns: all_embs [N, L, h], spd_unbatched [N, N], hamming [N, N],
    leaf_idx [n_leaves], full_basis [N, n_leaves, 2], full_depths [N],
    full_counts [N], plus the ints n_leaves, n_obs, L.
    """
    assert leaf_features.dim() == 3, "Only unpooled leaf_features supported"
    device = leaf_features.device
    n_leaves = sum(n_leaves_per_tree)
    # obs_basis is [n_obs, n_leaves, 2]: ch0 rooted, ch1 squaretrick.
    assert obs_basis.dim() == 3 and obs_basis.shape[-1] == 2, \
        "obs_basis must be [n_obs, n_leaves, 2] (rooted + squaretrick channels)"
    n_obs = obs_basis.shape[0]
    N = n_leaves + n_obs
    L = leaf_features.shape[1]

    obs_basis = obs_basis.to(device)

    leaf_embs = leaf_features + pos_type_emb(
        torch.ones(n_leaves, device=device, dtype=torch.long)
    ).unsqueeze(1)

    # Seed node embeddings from the subtree-membership basis.
    if not seed_nodes:
        # Ablation: skip the leaf-averaging node seed entirely. Internal nodes
        # start at zero and pick up topology only via the structural seed +
        # refresh inside the trunk.
        node_embs = leaf_features.new_zeros(n_obs, L, leaf_features.shape[-1])
    elif node_seed_per_position:
        # Per-position weighted average of subtree leaves (matches the refresh
        # step); keeps MSA-column structure instead of collapsing it.
        if basis_weights is not None:
            # Learnable mix of the rooted + squaretrick basis channels.
            # Row-normalize each channel to sum 1 first, so the seed is a true
            # per-position MEAN of subtree leaves. The training-time rooted channel
            # is L2-normalized (build_basis_from_graph), which otherwise makes node
            # norm grow ~sqrt(subtree size) under any leaf common-mode; sum-to-1
            # restores subtree-size invariance and matches the sampling-time basis.
            b = obs_basis / obs_basis.sum(dim=1, keepdim=True)
            node_embs = torch.einsum(
                "nli,i,lph->nph", b, basis_weights.to(b.dtype), leaf_features
            )  # [n_obs, L, h]
        else:
            node_embs = torch.einsum("nl,lph->nph", obs_basis[..., 0], leaf_features)  # [n_obs, L, h]
    else:
        # Legacy position-CONSTANT seed (parity with prior pooled behavior).
        node_embs = obs_basis[..., 0] @ leaf_features.mean(dim=1)
        node_embs = node_embs.unsqueeze(1).expand(-1, L, -1)

    all_embs = torch.cat([leaf_embs, node_embs], dim=0)  # [N, L, h]

    # Expand basis/depths/counts to [N, ...]: identity rows (both channels) for leaves.
    eye = torch.eye(n_leaves, device=device)
    full_basis = torch.zeros(N, n_leaves, 2, device=device)
    full_basis[:n_leaves, :, 0] = eye
    full_basis[:n_leaves, :, 1] = eye
    full_basis[n_leaves:] = obs_basis
    full_depths = torch.zeros(N, device=device)
    full_depths[n_leaves:] = depths.float()
    full_counts = torch.ones(N, device=device)
    full_counts[n_leaves:] = subtree_counts.float()

    # Expand hamming from [n_leaves, n_leaves] to [N, N]
    spd_unbatched = spd[0] if spd.dim() == 3 else spd
    hamming_leaf = hamming_leaf.to(device)
    hamming = torch.full((N, N), -1.0, device=device, dtype=hamming_leaf.dtype)
    hamming[:n_leaves, :n_leaves] = hamming_leaf
    hamming.fill_diagonal_(-1)

    leaf_idx = torch.arange(n_leaves, device=device)

    return (all_embs, spd_unbatched, hamming, leaf_idx,
            full_basis, full_depths, full_counts, n_leaves, n_obs, L)


class TopDownAttention(nn.Module):
    def __init__(self, h_size, n_layers, n_heads, max_spd, unpooled=False,
                 pair_dim=256, pair_inner_dim=32, use_layernorm=True,
                 depth_mup=False, **kwargs):
        super().__init__()
        assert unpooled, "Only unpooled mode is supported"
        self.h_size = h_size
        self.unpooled = True
        self.pos_type_emb = nn.Embedding(2, h_size)

        self.attn_module = SPDBiasedAttention(
            h_size, n_layers, n_heads, max_spd,
            pair_dim=pair_dim, pair_inner_dim=pair_inner_dim,
            use_layernorm=use_layernorm, depth_mup=depth_mup,
            seed_struct=True,
        )

        self.pair_readout = nn.Sequential(
            nn.Linear(h_size * 2, h_size),
            nn.GELU(),
            nn.Linear(h_size, 1),
        )
        # µP readout: extra 1/sqrt(D) on init (σ ~ 1/D), bias zeroed.
        with torch.no_grad():
            self.pair_readout[-1].weight.mul_(1.0 / math.sqrt(h_size))
            if self.pair_readout[-1].bias is not None:
                self.pair_readout[-1].bias.zero_()
        self.empty_emb = nn.Parameter(torch.randn(h_size))

    def _compute_output_logits(self, leaf_embs, node_embs, leaf_to_left, leaf_to_right):
        node_embs_padded = torch.cat([node_embs, self.empty_emb.unsqueeze(0)], dim=0)
        left_embs = node_embs_padded[leaf_to_left]
        right_embs = node_embs_padded[leaf_to_right]

        left_sum = leaf_embs + left_embs
        left_prod = leaf_embs * left_embs
        left_logits = self.pair_readout(torch.cat([left_sum, left_prod], dim=-1)).squeeze(-1)

        right_sum = leaf_embs + right_embs
        right_prod = leaf_embs * right_embs
        right_logits = self.pair_readout(torch.cat([right_sum, right_prod], dim=-1)).squeeze(-1)

        return torch.stack([left_logits, right_logits], dim=-1)

    def forward(
        self,
        leaf_features,
        spd,
        n_leaves_per_tree,
        host_idx_in_obs,
        n_obs_per_graph,
        depths,
        subtree_counts,
        left_child,
        right_child,
        timer=False,
        hamming_leaf=None,
        obs_basis=None,
    ):
        (all_embs, spd_unbatched, hamming, leaf_idx,
         full_basis, full_depths, full_counts, n_leaves, n_obs, L) = build_attn_inputs(
            leaf_features, spd, n_leaves_per_tree, host_idx_in_obs,
            depths, subtree_counts, hamming_leaf, obs_basis, self.pos_type_emb,
        )

        # Node-type tag (row 0) for the structural node seed inside the trunk;
        # leaves already carry row 1 (added in build_attn_inputs).
        node_type = self.pos_type_emb(
            torch.zeros(1, device=all_embs.device, dtype=torch.long)
        ).squeeze(0)  # [h]
        out = self.attn_module(
            all_embs, spd_unbatched, hamming,
            leaf_idx, full_basis, full_depths, full_counts, node_type,
        )

        leaf_embs = out[:n_leaves]
        node_embs = out[n_leaves:]

        leaf_to_left = left_child[host_idx_in_obs]
        leaf_to_right = right_child[host_idx_in_obs]
        return self._compute_output_logits(
            leaf_embs, node_embs, leaf_to_left, leaf_to_right
        )
