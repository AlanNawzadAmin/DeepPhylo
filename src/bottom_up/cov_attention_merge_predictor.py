import math

import torch
import torch.nn as nn

from ..spd_attn import SPDBiasedAttention


class CovAttentionMergePredictor(nn.Module):
    """Covariance-readout merge head for bottom up.

    Same SPD-biased attention trunk as :class:`AttentionMergePredictor`, but
    instead of pooling each node to a single vector and reading out a bilinear
    (sum/prod) pair score, this head keeps the *full* per-position activations
    ``h[node, position, i]`` and scores a candidate merge pair (exposed nodes
    A, B) by the position-averaged, position-centered per-channel cross-
    covariance::

        c_i(A, B) = (1 / L) * Σ_p ĥ[A, p, i] · ĥ[B, p, i],   ĥ = h − mean_p h

    The covariance is symmetric in (A, B), so only the upper-triangle candidate
    pairs are scored. The resulting hidden_dim vector ``c(A, B)`` is fed through
    an MLP to a scalar logit. Pairs are read out in ``triu_indices(M, M, 1)``
    order over the exposed nodes — identical to :class:`AttentionMergePredictor`
    — so the training targets line up unchanged.

    Optionally, before the readout, attention runs over the covariance grid:
    each candidate pair's covariance vector is a token in a shared sequence, and
    EvoPF's symmetric :class:`PairAttention` + :class:`Transition` (both
    residual) let one pair's score depend on the others. This is the bottom up
    analogue of the top down covariance head's "quartic" attention, restricted
    to the upper-triangle candidate cells. Default on.
    """

    def __init__(self, h_size, n_layers, n_heads, max_spd, unpooled=False,
                 pair_dim=256, pair_inner_dim=32, use_layernorm=True,
                 depth_mup=False, cov_layernorm=False,
                 use_cov_attention=True, cov_attn_layers=2,
                 use_second_basis=True, use_pair=True, use_refresh=True,
                 use_hamming=True, use_cls=True, use_spd=True,
                 use_node_seed=True, **kwargs):
        super().__init__()
        assert unpooled, "Only unpooled mode is supported"
        self.h_size = h_size
        self.unpooled = True
        # Ablation toggle: when False, zero the squaretrick / unrooted basis
        # channel so only the rooted channel feeds the node seed + refresh.
        self.use_second_basis = use_second_basis
        # Ablation toggle: when False, internal nodes are NOT seeded from the
        # basis-averaged subtree leaf features (they start at zero and acquire
        # topology only via the structural seed + refresh). basis_weights is then
        # unused, so it's only created when the seed is on (DDP errors on unused).
        self.use_node_seed = use_node_seed
        self.pos_type_emb = nn.Embedding(2, h_size)
        # Leaf/node type tag. Shrink to ~unit norm so the shared row-1 vector is a
        # small perturbation, not a dominant rank-1 common-mode (see cov head).
        nn.init.normal_(self.pos_type_emb.weight, std=1.0 / math.sqrt(h_size))
        # Learnable mix of the two basis channels (rooted, squaretrick) used to
        # build the per-position node seed; init 1/2 each. Only built when the
        # node seed is on — otherwise it would be an unused parameter (DDP errors).
        if use_node_seed:
            self.basis_weights = nn.Parameter(torch.full((2,), 0.5))
        # Normalize the species-encoder leaf features to ~unit RMS before they
        # enter the trunk (parity with the top down cov head): scale-only,
        # per-token, so it fixes magnitude without changing leaf-leaf cosine
        # structure or the node seeds (which are means of these).
        self.leaf_norm = nn.RMSNorm(h_size)

        # pool_output=False → the trunk returns the unpooled [N, L(+1), h] tensor
        # so we can read out per-position activations. seed_struct=True → the
        # structural (depth+count+node-type) seed is injected into non-leaf nodes.
        self.attn_module = SPDBiasedAttention(
            h_size, n_layers, n_heads, max_spd,
            pair_dim=pair_dim, pair_inner_dim=pair_inner_dim,
            use_layernorm=use_layernorm, depth_mup=depth_mup,
            use_pair=use_pair, use_refresh=use_refresh,
            use_hamming=use_hamming, use_cls=use_cls,
            use_spd=use_spd,
            pool_output=False, seed_struct=True,
        )

        # Optional affine norm of the covariance vector before the MLP. Off by
        # default: LayerNorm would wash out the *magnitude* of c(A, B) (how
        # strongly A aligns with B), which is itself signal for the logit.
        self.cov_norm = nn.LayerNorm(h_size) if cov_layernorm else nn.Identity()
        self.cov_readout = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.GELU(),
            nn.Linear(h_size, 1),
        )
        # µP readout: extra 1/sqrt(D) on init (σ ~ 1/D), bias zeroed.
        with torch.no_grad():
            self.cov_readout[-1].weight.mul_(1.0 / math.sqrt(h_size))
            if self.cov_readout[-1].bias is not None:
                self.cov_readout[-1].bias.zero_()

        # Optional attention over the upper-triangle covariance vectors before
        # readout: each candidate pair is a token, EvoPF's symmetric PairAttention
        # (gated MHA, µP K-scale) + a Transition FFN, both residual. Lets one
        # pair's score depend on the other candidate pairs. Default on.
        self.use_cov_attention = use_cov_attention
        if use_cov_attention:
            from .BayesNJ.evopf import PairAttention, Transition
            self.cov_attn = nn.ModuleList(
                [PairAttention(h_size, n_heads, symmetric=True, mup=True)
                 for _ in range(cov_attn_layers)]
            )
            self.cov_attn_trans = nn.ModuleList(
                [Transition(h_size) for _ in range(cov_attn_layers)]
            )

    def forward(self, leaf_features, spd, hamming_dist,
                leaf_idx, basis, exposed_nodes, depths, counts):
        device = leaf_features.device
        basis = basis.to(device)
        # basis is [N, n_leaves, 2]: ch0 rooted subtree, ch1 squaretrick absorption.
        assert basis.dim() == 3 and basis.shape[-1] == 2, \
            "basis must be [N, n_leaves, 2]"
        assert leaf_features.dim() == 3, "Only unpooled leaf_features supported"

        if not self.use_second_basis:
            basis = basis.clone()
            basis[..., 1] = 0.0

        # ~unit-RMS leaf features (also sets the scale of the node seeds, which
        # are means of these).
        leaf_features = self.leaf_norm(leaf_features)
        N = basis.shape[0]
        L = leaf_features.shape[1]
        h = self.h_size
        leaf_idx = leaf_idx.to(device)

        # Seed all node embeddings from the subtree-membership basis (a
        # per-position weighted mean of subtree leaves), then overwrite the leaf
        # rows with their raw features + leaf tag. Internal nodes additionally get
        # the structural depth/count/node-type seed inside the trunk.
        if self.use_node_seed:
            # Row-normalize each channel to sum 1 first, so the seed is a true
            # per-position MEAN of subtree leaves (the rooted channel is
            # L2-normalized at build time; sum-to-1 restores subtree-size
            # invariance and matches the sampling-time basis).
            b = basis / basis.sum(dim=1, keepdim=True).clamp(min=1e-9)
            all_embs = torch.einsum(
                "nli,i,lph->nph", b, self.basis_weights.to(b.dtype), leaf_features
            )  # [N, L, h]
        else:
            # Ablation: internal nodes start at zero; topology enters only via the
            # structural seed + refresh inside the trunk.
            all_embs = leaf_features.new_zeros(N, L, h)

        leaf_tag = self.pos_type_emb(
            torch.ones(1, device=device, dtype=torch.long)
        ).squeeze(0)  # [h]
        all_embs = all_embs.index_copy(
            0, leaf_idx, (leaf_features + leaf_tag).to(all_embs.dtype)
        )

        # Expand the leaf-only Hamming distances to the full node set (-1 = mask).
        # hamming_dist may be None when the trunk's use_hamming ablation is off;
        # an all-mask matrix is then never read inside the trunk.
        dtype = hamming_dist.dtype if hamming_dist is not None else all_embs.dtype
        full_hamming = torch.full((N, N), -1.0, device=device, dtype=dtype)
        if hamming_dist is not None:
            full_hamming[leaf_idx.unsqueeze(1), leaf_idx.unsqueeze(0)] = hamming_dist.to(device)
            full_hamming.fill_diagonal_(-1)

        # Node-type tag (row 0) for the structural node seed inside the trunk;
        # leaves already carry row 1 above.
        node_type = self.pos_type_emb(
            torch.zeros(1, device=device, dtype=torch.long)
        ).squeeze(0)  # [h]
        out = self.attn_module(
            all_embs, spd, full_hamming,
            leaf_idx, basis, depths.to(device), counts.to(device), node_type,
        )

        if exposed_nodes.numel() < 2:
            return torch.empty(0, device=device, dtype=out.dtype)

        # pool_output=False → out is [N, L(+1), h]; drop the CLS column (position
        # L, present when use_cls=True) so we average only over the L real MSA
        # positions, then center each node's activations across positions.
        acts = out[:, :L, :]
        acts = acts - acts.mean(dim=1, keepdim=True)

        exposed_nodes = exposed_nodes.to(device)
        exp = acts[exposed_nodes]  # [M, L, h]
        M = exp.shape[0]

        # Per-channel cross-covariance of every (exposed-node A, exposed-node B)
        # pair: C[A, B, i] = (1/L) Σ_p exp[A, p, i] · exp[B, p, i]. Symmetric in
        # (A, B); only the upper triangle holds candidate pairs.
        C = torch.einsum("apc,bpc->abc", exp, exp) / L  # [M, M, h]
        i, j = torch.triu_indices(M, M, 1, device=device)
        pair_cov = C[i, j]  # [n_pairs, h]

        if self.use_cov_attention:
            x = pair_cov.unsqueeze(0)  # [1, n_pairs, h]
            for attn, trans in zip(self.cov_attn, self.cov_attn_trans):
                x = x + attn(x)
                x = x + trans(x)
            pair_cov = x.squeeze(0)

        return self.cov_readout(self.cov_norm(pair_cov)).squeeze(-1)
