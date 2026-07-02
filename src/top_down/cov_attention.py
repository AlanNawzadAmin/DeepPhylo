import math

import torch
import torch.nn as nn

from ..spd_attn import SPDBiasedAttention
from .attention import build_attn_inputs


class TopDownCovAttention(nn.Module):
    """Covariance-readout head for the top down top-down model.

    Same SPD-biased attention trunk as :class:`TopDownAttention`, but instead of
    pooling each node to a single CLS vector and reading out a bilinear (sum/prod)
    pair score, this head keeps the *full* per-position activations
    ``h[node, position, i]`` and scores a (species X, candidate node N) pair by the
    position-averaged, position-centered per-channel cross-covariance::

        c_i(X, N) = (1 / L) * Σ_p ĥ[N, p, i] · ĥ[X, p, i],   ĥ = h − mean_p h

    The resulting hidden_dim vector ``c(X, N)`` is fed through an MLP to a scalar
    logit. For each species we score its host's left and right children and stack
    the two logits into ``[n_leaves, 2]`` — the same interface the rate predictor
    exposes to the model (which then adds the PDA offset).

    Empty children (the chosen branch doesn't exist yet) have no activations, so
    their covariance vector is modeled from the *occupied sibling*::

        c_empty(X) = −scale · c(X, sibling) + empty_vec

    ``empty_sibling_scale`` is a single learnable scalar initialized to 1.0, so at
    init this is exactly "negate the sibling's covariance, then add a learned empty
    vector". It can learn toward 0 to fall back to a pure learned bias if the
    complement prior turns out to hurt. When *both* children are empty (a fresh
    host, e.g. SYMMETRIC leaves during sampling) there is no sibling, so both
    directions just get ``empty_vec`` — symmetric logits, as desired.
    """

    def __init__(self, h_size, n_layers, n_heads, max_spd, unpooled=False,
                 pair_dim=256, pair_inner_dim=32, use_layernorm=True,
                 depth_mup=False, cov_layernorm=False, empty_sibling_init=1.0,
                 use_cov_attention=True, cov_attn_layers=2,
                 use_second_basis=True, use_pair=True, use_refresh=True,
                 use_hamming=True, use_cls=True, use_spd=True,
                 use_node_seed=True, op_mean_mup=True, **kwargs):
        super().__init__()
        assert unpooled, "Only unpooled mode is supported"
        self.h_size = h_size
        self.unpooled = True
        # Ablation toggle: when False, zero obs_basis channel 1 (the squaretrick /
        # unrooted basis) so only the rooted channel feeds the node seed + refresh.
        self.use_second_basis = use_second_basis
        # Ablation toggle: when False, internal nodes are NOT seeded from the
        # basis-averaged subtree leaf features (they start at zero and acquire
        # topology only via the structural seed + refresh). basis_weights is then
        # unused, so it's only created when the seed is on (DDP errors on unused).
        self.use_node_seed = use_node_seed
        self.pos_type_emb = nn.Embedding(2, h_size)
        # Leaf/node type tag. Default nn.Embedding init is N(0,1) → a shared
        # vector of norm ~sqrt(h) added to EVERY leaf at EVERY position, i.e. a
        # dominant rank-1 common-mode that makes leaves look alike at the SPD
        # input (and corrupts what the attention layers compute downstream).
        # Shrink to ~unit norm so it's a small perturbation, not the signal.
        nn.init.normal_(self.pos_type_emb.weight, std=1.0 / math.sqrt(h_size))
        # Learnable mix of the two obs_basis channels (rooted, squaretrick) used
        # to build the per-position node seed; init 1/2 each. Only built when the
        # node seed is on — otherwise it would be an unused parameter (DDP errors).
        if use_node_seed:
            self.basis_weights = nn.Parameter(torch.full((2,), 0.5))
        # Normalize the species-encoder leaf features to ~unit RMS before they
        # enter the trunk. The species_encoder Linear contracts (~/sqrt(3)) so they
        # otherwise arrive at ~0.4 RMS. RMSNorm is per-token scale-only (no
        # mean-subtraction, no cross-token mixing), so it fixes magnitude WITHOUT
        # changing leaf-leaf cosine structure.
        self.leaf_norm = nn.RMSNorm(h_size)

        # pool_output=False → the trunk returns the unpooled [N, L(+1), h] tensor
        # so we can read out per-position activations.
        self.attn_module = SPDBiasedAttention(
            h_size, n_layers, n_heads, max_spd,
            pair_dim=pair_dim, pair_inner_dim=pair_inner_dim,
            use_layernorm=use_layernorm, depth_mup=depth_mup,
            use_pair=use_pair, use_refresh=use_refresh,
            use_hamming=use_hamming, use_cls=use_cls,
            use_spd=use_spd, op_mean_mup=op_mean_mup,
            pool_output=False, seed_struct=True,
        )

        # Optional affine norm of the covariance vector before the MLP. Off by
        # default: LayerNorm would wash out the *magnitude* of c(X, N) (how
        # strongly X aligns with N), which is itself signal for the logit.
        self.cov_norm = nn.LayerNorm(h_size) if cov_layernorm else nn.Identity()
        self.cov_readout = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.GELU(),
            nn.Linear(h_size, 1),
        )
        # µP readout: extra 1/sqrt(D) on init (σ ~ 1/D), bias zeroed (matches
        # TopDownAttention.pair_readout).
        with torch.no_grad():
            self.cov_readout[-1].weight.mul_(1.0 / math.sqrt(h_size))
            if self.cov_readout[-1].bias is not None:
                self.cov_readout[-1].bias.zero_()

        self.empty_vec = nn.Parameter(torch.zeros(h_size))
        self.empty_sibling_scale = nn.Parameter(torch.tensor(float(empty_sibling_init)))

        # Optional attention over the [n_leaves, n_obs, h] covariance grid before
        # readout: each layer is axial node-attention (reusing EvoPF's quartic
        # PairAttention design) + an EvoPF Transition FFN, both residual. Lets a
        # (species, candidate) score depend on the species' covariance with the
        # other candidate nodes. Default on.
        self.use_cov_attention = use_cov_attention
        if use_cov_attention:
            from ..bottom_up.BayesNJ.evopf import PairAttention, Transition
            # Full ("quartic") self-attention over every (species, node) cell as a
            # token: EvoPF's PairAttention in symmetric mode runs MHA over the whole
            # flat token sequence (gated, µP K-scale), + a Transition FFN.
            self.cov_attn = nn.ModuleList(
                [PairAttention(h_size, n_heads, symmetric=True, mup=True)
                 for _ in range(cov_attn_layers)]
            )
            self.cov_attn_trans = nn.ModuleList(
                [Transition(h_size) for _ in range(cov_attn_layers)]
            )

    def _cov_logits(self, leaf_acts, node_acts, host_idx_in_obs, left_child, right_child, L):
        """leaf_acts [n_leaves, L, h], node_acts [n_obs, L, h] (position-centered).
        host_idx_in_obs [n_leaves] is each species' host node; left_child/right_child
        [n_obs] are the observed child node ids (-1 = empty). Returns logits
        [n_leaves, 2] = (left, right) for each species' host."""
        n_leaves, _, h = leaf_acts.shape
        n_obs = node_acts.shape[0]
        device = leaf_acts.device
        rows = torch.arange(n_leaves, device=device)

        # Per-channel cross-covariance of every (leaf, observed-node) pair:
        #   C[X, N, i] = (1/L) Σ_p leaf_acts[X, p, i] · node_acts[N, p, i]
        C = torch.einsum("xpi,npi->xni", leaf_acts, node_acts) / L  # [n_leaves, n_obs, h]

        if not self.use_cov_attention:
            # Cheap path (no attention): empties handled at readout.
            leaf_to_left = left_child[host_idx_in_obs]
            leaf_to_right = right_child[host_idx_in_obs]
            has_left = leaf_to_left.ne(-1)
            has_right = leaf_to_right.ne(-1)
            lrows = has_left.nonzero(as_tuple=True)[0]
            rrows = has_right.nonzero(as_tuple=True)[0]
            # C can be bf16 under autocast (einsum) while leaf_acts stays fp32;
            # base the destinations on C so the index_put dtypes match.
            real_left = C.new_zeros(n_leaves, h)
            real_right = C.new_zeros(n_leaves, h)
            real_left[lrows] = C[lrows, leaf_to_left[lrows]]
            real_right[rrows] = C[rrows, leaf_to_right[rrows]]
            return self._readout(real_left, real_right, has_left, has_right)

        # Add a column for every EMPTY child slot, so "spawn a new branch" is a
        # first-class candidate the attention sees. An empty slot's value for
        # species X is −s·C[X, sibling] + empty_vec (just empty_vec when the host
        # has no observed sibling either). left/right are then picked purely by
        # gather index — no learned left/right convention.
        empty_left = (left_child == -1).nonzero(as_tuple=True)[0]    # nodes w/ empty left slot
        empty_right = (right_child == -1).nonzero(as_tuple=True)[0]
        slot_sib = torch.cat([right_child[empty_left], left_child[empty_right]])  # sibling id, -1 if none
        n_le = empty_left.numel()
        n_empty = slot_sib.numel()

        # augmented-column index per node's left/right child: the observed child id,
        # or the empty slot's column (n_obs + slot index). No -1 remains.
        left_col = left_child.clone()
        right_col = right_child.clone()
        left_col[empty_left] = n_obs + torch.arange(n_le, device=device)
        right_col[empty_right] = n_obs + n_le + torch.arange(empty_right.numel(), device=device)

        empty_cols = self.empty_vec.expand(n_leaves, n_empty, h).clone()  # [n_leaves, n_empty, h]
        sib = (slot_sib != -1).nonzero(as_tuple=True)[0]
        empty_cols[:, sib, :] = -self.empty_sibling_scale * C[:, slot_sib[sib], :] + self.empty_vec

        aug = torch.cat([C, empty_cols], dim=1)  # [n_leaves, n_cols, h]
        n_cols = aug.shape[1]
        # Quartic attention: flatten the WHOLE grid so every (species, node) cell
        # is its own token in one shared sequence, then full self-attention.
        x = aug.reshape(1, n_leaves * n_cols, h)
        for attn, trans in zip(self.cov_attn, self.cov_attn_trans):
            x = x + attn(x)
            x = x + trans(x)
        aug = x.reshape(n_leaves, n_cols, h)

        cov_left = aug[rows, left_col[host_idx_in_obs]]
        cov_right = aug[rows, right_col[host_idx_in_obs]]
        left_logit = self.cov_readout(self.cov_norm(cov_left)).squeeze(-1)
        right_logit = self.cov_readout(self.cov_norm(cov_right)).squeeze(-1)
        return torch.stack([left_logit, right_logit], dim=-1)

    def _readout(self, real_left, real_right, has_left, has_right):
        """Empty-child handling + MLP. real_{left,right} [n_leaves, h] are the
        (possibly attention-contextualized) covariance vectors for each host's
        children; has_{left,right} [n_leaves] mark which children actually exist."""
        empty = self.empty_vec
        s = self.empty_sibling_scale
        hl = has_left.unsqueeze(-1)
        hr = has_right.unsqueeze(-1)
        # left exists -> real; left empty & right exists -> −s·sibling + empty;
        # both empty -> empty. (real_right/real_left are only used where the
        # sibling actually exists, so the zero-placeholder rows are never read.)
        cov_left = torch.where(
            hl, real_left,
            torch.where(hr, -s * real_right + empty, empty.expand_as(real_left)),
        )
        cov_right = torch.where(
            hr, real_right,
            torch.where(hl, -s * real_left + empty, empty.expand_as(real_right)),
        )

        left_logit = self.cov_readout(self.cov_norm(cov_left)).squeeze(-1)
        right_logit = self.cov_readout(self.cov_norm(cov_right)).squeeze(-1)
        return torch.stack([left_logit, right_logit], dim=-1)

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
        if not self.use_second_basis:
            obs_basis = obs_basis.clone()
            obs_basis[..., 1] = 0.0
        # ~unit-RMS leaf features (also sets the scale of the node seeds, which
        # are means of these).
        leaf_features = self.leaf_norm(leaf_features)
        (all_embs, spd_unbatched, hamming, leaf_idx,
         full_basis, full_depths, full_counts, n_leaves, n_obs, L) = build_attn_inputs(
            leaf_features, spd, n_leaves_per_tree, host_idx_in_obs,
            depths, subtree_counts, hamming_leaf, obs_basis, self.pos_type_emb,
            # per-position node seed: a position-constant seed would be wiped out
            # by this head's position-centering, so keep the per-column structure.
            node_seed_per_position=True,
            basis_weights=self.basis_weights if self.use_node_seed else None,
            seed_nodes=self.use_node_seed,
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

        # pool_output=False → out is [N, L(+1), h]; drop the CLS column (position
        # L, present when the trunk's use_cls=True) so we average only over the L
        # real MSA positions, then center each node's activations across positions.
        acts = out[:, :L, :]
        acts = acts - acts.mean(dim=1, keepdim=True)
        leaf_acts = acts[:n_leaves]   # [n_leaves, L, h]
        node_acts = acts[n_leaves:]   # [n_obs,    L, h]

        return self._cov_logits(
            leaf_acts, node_acts, host_idx_in_obs, left_child, right_child, L
        )
