"""MiddleOut edge-scoring heads and the merge predictor.

Split out of the former monolithic ``unified.py``."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..spd_attn import SPDBiasedAttention
from .pfn_transformer import TransformerWithFeatures
from .data import MiddleOutStepExample, MiddleOutTrainStep


def build_node_embeddings_from_basis(
    node_basis: torch.Tensor, leaf_features: torch.Tensor,
) -> torch.Tensor:
    """Weighted average of leaf features per tree node.

    Args:
        node_basis: [n_nodes, n_total_leaves] — weights (sum to 1 per row for leaves in subtree).
        leaf_features: [n_total_leaves, (L,) h].

    Returns:
        [n_nodes, (L,) h] — node embeddings.
    """
    basis = node_basis.to(leaf_features.device)
    if leaf_features.dim() == 3:
        return torch.einsum('ij,jlh->ilh', basis, leaf_features)
    return basis @ leaf_features


def _extend_attn_inputs(node_embs, new_node_emb, spd, hamming_dist, leaf_idx,
                        leaf_pos_to_original):
    """Append the missing leaves to the tree as context tokens and extend the
    SPD / Hamming matrices to the augmented [N_ext, N_ext] set.

    ``new_node_emb`` may be pooled ``[M, h]`` (bilinear head — broadcast across
    the L MSA positions) or unpooled ``[M, L, h]`` (covariance head — used as-is
    so the species keep their per-position structure). Returns
    ``(all_embs, spd_ext, hamming_ext, n_tree, n_missing)``.
    """
    n_tree = node_embs.shape[0]
    if node_embs.dim() == 3:
        # Unpooled tree nodes [N_tree, L, h].
        L = node_embs.shape[1]
        if new_node_emb.dim() == 2:
            missing_ctx = new_node_emb.unsqueeze(1).expand(-1, L, -1)  # [M, L, h]
        else:
            missing_ctx = new_node_emb  # already [M, L, h]
    else:
        missing_ctx = new_node_emb  # [M, h]
    n_missing = missing_ctx.shape[0]
    all_embs = torch.cat([node_embs, missing_ctx], dim=0)  # [N_ext, (L,) h]

    # Extend SPD: missing leaves use -2 (not in tree), diagonal 0.
    N_ext = n_tree + n_missing
    spd_ext = torch.full((N_ext, N_ext), -2, device=spd.device, dtype=spd.dtype)
    spd_ext[:n_tree, :n_tree] = spd
    idx = torch.arange(n_tree, N_ext, device=spd.device)
    spd_ext[idx, idx] = 0

    # Extend hamming using leaf_pos_to_original so missing leaves get real
    # distances at their N_ext positions, not -1. Root token (id == n_real)
    # has no sequence so its entries stay at -1.
    hamming_ext = torch.full((N_ext, N_ext), -1, device=hamming_dist.device, dtype=hamming_dist.dtype)
    if leaf_pos_to_original is not None:
        n_real = hamming_dist.shape[0]
        is_real = leaf_pos_to_original < n_real
        safe_ids = torch.where(is_real, leaf_pos_to_original,
                               torch.zeros_like(leaf_pos_to_original))
        hamming_pairs = hamming_dist[safe_ids[:, None], safe_ids[None, :]]
        real_pairs = is_real[:, None] & is_real[None, :]
        hamming_pairs = torch.where(real_pairs, hamming_pairs,
                                    torch.full_like(hamming_pairs, -1))
        hamming_ext[leaf_idx[:, None], leaf_idx[None, :]] = hamming_pairs
    else:
        # Fallback (e.g. top down/bottom up paths that don't pass leaf_pos_to_original).
        nh = hamming_dist.shape[0]
        hamming_ext[:nh, :nh] = hamming_dist
    hamming_ext.fill_diagonal_(-1)

    return all_embs, spd_ext, hamming_ext, n_tree, n_missing


class SPDAttentionTreeModelWithFeatures(nn.Module):
    def __init__(self, n_max, cfg, attn_cfg):
        super().__init__()
        if attn_cfg is None:
            raise ValueError("attn_cfg must be provided when attn_head=True")
        self.hidden_dim = cfg.hidden_dim
        self.depth_dim = self.hidden_dim
        self.n_max = n_max
        assert getattr(attn_cfg, 'unpooled', False), "Only unpooled mode is supported"
        self.unpooled = True
        self.attn = SPDBiasedAttention(
            self.hidden_dim,
            attn_cfg.n_layers,
            attn_cfg.n_heads,
            attn_cfg.max_spd,
            use_refresh=getattr(attn_cfg, 'use_refresh', True),
            use_cls=getattr(attn_cfg, 'use_cls', True),
            use_pair=getattr(attn_cfg, 'use_pair', True),
            use_hamming=getattr(attn_cfg, 'use_hamming', True),
            use_rope=getattr(attn_cfg, 'use_rope', True),
        )
        # LN on the attention output before the bilinear/cubic head: bounds
        # u·v and queries·edge_feat so per-node magnitude drift can't compound
        # through the products and blow up the logits in bf16.
        self.score_in_ln = nn.LayerNorm(self.hidden_dim)
        self.use_norm_head = bool(getattr(attn_cfg, 'use_norm_head', False))
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        # Edge feature from cat([u*v, u+v]) → 2h → h, then pair_readout over
        # cat([query+edge, query*edge]) matching bottom up / top down.
        self.edge_proj = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.pair_readout = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        # µP readout: extra 1/sqrt(D) on init (σ ~ 1/D), bias zeroed.
        with torch.no_grad():
            self.pair_readout[-1].weight.mul_(1.0 / math.sqrt(self.hidden_dim))
            if self.pair_readout[-1].bias is not None:
                self.pair_readout[-1].bias.zero_()

    def forward(self, node_embs, spd, t, new_node_emb, u, v, hamming_dist,
                leaf_idx, basis, depths, subtree_counts, leaf_pos_to_original=None,
                root_position_in_tree=None):
        if new_node_emb.ndim == 1:
            new_node_emb = new_node_emb.unsqueeze(0)

        all_embs, spd_ext, hamming_ext, n_tree, n_missing = _extend_attn_inputs(
            node_embs, new_node_emb, spd, hamming_dist, leaf_idx,
            leaf_pos_to_original,
        )

        out = self.attn(
            all_embs, spd_ext, hamming_ext,
            leaf_idx, basis, depths.float(), subtree_counts.float(),
        )  # [N_ext, h]
        out = self.score_in_ln(out)
        node_embs = out[:n_tree]   # [n_tree, h]
        new_node_emb = out[n_tree:]       # [n_missing, h]

        # Edge scoring: pair_readout(cat([sum, prod])) over (query, edge_feat),
        # matching bottom up / top down flexibility.
        u_emb = node_embs[u]
        v_emb = node_embs[v]
        # u - v (child - parent) instead of u + v: removes the shared-mean component
        # so edge_feat differentiates across edges by the *direction* of the parent→child
        # difference, not just the (largely-symmetric) sum of endpoints.
        edge_feat = self.edge_proj(torch.cat([u_emb * v_emb, u_emb - v_emb], dim=-1))  # [E, h]
        if self.use_norm_head and root_position_in_tree is not None:
            # PairNorm-style rescaling, then re-anchor with root_emb instead of the
            # cross-leaf mean. Each query becomes: root_emb (tree context) + the
            # per-leaf deviation, rescaled so its norm matches the original input's
            # norm. At M=1, centered=0 → output = root_emb, which is the right thing
            # (the only signal at the last placement step is the tree itself).
            root_emb = node_embs[root_position_in_tree]
            mean_emb = new_node_emb.mean(dim=0, keepdim=True)
            centered = new_node_emb - mean_emb
            scale = new_node_emb.norm() / centered.norm().clamp(min=1e-2)
            queries_in = centered * scale + root_emb
        else:
            queries_in = new_node_emb
        queries = self.query_proj(queries_in)                                           # [M, h]
        qe_sum  = queries.unsqueeze(1) + edge_feat.unsqueeze(0)                         # [M, E, h]
        qe_prod = queries.unsqueeze(1) * edge_feat.unsqueeze(0)                         # [M, E, h]
        logits = self.pair_readout(torch.cat([qe_sum, qe_prod], dim=-1)).squeeze(-1)    # [M, E]
        return logits


class SPDAttentionCovTreeModel(nn.Module):
    """Covariance-readout edge scorer for MiddleOut (port of top down
    :class:`TopDownCovAttention`).

    Same SPD-biased trunk as :class:`SPDAttentionTreeModelWithFeatures`, but
    ``pool_output=False`` so we keep the per-position activations. We then score
    each (missing species X, tree edge e) pair by the position-averaged,
    position-centered per-channel cross-covariance of X with the edge's child and
    parent nodes::

        c_i(X, N) = (1/L) Σ_p ĥ[X, p, i] · ĥ[N, p, i],   ĥ = h − mean_p h

    An edge e = (child u_e, parent v_e); its score is read out from
    ``edge_proj(cat([c(X, u_e), c(X, v_e)]))``. Unlike top down there is no
    empty-child case — every MiddleOut edge has a real child and parent node.

    ``cov_edge_mode`` controls where the optional quartic PairAttention over the
    covariance grid runs relative to collapsing nodes → edges:

    - ``"node_attn_then_edge"`` (default): attend over the [species × node] grid
      (N nodes), then gather + combine to edges. Cheapest, mirrors top down.
    - ``"edge_then_attn"``: combine to the [species × edge] grid first (~2N
      tokens), then attend over edges, then read out.
    """

    def __init__(self, n_max, cfg, attn_cfg):
        super().__init__()
        if attn_cfg is None:
            raise ValueError("attn_cfg must be provided when cov_head=True")
        self.hidden_dim = cfg.hidden_dim
        self.n_max = n_max
        assert getattr(attn_cfg, 'unpooled', False), "Only unpooled mode is supported"
        self.unpooled = True
        self.cov_edge_mode = getattr(attn_cfg, 'cov_edge_mode', 'node_attn_then_edge')
        assert self.cov_edge_mode in ('node_attn_then_edge', 'edge_then_attn'), \
            f"unknown cov_edge_mode {self.cov_edge_mode!r}"

        # Same trunk flags as the bilinear head, but pool_output=False so we can
        # read the per-position activations for the covariance.
        self.attn = SPDBiasedAttention(
            self.hidden_dim,
            attn_cfg.n_layers,
            attn_cfg.n_heads,
            attn_cfg.max_spd,
            use_refresh=getattr(attn_cfg, 'use_refresh', True),
            use_cls=getattr(attn_cfg, 'use_cls', True),
            use_pair=getattr(attn_cfg, 'use_pair', True),
            use_hamming=getattr(attn_cfg, 'use_hamming', True),
            use_rope=getattr(attn_cfg, 'use_rope', True),
            use_spd=getattr(attn_cfg, 'use_spd', True),
            pool_output=False,
        )

        # Edge cov vector from cat([c(X,u), c(X,v)]) → h, then scalar readout.
        self.edge_proj = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        # Optional affine norm of the cov vector before readout. Off by default:
        # LayerNorm would wash out the *magnitude* of the alignment, itself signal.
        cov_layernorm = bool(getattr(attn_cfg, 'cov_layernorm', False))
        self.cov_norm = nn.LayerNorm(self.hidden_dim) if cov_layernorm else nn.Identity()
        self.cov_readout = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        # µP readout: extra 1/sqrt(D) on init (σ ~ 1/D), bias zeroed.
        with torch.no_grad():
            self.cov_readout[-1].weight.mul_(1.0 / math.sqrt(self.hidden_dim))
            if self.cov_readout[-1].bias is not None:
                self.cov_readout[-1].bias.zero_()

        # Optional quartic PairAttention over the covariance grid (every cell is a
        # token in one shared sequence), reusing EvoPF's symmetric PairAttention.
        self.use_cov_attention = bool(getattr(attn_cfg, 'use_cov_attention', True))
        if self.use_cov_attention:
            from ..bottom_up.BayesNJ.evopf import PairAttention, Transition
            cov_attn_layers = int(getattr(attn_cfg, 'cov_attn_layers', 2))
            self.cov_attn = nn.ModuleList(
                [PairAttention(self.hidden_dim, attn_cfg.n_heads, symmetric=True, mup=True)
                 for _ in range(cov_attn_layers)]
            )
            self.cov_attn_trans = nn.ModuleList(
                [Transition(self.hidden_dim) for _ in range(cov_attn_layers)]
            )

    def _grid_attn(self, x):
        """Quartic self-attention over a [M, K, h] cov grid (flatten to one
        token sequence). No-op when use_cov_attention is False."""
        if not self.use_cov_attention:
            return x
        M, K, h = x.shape
        x = x.reshape(1, M * K, h)
        for attn, trans in zip(self.cov_attn, self.cov_attn_trans):
            x = x + attn(x)
            x = x + trans(x)
        return x.reshape(M, K, h)

    def forward(self, node_embs, spd, t, new_node_emb, u, v, hamming_dist,
                leaf_idx, basis, depths, subtree_counts, leaf_pos_to_original=None,
                root_position_in_tree=None):
        assert node_embs.dim() == 3, "cov head requires unpooled node_embs [n_tree, L, h]"
        assert new_node_emb.dim() == 3, \
            "cov head requires unpooled missing-leaf features [n_missing, L, h]"
        L = node_embs.shape[1]

        all_embs, spd_ext, hamming_ext, n_tree, n_missing = _extend_attn_inputs(
            node_embs, new_node_emb, spd, hamming_dist, leaf_idx,
            leaf_pos_to_original,
        )

        out = self.attn(
            all_embs, spd_ext, hamming_ext,
            leaf_idx, basis, depths.float(), subtree_counts.float(),
        )  # [N_ext, L(+1), h] (pool_output=False; CLS column present if use_cls)

        # Drop the CLS column (position L) so we average only over the L real MSA
        # positions, then center each token's activations across positions.
        acts = out[:, :L, :]
        acts = acts - acts.mean(dim=1, keepdim=True)
        node_acts = acts[:n_tree]    # [n_tree,    L, h] — edge candidates
        leaf_acts = acts[n_tree:]    # [n_missing, L, h] — species to place

        # Per-channel cross-covariance of every (species, tree-node) pair:
        #   C[X, N, i] = (1/L) Σ_p leaf_acts[X, p, i] · node_acts[N, p, i]
        C = torch.einsum("xpi,npi->xni", leaf_acts, node_acts) / L  # [M, n_tree, h]

        if self.cov_edge_mode == 'node_attn_then_edge':
            C = self._grid_attn(C)                                  # [M, n_tree, h]
            cov_u = C[:, u]                                         # [M, E, h]
            cov_v = C[:, v]
            edge_cov = self.edge_proj(torch.cat([cov_u, cov_v], dim=-1))  # [M, E, h]
        else:  # edge_then_attn
            cov_u = C[:, u]
            cov_v = C[:, v]
            edge_cov = self.edge_proj(torch.cat([cov_u, cov_v], dim=-1))  # [M, E, h]
            edge_cov = self._grid_attn(edge_cov)

        logits = self.cov_readout(self.cov_norm(edge_cov)).squeeze(-1)  # [M, E]
        return logits


class MiddleOutMergePredictor(nn.Module):
    def __init__(self, n_max, cfg, h_size, attn_head=False, attn_cfg=None,
                 use_input_mlp=False, cov_head=False):
        super().__init__()
        self.n_max = n_max + 1
        self.h_size = h_size
        self.attn_head = bool(attn_head)
        # cov_head swaps the bilinear CLS readout for the covariance readout
        # (SPDAttentionCovTreeModel); it implies the attn trunk, so it only has
        # effect when attn_head is on.
        self.cov_head = bool(cov_head) and self.attn_head
        self.use_input_mlp = bool(use_input_mlp)
        self.unpooled = getattr(attn_cfg, 'unpooled', False) if attn_cfg else False
        if self.attn_head:
            if self.cov_head:
                self.tree_model = SPDAttentionCovTreeModel(
                    self.n_max, cfg, attn_cfg
                )
            else:
                self.tree_model = SPDAttentionTreeModelWithFeatures(
                    self.n_max, cfg, attn_cfg
                )
            # Old-style per-tree-node identity: cat([basis_padded(n_max), basis@leaf_feats])
            # → MLP → h. Gives each internal node an explicit subtree-membership code
            # alongside its content mean. Matches pre-April build_full_node_embeddings.
            if self.use_input_mlp:
                self.input_mlp = nn.Sequential(
                    nn.Linear(self.n_max + h_size, h_size),
                    nn.GELU(),
                    nn.Linear(h_size, h_size),
                    nn.LayerNorm(h_size),
                )
        else:
            self.tree_model = TransformerWithFeatures(self.n_max, cfg)

    def score_step(
        self, step: MiddleOutStepExample | MiddleOutTrainStep, leaf_features: torch.Tensor,
        hamming_dist=None,
    ):
        node_embs = build_node_embeddings_from_basis(
            step.node_basis, leaf_features,
        )
        new_node_emb = leaf_features[step.missing_leaves]
        # The cov head needs the species' per-position activations, so keep the
        # missing-leaf features unpooled [M, L, h]. The bilinear head uses a
        # pooled [M, h] query.
        if new_node_emb.dim() == 3 and not self.cov_head:
            new_node_emb = new_node_emb.mean(dim=1)
        edge_children = torch.arange(
            step.edge_parents.shape[0], device=leaf_features.device
        )
        if self.attn_head:
            if step.spd is None:
                raise ValueError("SPD is required when attn_head=True")
            device = leaf_features.device
            n_tree = node_embs.shape[0]
            basis = step.node_basis.to(device)  # [n_tree, n_total_leaves]

            # Per-tree-node identity MLP: cat(basis_padded, node_embs) → h.
            if self.use_input_mlp:
                n_total = leaf_features.shape[0]
                basis_pad = torch.full(
                    (n_tree, self.n_max), -1.0, device=device, dtype=node_embs.dtype
                )
                basis_pad[:, :n_total] = basis
                if node_embs.dim() == 3:
                    L = node_embs.shape[1]
                    basis_pad_bc = basis_pad.unsqueeze(1).expand(-1, L, -1)
                else:
                    basis_pad_bc = basis_pad
                node_embs = self.input_mlp(torch.cat([basis_pad_bc, node_embs], dim=-1))

            # Identify leaf vs internal positions in the tree
            is_leaf = (basis != 0).sum(dim=1) == 1
            tree_leaf_positions = is_leaf.nonzero(as_tuple=False).squeeze(-1)
            # Original leaf IDs for tree leaves (column index of the 1 in each one-hot row)
            tree_leaf_ids = basis[tree_leaf_positions].argmax(dim=1)
            missing_leaf_ids = step.missing_leaves.to(device)
            n_missing = len(missing_leaf_ids)
            missing_leaf_positions = torch.arange(n_tree, n_tree + n_missing, device=device)
            leaf_idx = torch.cat([tree_leaf_positions, missing_leaf_positions])

            # Build [N_ext, n_leaf_positions, 2] stacked basis for SPDBiasedAttention.
            # Channel 0 = rooted subtree basis; channel 1 = squaretrick absorption.
            n_leaf_positions = leaf_idx.shape[0]
            n_tree_leaves = len(tree_leaf_positions)
            N_ext = n_tree + n_missing
            leaf_pos_to_original = torch.cat([tree_leaf_ids, missing_leaf_ids])
            internal_positions = (~is_leaf).nonzero(as_tuple=False).squeeze(-1)

            if step.node_basis_topo is None:
                raise ValueError("node_basis_topo is required when attn_head=True")
            basis_topo = step.node_basis_topo.to(device)

            basis_ext = torch.zeros(N_ext, n_leaf_positions, 2, device=device)
            # Tree + missing leaves: identity rows in both channels.
            tl_arange = torch.arange(n_tree_leaves, device=device)
            basis_ext[tree_leaf_positions, tl_arange, :] = 1.0
            basis_ext[missing_leaf_positions,
                      torch.arange(n_tree_leaves, n_tree_leaves + n_missing, device=device), :] = 1.0
            # Internals: reindex by leaf_pos_to_original, both channels.
            basis_ext[internal_positions, :, 0] = basis[internal_positions][:, leaf_pos_to_original]
            basis_ext[internal_positions, :, 1] = basis_topo[internal_positions][:, leaf_pos_to_original]

            # Extend depths/counts for missing leaves (zeros)
            depths_ext = torch.zeros(N_ext, device=device)
            counts_ext = torch.ones(N_ext, device=device)
            depths_ext[:n_tree] = step.depths.to(device)
            counts_ext[:n_tree] = step.subtree_counts.to(device)

            root_position_in_tree = None
            if getattr(self.tree_model, 'use_norm_head', False):
                # Root token is the last leaf id; find its position in the tree.
                root_token_id = basis.shape[1] - 1
                root_position_in_tree = basis[:, root_token_id].argmax()

            return self.tree_model(
                node_embs,
                step.spd.to(device),
                step.time,
                new_node_emb,
                edge_children,
                step.edge_parents.to(device),
                hamming_dist=hamming_dist,
                leaf_idx=leaf_idx,
                basis=basis_ext,
                depths=depths_ext,
                subtree_counts=counts_ext,
                leaf_pos_to_original=leaf_pos_to_original,
                root_position_in_tree=root_position_in_tree,
            )
        return self.tree_model(
            node_embs.unsqueeze(0).expand(len(step.missing_leaves), -1, -1).contiguous(),
            step.time,
            new_node_emb,
            edge_children,
            step.edge_parents.to(leaf_features.device),
        )
