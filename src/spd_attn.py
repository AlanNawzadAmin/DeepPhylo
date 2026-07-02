import math

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def sinusoidal_embedding(x, dim, base=10000):
    device = x.device
    half_dim = dim // 2
    emb = math.log(base) / half_dim
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = x.unsqueeze(-1) * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    # Unit-norm: the raw norm is exactly sqrt(half_dim) (each frequency
    # contributes sin²+cos²=1), so dividing by that constant gives ‖emb‖=1
    # for every input, independent of dim.
    return emb / math.sqrt(half_dim)


def compute_spd_nodes(obs_graph):
    """Build full SPD matrix over the observed graph nodes."""
    device = obs_graph.device
    if obs_graph.num_edges() == 0:
        return torch.zeros(
            obs_graph.num_nodes(),
            obs_graph.num_nodes(),
            device=device,
            dtype=torch.long,
        )
    # dgl.to_bidirected -> to_simple requires a CPU graph.
    obs_undirected = dgl.to_bidirected(obs_graph.cpu())
    return dgl.shortest_dist(obs_undirected).to(device)


def build_basis_from_graph(graph, leaf_idx):
    """[n_nodes, n_leaves] subtree-membership basis with L2-normalized rows."""
    n_leaves = leaf_idx.shape[0]
    n_nodes = graph.num_nodes()
    basis = torch.zeros(n_nodes, n_leaves, device=graph.device)
    basis[leaf_idx, torch.arange(n_leaves, device=graph.device)] = 1.0
    with graph.local_scope():
        graph.ndata["emb"] = basis
        dgl.prop_nodes_topo(
            graph,
            lambda edges: {"m": edges.src["emb"]},
            lambda nodes: {"emb": torch.sum(nodes.mailbox["m"], dim=1) + nodes.data["emb"]},
        )
        raw = graph.ndata["emb"]
        return raw / raw.norm(dim=1, keepdim=True).clamp(min=1e-9)


class GraphormerAttentionBlock(nn.Module):
    def __init__(self, h_size, n_heads, max_spd, pair_dim=None,
                 use_hamming=True, use_spd=True, use_layernorm=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = h_size // n_heads

        self.q_proj = nn.Linear(h_size, h_size, bias=False)
        self.k_proj = nn.Linear(h_size, h_size, bias=False)
        self.v_proj = nn.Linear(h_size, h_size, bias=False)
        self.g_proj = nn.Linear(h_size, h_size)
        self.out_proj = nn.Linear(h_size, h_size)

        self.max_spd = max_spd
        self.use_spd = use_spd
        if self.use_spd:
            # +1 clamped/unreachable, +2 not-in-tree.
            self.spd_bias = nn.Embedding(max_spd + 3, n_heads)
        self.hamming_emb_dim = 32
        self.use_hamming = use_hamming
        if self.use_hamming:
            self.hamming_proj = nn.Linear(self.hamming_emb_dim, n_heads, bias=False)
            nn.init.normal_(self.hamming_proj.weight, std=1/self.hamming_emb_dim)

        self.ln1 = nn.LayerNorm(h_size) if use_layernorm else nn.Identity()

        self.use_pair = pair_dim is not None
        if self.use_pair:
            self.pairs_norm = nn.LayerNorm(pair_dim) if use_layernorm else nn.Identity()
            self.pair_bias_proj = nn.Linear(pair_dim, n_heads, bias=False)
            # Alignment-aware: 1/sqrt(pair_dim) extra factor (input is learned).
            with torch.no_grad():
                self.pair_bias_proj.weight.mul_(1.0 / math.sqrt(pair_dim))

    def forward(self, x, spd_idx, hamming_feats, hmask, pair_rep=None):
        x = self.ln1(x)
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        g = F.sigmoid(self.g_proj(x)).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        attn_bias = None
        if self.use_spd:
            attn_bias = self.spd_bias(spd_idx).permute(2, 0, 1).unsqueeze(0)  # [1, n_heads, N, N]
        if self.use_hamming:
            h_bias = self.hamming_emb_dim*(self.hamming_proj(hamming_feats) * hmask.unsqueeze(-1)).permute(2, 0, 1).unsqueeze(0)
            attn_bias = h_bias if attn_bias is None else attn_bias + h_bias
        if self.use_pair and pair_rep is not None:
            pair_bias = self.pair_bias_proj(self.pairs_norm(pair_rep)).permute(2, 0, 1).unsqueeze(0)
            attn_bias = pair_bias if attn_bias is None else attn_bias + pair_bias

        # µP attention scale: 1/d_head (not the default 1/sqrt(d_head)).
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias,
                                             scale=1.0 / self.head_dim)
        out = g * out

        out = out.transpose(1, 2).reshape(B, S, -1)
        out = self.out_proj(out)
        return out


class SPDBiasedAttention(nn.Module):
    """Interleaved column/row attention with SPD and Hamming biases.

    Operates on unpooled embeddings [N, L, h] with batch_size=1.
    A learnable global token is appended per node and used (together
    with the mean over MSA positions) as the output representation.
    """

    def __init__(
        self, h_size, n_layers, n_heads, max_spd,
        pair_dim=256, pair_inner_dim=32, d_proj=32,
        use_refresh=True, use_cls=True, use_pair=True, use_hamming=True,
        use_rope=True, use_spd=True, pool_output=True, use_layernorm=True,
        depth_mup=False, seed_struct=False, op_mean_mup=True,
    ):
        super().__init__()
        from .bottom_up.BayesNJ.evopf import OuterProductMean, RowAttenGated, Transition

        self.h_size = h_size
        self.d_proj = d_proj
        self.use_refresh = use_refresh
        self.use_cls = use_cls
        self.use_pair = use_pair
        self.use_hamming = use_hamming
        self.use_rope = use_rope
        self.use_spd = use_spd
        self.pool_output = pool_output
        # depth_mup=True switches the residual stack from CompleteP (1/L branch
        # scaling) to Adam Depth-µP: 1/√L branch scaling here + an extra 1/√L on
        # the depth-stack LR (configure_optimizers keys off _mup_depth_scaled,
        # which is only set when this is on). Off by default → CompleteP.
        self.depth_mup = depth_mup
        # use_layernorm=False swaps every pre-norm in the stack (col/row/msa-FFN +
        # pair-bias/update/transition norms) for Identity. refresh_ln is kept
        # regardless — it's structural (normalizes cat([rooted, topo]) before the
        # refresh MLP), not a residual-branch pre-norm.
        self.use_layernorm = use_layernorm
        # use_pair=False disables pair track entirely (pair_bias + pair_init + updates + transitions).
        self.pair_dim = pair_dim if use_pair else None
        block_pair_dim = pair_dim if use_pair else None
        self.attn_layers = nn.ModuleList(
            [GraphormerAttentionBlock(h_size, n_heads, max_spd, pair_dim=block_pair_dim,
                                      use_hamming=use_hamming, use_spd=use_spd,
                                      use_layernorm=use_layernorm)
             for _ in range(n_layers)]
        )
        self.row_attn_layers = nn.ModuleList(
            [RowAttenGated(h_size, n_heads, global_token=True, use_rope=use_rope,
                           use_layernorm=use_layernorm)
             for _ in range(n_layers)]
        )
        # MSA transition (standalone FFN with its own residual, matching EvoPF's
        # col → row → msa_trans block ordering). Pre-norm LN + 4× expand + GELU.
        self.msa_trans = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(h_size) if use_layernorm else nn.Identity(),
                    nn.Linear(h_size, 4 * h_size),
                    nn.GELU(),
                    nn.Linear(4 * h_size, h_size),
                )
                for _ in range(n_layers)
            ]
        )
        # Built only when used so disabled ablations leave no unused parameters
        # (DDP errors on those). cls_token: use_cls; refresh_ln/node_refresh_projs:
        # use_refresh. depth/counts proj stay — they also feed the structural seed.
        if use_cls:
            self.cls_token = nn.Parameter(torch.randn(h_size))
        # Refresh combines two basis channels (rooted + squaretrick absorption)
        # via pre-norm LN over 2h, then a 2-layer MLP back to h. depth+count emb
        # is folded into the rooted branch via a broadcast add. Kept even when
        # use_layernorm=False (structural, not a residual pre-norm).
        if use_refresh:
            self.refresh_ln = nn.LayerNorm(2 * h_size)
        self.depth_proj = nn.Linear(self.d_proj, h_size, bias=False)
        self.counts_proj = nn.Linear(self.d_proj, h_size, bias=False)
        nn.init.normal_(self.depth_proj.weight,
                        std=1/self.d_proj)
        nn.init.normal_(self.counts_proj.weight,
                            std=1/self.d_proj)
        # Opt-in structural node seed: the combined depth+count+node-type vector is
        # unit-normalized, then this learned projection sets the seeded direction
        # AND magnitude (see forward). Identity init → seed starts as the plain
        # normalized combination. Only built when requested (the top down heads
        # pass seed_struct=True + node_type); other callers are unaffected.
        self.seed_struct = seed_struct
        if seed_struct:
            self.seed_proj = nn.Linear(h_size, h_size, bias=False)
            nn.init.eye_(self.seed_proj.weight)
        if use_refresh:
            self.node_refresh_projs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * h_size, h_size),
                        nn.GELU(),
                        nn.Linear(h_size, h_size),
                    )
                    for _ in range(n_layers)
                ]
            )

        # Pair stack (matches EvoPF except for pair self-attention).
        # Last layer's pair update isn't consumed by any subsequent col_layer.
        # op_mean_mup controls the OuterProductMean column normalization: True
        # (default, µP) divides the column contraction by √L; False divides by L
        # (the paper's column mean). Exposed so we can A/B the √L vs 1/L choice.
        if self.use_pair:
            self.pair_init_proj = nn.Linear(h_size, pair_dim)
            self.pair_updates = nn.ModuleList(
                [OuterProductMean(h_size, pair_dim, pair_inner_dim, symmetric=False,
                                  use_layernorm=use_layernorm, mup=op_mean_mup)
                 for _ in range(n_layers - 1)]
            )
            self.pair_transitions = nn.ModuleList(
                [Transition(pair_dim, use_layernorm=use_layernorm) for _ in range(n_layers - 1)]
            )

        # Depth-µP: parameters in these per-layer lists feed a branch scaled by
        # beta=1/√L in forward, so configure_optimizers gives them an extra 1/√L
        # LR. Entrance/shared params (cls_token, depth/counts/pair_init projs,
        # refresh_ln) are NOT depth-scaled and keep their O(1) µP LR. Only marked
        # when depth_mup=True; otherwise the optimizer leaves the LR untouched.
        if self.depth_mup:
            depth_lists = [self.attn_layers, self.row_attn_layers, self.msa_trans]
            if use_refresh:
                depth_lists.append(self.node_refresh_projs)
            if self.use_pair:
                depth_lists += [self.pair_updates, self.pair_transitions]
            for ml in depth_lists:
                ml._mup_depth_scaled = True

    def _attn_block(
        self,
        all_embs, pair_rep,
        col_layer, row_layer, msa_trans,
        spd_idx, hamming_feats, hmask,
        beta,
    ):
        """One col-attn → row-attn → msa-FFN block, bundled so a single
        checkpoint() per layer also frees the inter-sub-block activations
        (row's input/output, msa_trans's intermediates) instead of just col's
        and msa_trans's separately."""
        x = all_embs.transpose(0, 1)
        branch = col_layer(x, spd_idx, hamming_feats, hmask, pair_rep)
        all_embs = (x + beta * branch).transpose(0, 1)  # [N, L+1, h]

        x = all_embs.permute(2, 0, 1).unsqueeze(0)  # [1, h, N, L+1]
        all_embs = (x + beta * row_layer(x)).squeeze(0).permute(1, 2, 0)  # [N, L+1, h]

        all_embs = all_embs + beta * msa_trans(all_embs)
        return all_embs

    def _layer_block(
        self,
        all_embs, pair_rep,
        basis_nl, dc_nl, leaf_idx, non_leaf_idx,
        spd_idx, hamming_feats, hmask,
        i, beta, L, n_layers,
    ):
        """One full layer: per-node refresh → col/row/msa attn block → pair update.

        Bundling all three into a single checkpoint() per layer frees the refresh
        and pair-update activations too (they previously ran OUTSIDE the per-block
        checkpoint and were retained across every layer). The computation — order,
        ops, and values — is byte-for-byte identical to running them inline; only
        the activation-saving boundary moves, so the math (forward + gradients) is
        unchanged."""
        # Refresh: re-derive non-leaf node embeddings per position; write back with
        # index_add so leaf rows are untouched. Two basis channels: [...,0] rooted
        # subtree, [...,1] squaretrick. depth+count folded into the rooted branch,
        # then cat([rooted, topo]) → LN(2h) → Linear(2h, h).
        if self.use_refresh:
            refresh_proj = self.node_refresh_projs[i]
            leaf_embs = all_embs.index_select(0, leaf_idx)                           # [n_leaves, L+1, h]
            fresh_rooted = torch.einsum("nl,lph->nph", basis_nl[..., 0], leaf_embs) + dc_nl.unsqueeze(1)
            fresh_topo   = torch.einsum("nl,lph->nph", basis_nl[..., 1], leaf_embs)  # [n_obs, L+1, h]
            stacked  = torch.cat([fresh_rooted, fresh_topo], dim=-1)                 # [n_obs, L+1, 2h]
            fresh_nl = refresh_proj(self.refresh_ln(stacked))                        # [n_obs, L+1, h]
            # index_add_ requires matching dtypes (unlike +, which implicitly upcasts);
            # under autocast, refresh_proj outputs bf16 while all_embs stays fp32.
            all_embs = all_embs.index_add(0, non_leaf_idx, (beta * fresh_nl).to(all_embs.dtype))

        all_embs = self._attn_block(
            all_embs, pair_rep,
            self.attn_layers[i], self.row_attn_layers[i], self.msa_trans[i],
            spd_idx, hamming_feats, hmask, beta,
        )

        # Pair stack (MSA → Pair via OPM, then pair FFN). OPM expects [B, d_m, n, L].
        # Skipped on the last layer (its pair_rep is never consumed by a subsequent
        # col_layer) and when use_pair is off.
        if self.use_pair and i != n_layers - 1:
            msa_in = all_embs[:, :L, :].permute(2, 0, 1).unsqueeze(0)  # [1, h, N, L]
            pair_rep = pair_rep + beta * self.pair_updates[i](msa_in).squeeze(0)
            pair_rep = pair_rep + beta * self.pair_transitions[i](pair_rep)
        return all_embs, pair_rep

    def forward(self, all_embs, spd, hamming_dist, leaf_idx, basis, depths, subtree_counts,
                node_type=None):
        assert all_embs.dim() == 3, "Only unpooled mode is supported"
        N, L, h = all_embs.shape

        # PairEmbedder-style init (EvoPF): μ_i + μ_j over L, with Linear+GELU on node features.
        if self.use_pair:
            f = F.gelu(self.pair_init_proj(all_embs)).mean(dim=1)  # [N, pair_dim]
            pair_rep = f.unsqueeze(0) + f.unsqueeze(1)  # [N, N, pair_dim]
        else:
            pair_rep = None

        # Structural node features (depth + subtree size), shared by the node seed
        # and the per-layer refresh. Only computed when something consumes them:
        # the top down heads pass node_type (→ seed) and/or set use_refresh. Other
        # SPDBiasedAttention callers pass neither, so their behavior is unchanged.
        if self.use_refresh or node_type is not None:
            mask = torch.ones(N, dtype=torch.bool, device=all_embs.device)
            mask[leaf_idx] = False
            non_leaf_idx = mask.nonzero(as_tuple=True)[0]
            depth_emb = self.d_proj*self.depth_proj(sinusoidal_embedding(depths, self.d_proj).to(all_embs.dtype))
            counts_emb = self.d_proj*self.counts_proj(sinusoidal_embedding(subtree_counts, self.d_proj).to(all_embs.dtype))
            dc_nl = (depth_emb + counts_emb).index_select(0, non_leaf_idx)       # [n_obs, h]

        # Seed structural features into the node init: node-type tag + depth +
        # subtree size, combined THEN unit-normalized per node so the (d_proj-
        # scaled) signal perturbs rather than swamps the leaf-derived seed.
        # Position-constant (matches refresh); when use_refresh=False this is the
        # only place these features enter the model.
        if node_type is not None:
            struct = dc_nl + node_type.to(all_embs.dtype)                        # [n_obs, h]
            struct = struct / (struct.norm(dim=-1, keepdim=True) + 1e-6)         # unit norm first
            struct = self.seed_proj(struct)                                      # learned dir + scale
            all_embs = all_embs.index_add(
                0, non_leaf_idx, struct.unsqueeze(1).expand(-1, L, h).to(all_embs.dtype)
            )

        # Refresh-only ingredient: only needed when use_refresh=True.
        if self.use_refresh:
            basis_nl = basis.index_select(0, non_leaf_idx)                       # [n_obs, n_leaves, 2]

        # Append global token as position L+1 for every node (if enabled).
        if self.use_cls:
            cls = self.cls_token.to(all_embs.dtype).expand(N, 1, h)
            all_embs = torch.cat([all_embs, cls], dim=1)  # [N, L+1, h]

        # Precompute SPD indices (only if use_spd).
        if self.use_spd:
            spd_idx = spd.clone()
            not_in_tree = spd_idx == -2
            spd_idx[spd_idx < 0] = self.attn_layers[0].max_spd + 1
            spd_idx[not_in_tree] = self.attn_layers[0].max_spd + 2
            spd_idx = spd_idx.long()
        else:
            spd_idx = None

        # Precompute hamming sinusoidal features (only if use_hamming).
        if self.use_hamming:
            hmask = hamming_dist >= 0
            d = hamming_dist.clamp(min=0)
            hamming_emb_dim = self.attn_layers[0].hamming_emb_dim
            # sinusoidal_embedding already returns unit-norm vectors.
            hamming_feats = sinusoidal_embedding(d * 100.0, hamming_emb_dim).to(all_embs.dtype)
        else:
            hmask = None
            hamming_feats = None

        n_layers = len(self.attn_layers)
        # Residual-branch scaling. Default (CompleteP): each branch contributes
        # 1/L. depth_mup=True switches to Adam Depth-µP's 1/√L, which — combined
        # with the extra 1/√L on the depth-stack LR in configure_optimizers —
        # admits a non-trivial, depth-independent init.
        beta = 1.0 / math.sqrt(n_layers) if self.depth_mup else 1.0 / n_layers
        # refresh ingredients only exist when use_refresh; gate the pass-through so
        # _layer_block never touches an undefined name (it reads them only then).
        if self.use_refresh:
            r_basis, r_dc, r_nl = basis_nl, dc_nl, non_leaf_idx
        else:
            r_basis, r_dc, r_nl = None, None, None
        for i in range(n_layers):
            use_ckpt = self.training
            # One bundled layer (refresh → col/row/msa attn → pair update) per
            # checkpoint, so refresh + pair-update activations are freed/recomputed
            # too instead of retained across every layer (see _layer_block).
            args = (
                all_embs, pair_rep,
                r_basis, r_dc, leaf_idx, r_nl,
                spd_idx, hamming_feats, hmask,
                i, beta, L, n_layers,
            )
            if use_ckpt:
                all_embs, pair_rep = checkpoint(self._layer_block, *args, use_reentrant=False)
            else:
                all_embs, pair_rep = self._layer_block(*args)

        if not self.pool_output:
            # Return un-pooled [N, L(+1), h] so caller can extract per-position features.
            return all_embs
        if self.use_cls:
            return all_embs[:, -1, :]
        return all_embs.mean(dim=1)


# Keep old names as aliases for backward compatibility with configs/checkpoints
SPDBiasedAtention = SPDBiasedAttention
SimpleBiasedAttention = SPDBiasedAttention
