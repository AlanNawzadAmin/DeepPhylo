import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottom_up.BayesNJ.evopf import EvoPF
from .spd_attn import SPDBiasedAttention


class EvoPFSpeciesEncoder(nn.Module):
    def __init__(self, h_size, n_blocks, n_heads, embed_dim, pair_dim, mode,
                 attn_head=True, use_rope=True, mup=True, activation="gelu"):
        super().__init__()
        self.vocab_size = 7 if mode == "dna" else 23

        evopf = EvoPF(
            vocab_size=self.vocab_size,
            n_blocks=n_blocks,
            n_heads=n_heads,
            h_dim=embed_dim,
            pair_dim=pair_dim,
            use_opm=True,
            distance_mlp=False,
            symmetric=True,
            use_rope=use_rope,
            mup=mup,
            activation=activation,
        )
        # pairs_to_dm is only read for the discarded dm output; pair_embedder
        # feeds pairs_to_dm and evoblocks — with n_blocks=0, both sinks are dead.
        evopf.pairs_to_dm.requires_grad_(False)
        if n_blocks == 0:
            evopf.pair_embedder.requires_grad_(False)
        else:
            # Last block's pair stack (pair_update → pair_atten → pair_trans)
            # produces the final `pairs` that only feeds the frozen pairs_to_dm.
            # No subsequent col_atten consumes it, so its params get no grad.
            last_block = evopf.evoblocks[-1]
            for name in ("pair_update", "pair_atten", "pair_trans"):
                getattr(last_block, name).requires_grad_(False)
        if attn_head:
            # BayesNJ (attn_head=False) is the only path that uses these heads.
            for name in ("parent_embedder", "q_embedder", "e_norm"):
                getattr(evopf, name).requires_grad_(False)
        self.model = evopf

        self.species_encoder = nn.Linear(embed_dim, h_size)
        if not attn_head:
            # BayesNJ path calls self.model directly, bypassing this Linear.
            self.species_encoder.requires_grad_(False)

    def forward(self, seqs, tree_to_n_leaves, pool=True, **kwargs):
        # **kwargs accepts (and ignores) tree-context args from middle_out's
        # encode_leaf_features so callers can use either encoder uniformly.
        def trim_msa(x):
            # Remove columns that are all zeros.
            cols_to_keep = x.sum(dim=0).gt(0)
            return x[:, cols_to_keep]

        # one_hot is int64; cast to whatever float dtype the model is in
        # (bf16 under bf16-true precision, fp32 otherwise) so conv2d's bias
        # dtype matches the input.
        model_dtype = next(self.model.parameters()).dtype
        one_hot_msas = [
            torch.nn.functional.one_hot(trim_msa(x), num_classes=self.vocab_size)
            .unsqueeze(0)
            .permute(0, 3, 2, 1)
            .contiguous()
            .to(model_dtype)
            for x in torch.split(seqs, tree_to_n_leaves.tolist())
        ]
        if pool:
            # [1, h_dim, n, L] -> pool -> [1, h_dim, n] -> [n, h_dim]
            outputs = [self.model(msa)[1].squeeze(0).permute(1, 0) for msa in one_hot_msas]
            return self.species_encoder(torch.vstack(outputs))
        else:
            assert len(one_hot_msas) == 1, "Unpooled mode requires batch size 1"
            # [1, h_dim, n, L] -> [n, L, h_dim]
            msa_emb = self.model(one_hot_msas[0], pool=False)[1].squeeze(0).permute(1, 2, 0)
            return self.species_encoder(msa_emb)  # [n, L, h_size]


class SPDAttnSpeciesEncoder(nn.Module):
    """Replacement for EvoPFSpeciesEncoder that runs the *exact same* tree ops
    as the middle_out rate predictor: embeds seqs, expands to [n_tree+n_missing, L, h]
    via the step's basis (same as score_step), runs SPDBiasedAttention on that
    extended set, then projects out the internal-node rows and returns the
    leaf rows in original-leaf-id order.

    All tree-aware flags (refresh, spd, hamming, cls, rope) default OFF so it's
    "EvoPF minus pair self-attention" by default. Caller flips them on to A/B.
    """

    def __init__(self, h_size, n_blocks, n_heads, embed_dim, pair_dim, mode,
                 attn_head=True,
                 use_refresh=False, use_cls=False, use_pair=True,
                 use_hamming=False, use_rope=False, use_spd=False,
                 max_spd=250, pair_inner_dim=32):
        super().__init__()
        assert attn_head, "SPDAttnSpeciesEncoder only supports attn_head=True (unpooled)."
        self.vocab_size = 7 if mode == "dna" else 23

        # Embed one-hot MSA → embed_dim → h_size (matches EvoPF's MSAEmbedder
        # + species_encoder Linear).
        self.embedder = nn.Sequential(
            nn.Linear(self.vocab_size, embed_dim),
            nn.GELU(),
        )
        self.species_encoder = nn.Linear(embed_dim, h_size)

        if n_blocks > 0:
            self.encoder = SPDBiasedAttention(
                h_size, n_blocks, n_heads, max_spd,
                pair_dim=pair_dim, pair_inner_dim=pair_inner_dim,
                use_refresh=use_refresh, use_cls=use_cls, use_pair=use_pair,
                use_hamming=use_hamming, use_rope=use_rope, use_spd=use_spd,
                pool_output=False,  # need per-position features back
            )
        else:
            self.encoder = None

    def forward(self, seqs, tree_to_n_leaves, pool=True, step=None, hamming_dist=None,
                **_unused):
        assert not pool, "SPDAttnSpeciesEncoder requires pool=False."
        assert tree_to_n_leaves.numel() == 1, "Unpooled mode requires batch size 1"
        assert step is not None, "SPDAttnSpeciesEncoder requires step (tree state)."

        def trim_msa(x):
            cols_to_keep = x.sum(dim=0).gt(0)
            return x[:, cols_to_keep]

        x = trim_msa(seqs)  # [n_leaves, L_trimmed]
        # Cast to the embedder's float dtype (bf16 under bf16-true).
        embed_dtype = self.embedder[0].weight.dtype
        one_hot = F.one_hot(x, num_classes=self.vocab_size).to(embed_dtype)

        msa_emb = self.embedder(one_hot)  # [n_leaves, L, embed_dim]
        leaf_features = self.species_encoder(msa_emb)  # [n_leaves, L, h_size]

        if self.encoder is None:
            return leaf_features

        # === Replicate score_step construction (n_tree+n_missing extended set) ===
        device = leaf_features.device
        L = leaf_features.shape[1]
        n_real = leaf_features.shape[0]

        basis = step.node_basis.to(device)             # [n_tree, n_real+1]  (root-token = n_real)
        basis_topo = step.node_basis_topo.to(device)   # [n_tree, n_real+1]
        n_tree = basis.shape[0]

        # Append synthetic root-token row (mean of real leaves) so basis cols line up.
        # The outer encode_leaf_features appends its own *learnable* root after we
        # return; here we only need a placeholder so the einsum dims match.
        root_feat = leaf_features.mean(dim=0, keepdim=True)  # [1, L, h]
        leaf_features_ext = torch.cat([leaf_features, root_feat], dim=0)  # [n_real+1, L, h]
        n_total_leaves = leaf_features_ext.shape[0]

        # node_embs from rooted basis (same as build_node_embeddings_from_basis).
        node_embs = torch.einsum('ij,jlh->ilh', basis, leaf_features_ext)  # [n_tree, L, h]

        # Missing-leaf context tokens.
        missing_leaf_ids = step.missing_leaves.to(device)
        n_missing = len(missing_leaf_ids)
        new_node_emb = leaf_features_ext[missing_leaf_ids]  # [n_missing, L, h]

        all_embs = torch.cat([node_embs, new_node_emb], dim=0)  # [N_ext, L, h]
        N_ext = n_tree + n_missing

        # spd_ext (missing leaves: -2 = not in tree)
        spd_ext = torch.full((N_ext, N_ext), -2, device=device, dtype=step.spd.dtype)
        spd_ext[:n_tree, :n_tree] = step.spd.to(device)
        idx = torch.arange(n_tree, N_ext, device=device)
        spd_ext[idx, idx] = 0

        # Identify leaf positions and their original IDs (used for hamming_ext + basis_ext).
        is_leaf = (basis != 0).sum(dim=1) == 1
        tree_leaf_positions = is_leaf.nonzero(as_tuple=False).squeeze(-1)
        tree_leaf_ids = basis[tree_leaf_positions].argmax(dim=1)
        missing_leaf_positions = torch.arange(n_tree, N_ext, device=device)
        leaf_idx = torch.cat([tree_leaf_positions, missing_leaf_positions])

        n_leaf_positions = leaf_idx.shape[0]
        n_tree_leaves = len(tree_leaf_positions)
        leaf_pos_to_original = torch.cat([tree_leaf_ids, missing_leaf_ids])
        internal_positions = (~is_leaf).nonzero(as_tuple=False).squeeze(-1)

        # hamming_ext: pull leaf-leaf hamming into the extended set at the right
        # positions. Both tree leaves and missing leaves get real hamming distances
        # (root token has no real sequence, so its entries stay at -1).
        n_real = hamming_dist.shape[0]
        is_real = leaf_pos_to_original < n_real
        safe_ids = torch.where(is_real, leaf_pos_to_original,
                               torch.zeros_like(leaf_pos_to_original))
        hamming_pairs = hamming_dist[safe_ids[:, None], safe_ids[None, :]]
        real_pairs = is_real[:, None] & is_real[None, :]
        hamming_pairs = torch.where(real_pairs, hamming_pairs,
                                    torch.full_like(hamming_pairs, -1))
        hamming_ext = torch.full(
            (N_ext, N_ext), -1, device=device, dtype=hamming_dist.dtype,
        )
        hamming_ext[leaf_idx[:, None], leaf_idx[None, :]] = hamming_pairs
        hamming_ext.fill_diagonal_(-1)

        basis_ext = torch.zeros(N_ext, n_leaf_positions, 2, device=device)
        tl_arange = torch.arange(n_tree_leaves, device=device)
        basis_ext[tree_leaf_positions, tl_arange, :] = 1.0
        basis_ext[missing_leaf_positions,
                  torch.arange(n_tree_leaves, n_tree_leaves + n_missing, device=device), :] = 1.0
        basis_ext[internal_positions, :, 0] = basis[internal_positions][:, leaf_pos_to_original]
        basis_ext[internal_positions, :, 1] = basis_topo[internal_positions][:, leaf_pos_to_original]

        depths_ext = torch.zeros(N_ext, device=device)
        counts_ext = torch.ones(N_ext, device=device)
        depths_ext[:n_tree] = step.depths.to(device).float()
        counts_ext[:n_tree] = step.subtree_counts.to(device).float()

        # === Run SPDBiasedAttention on the extended set ===
        out = self.encoder(all_embs, spd_ext, hamming_ext,
                           leaf_idx, basis_ext, depths_ext, counts_ext)
        # out is [N_ext, L (or L+1 if cls), h] because pool_output=False.
        # Slice off the cls position if present.
        out = out[:, :L, :]

        # === Project out internals: gather only the leaf rows, in original order ===
        # `leaf_idx` lists positions in [0, N_ext) where leaves live; the j-th
        # entry corresponds to original-leaf-id leaf_pos_to_original[j].
        # We want output[i] = out[position of original leaf i] for each i.
        leaves = torch.empty(n_total_leaves, L, out.shape[-1], device=device, dtype=out.dtype)
        leaves[leaf_pos_to_original] = out[leaf_idx]
        # Drop the synthetic root row so the outer encode_leaf_features can append
        # its own learnable root from these (now attention-mixed) real-leaf features.
        return leaves[:n_real]  # [n_real, L, h]
