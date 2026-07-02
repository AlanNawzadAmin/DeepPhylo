import math

import torch
import torch.nn as nn

from ..spd_attn import SPDBiasedAttention


class AttentionMergePredictor(nn.Module):
    def __init__(self, h_size, n_layers, n_heads, max_spd, unpooled=False,
                 pair_dim=256, pair_inner_dim=32):
        super().__init__()
        assert unpooled, "Only unpooled mode is supported"
        self.h_size = h_size
        self.unpooled = True
        self.attn_module = SPDBiasedAttention(
            h_size, n_layers, n_heads, max_spd,
            pair_dim=pair_dim, pair_inner_dim=pair_inner_dim,
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

    def forward(self, leaf_features, spd, hamming_dist,
                leaf_idx, basis, exposed_nodes, depths, counts):
        device = leaf_features.device
        basis = basis.to(device)
        # basis is [N, n_leaves, 2]: ch0 rooted subtree, ch1 squaretrick absorption.
        assert basis.dim() == 3 and basis.shape[-1] == 2, \
            "basis must be [N, n_leaves, 2]"

        # Seed node embeddings use the rooted channel (parity with prior behavior).
        node_embeddings = torch.einsum('ij,jlh->ilh', basis[..., 0], leaf_features)

        N = node_embeddings.shape[0]
        full_hamming = torch.full((N, N), -1.0, device=device, dtype=hamming_dist.dtype)
        full_hamming[leaf_idx.unsqueeze(1), leaf_idx.unsqueeze(0)] = hamming_dist
        full_hamming.fill_diagonal_(-1)

        out = self.attn_module(
            node_embeddings, spd, full_hamming,
            leaf_idx, basis, depths.to(device), counts.to(device),
        )

        if exposed_nodes.numel() < 2:
            return torch.empty(0, device=device)

        exposed_out = out[exposed_nodes]
        i, j = torch.triu_indices(exposed_out.shape[0], exposed_out.shape[0], 1, device=device)
        pair_sum = exposed_out[i] + exposed_out[j]
        pair_prod = exposed_out[i] * exposed_out[j]
        pair_features = torch.cat([pair_sum, pair_prod], dim=-1)
        return self.pair_readout(pair_features).squeeze(-1)
