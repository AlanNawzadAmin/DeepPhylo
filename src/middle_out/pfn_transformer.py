import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, out_channels, bias=True):
        super().__init__()
        self.out_channels = out_channels

        self.register_buffer(
            "coef",
            torch.arange(0, out_channels, step=2).true_divide(out_channels)
            * math.log(10000),
        )
        self.embedding = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=bias),
            nn.ELU(),
            nn.Linear(out_channels, out_channels, bias=bias),
        )

    def forward(self, t):
        temb = torch.cat([torch.sin(self.coef * t), torch.cos(self.coef * t)], dim=-1)

        assert temb.shape == (self.out_channels,)
        temb = self.embedding(temb.unsqueeze(0))
        return temb


class TransformerWithFeatures(nn.Module):
    def __init__(self, n_max, cfg):
        super().__init__()
        self.n_max = n_max
        self.hidden_dim, self.n_head = cfg.hidden_dim, cfg.n_head
        self.MHA = nn.MultiheadAttention(self.hidden_dim, self.n_head, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(n_max + self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ELU(),
            nn.Linear(self.hidden_dim * 2, 1),
        )
        self.time_embedding = TimeEmbedding(2 * self.hidden_dim)
        self.query_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.register_buffer("query", torch.eye(n_max - 3, self.hidden_dim))

    def forward(self, node_features, t, new_node_emb, u, v):
        temb = self.time_embedding(t)
        if new_node_emb.ndim == 1:
            new_node_emb = new_node_emb.unsqueeze(0)
        batch_size = new_node_emb.shape[0]

        node_features = node_features.expand(batch_size, -1, -1).contiguous()
        node_features = self.mlp(node_features)
        new_node_query = self.query_mlp(new_node_emb)  # (B, D) -> (B, E)
        new_feature, _ = self.MHA(
            new_node_query[:, None, :], node_features, node_features
        )

        child_info = node_features[:, u]
        parent_info = node_features[:, v]
        edge_info = torch.max(child_info, parent_info)
        edge_info = torch.concat(
            [new_feature.expand(-1, edge_info.shape[1], -1), edge_info], dim=-1
        )
        logits = self.readout(edge_info + temb.unsqueeze(1)).squeeze(-1)
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        return logits
