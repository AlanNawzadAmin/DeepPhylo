"""QuartFormer network: per-quartet feature extraction + sparse attention over
quartet tokens + 3-way topology classifier.

Operates on a *batch of B blocks*, each a set of `k` taxa. Within a block the
C(k,4) quartets are the transformer tokens; a quartet attends only to quartets
that share >=3 taxa with it (QuartFormer's "sparse attention", here an additive
attention mask). Batching the blocks along a leading dimension keeps the GPU busy
(each block on its own would be a tiny sequence of quartet tokens).

Feature extractor: each alignment column restricted to a quartet's 4 taxa is a
4-tuple of residues (order = quartet slot order q0,q1,q2,q3, so it is sensitive
to which resolution is correct). We embed the four residues, concat, pass a small
MLP, and mean-pool over sites. Mean-pooling a nonlinear per-column map is a
learned generalization of the site-pattern-frequency vector used by the released
QuartFormer, and unlike a raw V**4 histogram it stays tractable for amino acids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuartetFeatureExtractor(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, feat_hidden=256, feat_dim=128,
                 site_chunk=512):
        super().__init__()
        self.res_embed = nn.Embedding(vocab_size, embed_dim)
        self.col_mlp = nn.Sequential(
            nn.Linear(4 * embed_dim, feat_hidden),
            nn.ReLU(),
            nn.Linear(feat_hidden, feat_dim),
        )
        self.feat_dim = feat_dim
        self.site_chunk = site_chunk

    def forward(self, quartet_residues):
        """quartet_residues: (B, Q, 4, L) long. Returns (B, Q, feat_dim)
        mean-pooled column features. Sites are processed in chunks to bound
        memory."""
        B, Q, _, L = quartet_residues.shape
        acc = quartet_residues.new_zeros(
            B, Q, self.feat_dim, dtype=self.col_mlp[0].weight.dtype)
        for start in range(0, L, self.site_chunk):
            chunk = quartet_residues[:, :, :, start:start + self.site_chunk]  # (B,Q,4,c)
            emb = self.res_embed(chunk)                          # (B,Q,4,c,e)
            c = emb.shape[3]
            emb = emb.permute(0, 1, 3, 2, 4).reshape(B, Q, c, -1)  # (B,Q,c,4e)
            col = self.col_mlp(emb)                              # (B,Q,c,feat_dim)
            acc = acc + col.sum(dim=2)
        return acc / L


class SparseAttentionBlock(nn.Module):
    """Pre-norm transformer block with masked multi-head self-attention over
    quartet tokens. `attn_bias` is an additive (Q, Q) mask (0 keep / -inf drop),
    broadcast over the block-batch and heads.

    Pre-norm (normalize the sublayer *input*, add the raw residual) rather than
    post-norm: with post-norm the transformer's input stream (input_layer output)
    is never normalized, and its magnitude grew unbounded over training until it
    saturated attention layer 0 and blew up the gradients (grad_norm ~2 -> ~1000+),
    collapsing every run to chance. Pre-norm keeps every sublayer's input bounded
    regardless of how large the residual stream grows, which removes that runaway."""

    def __init__(self, dim, n_heads, ffn_dim, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, dim),
        )
        self.dropout = dropout

    def _attn(self, x, attn_bias):
        B, Q, D = x.shape
        q = self.q_proj(x).view(B, Q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, Q, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, Q, self.n_heads, self.head_dim).transpose(1, 2)
        # (B, heads, Q, head_dim); mask (1,1,Q,Q) broadcasts over B and heads.
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias.unsqueeze(0).unsqueeze(0),
        )
        out = out.transpose(1, 2).reshape(B, Q, D)
        return self.o_proj(out)

    def forward(self, x, attn_bias):
        x = x + self._attn(self.norm1(x), attn_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class QuartFormerNet(nn.Module):
    def __init__(self, vocab_size, block_size, token_dim=256, n_heads=8,
                 n_layers=3, ffn_dim=512, embed_dim=16, feat_hidden=256,
                 feat_dim=128, site_chunk=512):
        super().__init__()
        self.block_size = block_size
        self.feature_extractor = QuartetFeatureExtractor(
            vocab_size, embed_dim=embed_dim, feat_hidden=feat_hidden,
            feat_dim=feat_dim, site_chunk=site_chunk,
        )
        # species multi-hot (block_size) concatenated with the quartet feature.
        self.input_layer = nn.Sequential(
            nn.Linear(block_size + feat_dim, token_dim), nn.ReLU(),
        )
        # Everything upstream of here — the residue embedding, col_mlp and the
        # mean-pool that produce `feat`, plus this input projection — is entirely
        # unnormalized, and res_embed carries the *full* muP LR while the Linears
        # crawl at lr/fan_in. So `feat` grows without bound during training and
        # drags the token stream up with it: pre-norm alone kept the transformer
        # itself healthy, but the runaway just moved upstream and still detonated
        # the gradients (input_layer act_rms 0.13 -> 1.9 -> grad_norm 1 -> 40+ ->
        # collapse, on cat/indels at lr>=0.005; jc merely creeps more slowly).
        # This is the standard embedding-LayerNorm: it bounds the stream entering
        # the stack, and because the gradient back through it scales as 1/||feat||,
        # it also damps the positive-feedback loop that drives the growth.
        self.input_norm = nn.LayerNorm(token_dim)
        self.layers = nn.ModuleList([
            SparseAttentionBlock(token_dim, n_heads, ffn_dim)
            for _ in range(n_layers)
        ])
        # Pre-norm leaves the final block's output as the raw (unnormalized)
        # residual stream, whose magnitude grows with depth/training; normalize
        # it before the classifier so the head input stays bounded.
        self.final_norm = nn.LayerNorm(token_dim)
        self.classifier = nn.Linear(token_dim, 3)

    def forward(self, block_seqs, local_quartets, species_multihot, attn_bias):
        """block_seqs: (B, k, L) long residues for B blocks of k taxa each.
        local_quartets: (Q, 4) long, local indices into 0..k-1.
        species_multihot: (Q, k) float multi-hot of the 4 taxa (shared across B).
        attn_bias: (Q, Q) additive mask (0 keep / -inf drop).
        Returns logits (B, Q, 3)."""
        B = block_seqs.shape[0]
        residues = block_seqs[:, local_quartets]      # (B, Q, 4, L)
        feat = self.feature_extractor(residues)       # (B, Q, feat_dim)
        sp = species_multihot.to(feat.dtype).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([sp, feat], dim=-1)             # (B, Q, k+feat_dim)
        x = self.input_layer(x)                       # (B, Q, token_dim)
        x = self.input_norm(x)                        # bound the token stream
        for layer in self.layers:
            x = layer(x, attn_bias)
        x = self.final_norm(x)                        # bound the head input
        return self.classifier(x)                     # (B, Q, 3)
