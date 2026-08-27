# pyright: basic

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import parametrize


from flash_attn.layers.rotary import apply_rotary_emb_qkv_

from .memory import make_mem_hook
from .pf_sdk.pf.modules import MSAEmbedder, _make_activation, seq2pairs


@torch.compiler.disable
def _apply_rope_no_trace(qkv, cos, sin):
    """Wrap flash_attn rotary so dynamo doesn't trace into it (avoids graph-break
    that corrupts DDP submodule outputs with bare ints).

    flash_attn's triton kernel does `with torch.cuda.device(x.device.index):`,
    which raises 'invalid argument to exchangeDevice' when device.index is None
    (i.e. the tensor lives on bare 'cuda'). Force an explicit index in that
    case — `.to('cuda:N')` is a no-op when the storage is already on cuda:N,
    so this only fixes the metadata."""
    if qkv.device.type == "cuda" and qkv.device.index is None:
        qkv = qkv.to(f"cuda:{torch.cuda.current_device()}")
    apply_rotary_emb_qkv_(qkv, cos, sin)


# Strict-lower tril indices for n×n where n ≤ _TRIL_N_MAX. Calling
# `torch.tril_indices(n, n, -1)` per forward forces dynamo to specialize the
# graph on n (the meta op's output shape is data-dependent), so n drives
# recompiles. Instead we precompute up to N_MAX once per module and slice the
# leading `n*(n-1)//2` entries at call time — this works because the row-major
# enumeration emits the n-pair set as a strict prefix of the N_MAX-pair set.
_TRIL_N_MAX = 128


def _register_tril_buffers(module: nn.Module, n_max: int = _TRIL_N_MAX) -> None:
    i, j = torch.tril_indices(n_max, n_max, -1)
    module.register_buffer("_tril_i", i, persistent=False)
    module.register_buffer("_tril_j", j, persistent=False)


def _tril_slice(module: nn.Module, n: int):
    nC2 = n * (n - 1) // 2
    return module._tril_i[:nC2], module._tril_j[:nC2]


class Symmetric(nn.Module):
    def forward(self, X: Tensor):
        return (X + X.transpose(-1, -2)) / 2


class ColAttenPairBias(nn.Module):
    def __init__(
        self,
        msa_dim: int,
        pairs_dim: int,
        n_heads: int,
        symmetric: bool = False,
        mup: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = msa_dim
        self.pairs_dim = pairs_dim
        self.n_heads = n_heads
        self.symmetric = symmetric
        self.mup = mup

        self.head_dim = msa_dim // n_heads
        self.sqrt_dim = math.sqrt(self.head_dim)

        # Layer norms
        self.msa_norm = nn.LayerNorm(msa_dim)
        self.pairs_norm = nn.LayerNorm(pairs_dim)

        # Fused QKV projection: one matmul reads `msa` once instead of three,
        # saves two kernel launches per call, and outputs a wider GEMM that
        # uses tensor cores more efficiently. q/k/v keep bias=False semantics.
        # Weight layout: rows [0:d_m) = q, [d_m:2*d_m) = k, [2*d_m:3*d_m) = v.
        self.qkv_proj = nn.Linear(msa_dim, 3 * msa_dim, bias=False)
        self.g_proj = nn.Linear(msa_dim, msa_dim, bias=True)
        self.b_proj = nn.Linear(pairs_dim, n_heads, bias=False)
        if self.mup:
            # Alignment-aware: 1/sqrt(pairs_dim) extra factor on b_proj.
            with torch.no_grad():
                self.b_proj.weight.mul_(1.0 / math.sqrt(pairs_dim))

        # Output projector
        self.out_proj = nn.Linear(msa_dim, msa_dim)

        if symmetric:
            _register_tril_buffers(self)

        self.pre_qkdiv = lambda _: None
        self.pre_bias = lambda _: None
        self.pre_attout = lambda _: None
        self.pre_reshape = lambda _: None
        self.qkdiv = lambda _: None
        self.bias = lambda _: None
        self.attout = lambda _: None
        self.reshape = lambda _: None

    def register_hooks(self, prefix, memfile, steps, dtypes):
        self.pre_qkdiv = make_mem_hook(
            memfile, f"{prefix}.qkdiv", "pre_fwd", steps, dtypes
        )
        self.qkdiv = make_mem_hook(memfile, f"{prefix}.qkdiv", "fwd", steps, dtypes)
        self.pre_attout = make_mem_hook(
            memfile, f"{prefix}.attout", "pre_fwd", steps, dtypes
        )
        self.attout = make_mem_hook(memfile, f"{prefix}.attout", "fwd", steps, dtypes)
        self.pre_bias = make_mem_hook(
            memfile, f"{prefix}.bias", "pre_fwd", steps, dtypes
        )
        self.bias = make_mem_hook(memfile, f"{prefix}.bias", "fwd", steps, dtypes)
        self.pre_reshape = make_mem_hook(
            memfile, f"{prefix}.reshape", "pre_fwd", steps, dtypes
        )
        self.reshape = make_mem_hook(memfile, f"{prefix}.reshape", "fwd", steps, dtypes)

    def forward(self, msa, pairs):
        # MSA: [B,d_m,n,L]
        # Pairs: [B,n,n,d_p] (or [B,nC2,d_p] is symmetric)
        B, d_m, n, L = msa.size()
        H = self.n_heads
        d_h = self.head_dim

        # Normalize
        msa = self.msa_norm(msa.transpose(-1, -3))  # [B,L,n,d_m]
        pairs = self.pairs_norm(pairs)  # [B,n,n,d_p] ([B,nC2,d_p])

        # pair bias: [B,n,n,H] ([B,nC2,H])
        b = self.b_proj(pairs)

        # Fused QKV projection: [B,L,n,d_m] -> [B,L,n,3*d_m] in one matmul.
        # Reshape to [B,L,n,3,H,d_h] (free), permute to [3,B,L,H,n,d_h], then
        # one .contiguous() copy folds three previously-separate copies into a
        # single big sequential write. Final shape per stream: [B*L,H,n,d_h].
        qkv = self.qkv_proj(msa).reshape(B, L, n, 3, H, d_h)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5).contiguous().view(3, B * L, H, n, d_h)
        q, k, v = qkv.unbind(dim=0)

        # gating: [B,L,n,d_m] -> [B*L,H,n,d_h] (kept separate from QKV so g's
        # bias semantics and sigmoid stay isolated from the attention path).
        g = F.sigmoid(self.g_proj(msa).reshape(B, L, n, H, d_h).transpose(2, 3))
        g = g.reshape(B * L, H, n, d_h)

        self.pre_qkdiv(None)

        # Build additive attention bias
        self.pre_bias(None)
        if self.symmetric:
            # b: [B, nC2, H]
            ix, jx = _tril_slice(self, n)

            # build full bias matrix
            full_bias = torch.zeros(B, H, n, n, device=b.device, dtype=b.dtype)

            # move H forward: [B, H, nC2]
            b_t = b.transpose(1, 2)

            # fill lower triangle
            full_bias[:, :, ix, jx] = b_t

            # mirror to upper triangle
            full_bias[:, :, jx, ix] = b_t

            attn_bias = full_bias.unsqueeze(1).expand(B, L, H, n, n)
        else:
            # b: [B,n,n,H] → [B,H,n,n]
            attn_bias = b.permute(0, 3, 1, 2)
            attn_bias = attn_bias.unsqueeze(1).expand(B, L, H, n, n)

        # reshape to match (B*L, H, n, n)
        attn_bias = attn_bias.reshape(B * L, H, n, n)
        self.bias(None)

        # µP attention scale: 1/d_head (not the default 1/sqrt(d_head)) when
        # self.mup is on; otherwise fall through to PyTorch's default.
        # Inputs are already bf16 under bf16-true precision; no autocast needed.
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=False,
            scale=(1.0 / self.head_dim) if self.mup else None,
        )

        self.qkdiv(None)
        self.pre_attout(None)

        # Apply gating
        attn_out = g * attn_out

        self.attout(None)

        self.pre_reshape(None)
        attn_out = attn_out.reshape(B, L, H, n, d_h)
        attn_out = attn_out.transpose(2, 3).reshape(B, L, n, d_m)
        self.reshape(None)

        # [B,L,n,d_m]
        out = self.out_proj(attn_out)

        # [B,d_m,n,L]  .contiguous() keeps strides canonical so torch.compile
        # guards don't recompile on the stride pattern of the next block's msa.
        return out.permute(0, 3, 2, 1).contiguous()


def _rotary_freqs(seq_len, dim, device, base=10_000.0):
    """Compute (seq_len, dim/2) cos/sin for rotary embeddings."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


class RowAttenGated(nn.Module):
    def __init__(self, msa_dim: int, n_heads: int, global_token: bool = False,
                 use_rope: bool = True, use_layernorm: bool = True,
                 mup: bool = True, pf2_gating: bool = False) -> None:
        super().__init__()

        self.msa_dim = msa_dim
        self.n_heads = n_heads
        self.head_dim = msa_dim // n_heads
        self.has_global_token = global_token
        self.use_rope = use_rope
        self.mup = mup
        # Phyloformer 2's released code builds this block from nn.MultiheadAttention,
        # so heads are mixed by MHA's bias-free out-projection before the gate;
        # Algorithm 5 gates per head and mixes once.
        self.pf2_gating = pf2_gating
        self.mix_proj = nn.Linear(msa_dim, msa_dim, bias=False) if pf2_gating else None

        # Fused QKV projection: one matmul + one DRAM read of `r` instead of
        # three. Output shape [B*n, L, 3*d_m] reshapes for free into the
        # [B*n, L, 3, H, d_h] layout that flash_attn.apply_rotary_emb_qkv_
        # expects, so we also drop the previous `torch.stack([q,k,v], dim=2)`.
        self.qkv_proj = nn.Linear(msa_dim, 3 * msa_dim, bias=False)
        self.norm = nn.LayerNorm(msa_dim) if use_layernorm else nn.Identity()
        self.g_proj = nn.Linear(msa_dim, msa_dim)
        self.out_proj = nn.Linear(msa_dim, msa_dim)

        if pf2_gating:
            # MHA xavier-initializes its fused in_proj_weight; nn.Linear(d, 3d)
            # would be ~22% narrower.
            nn.init.xavier_uniform_(self.qkv_proj.weight)

    def forward(self, msa):
        B, d_m, n, L = msa.size()

        # [B,L,n,d_m] -> [B*n,L,d_m]
        msa = self.norm(msa.transpose(-1, -3))
        r = msa.transpose(-2, -3).reshape(B * n, L, d_m)
        g = F.sigmoid(self.g_proj(r))

        # Fused projection lands directly in flash_attn's expected layout —
        # no torch.stack copy needed.
        qkv = self.qkv_proj(r).view(B * n, L, 3, self.n_heads, self.head_dim)
        if self.use_rope:
            # Apply RoPE — skip last position if it's the global token
            n_rope = L - 1 if self.has_global_token else L
            cos, sin = _rotary_freqs(n_rope, self.head_dim, r.device)
            _apply_rope_no_trace(qkv[:, :n_rope], cos, sin)

        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # µP attention scale: 1/d_head (not the default 1/sqrt(d_head)) so logits stay
        # O(1) under feature-learning q/k alignment. Paper (µP off) uses the
        # default 1/sqrt(d_head), matching repo_pf2's nn.MultiheadAttention.
        attn_out = F.scaled_dot_product_attention(
            q, k, v, scale=(1.0 / self.head_dim) if self.mup else None
        )
        # Avoid the explicit `.transpose(1,2).reshape(B*n, L, d_m)` copy: leave
        # attn_out non-contiguous and let the mul kernel produce a contiguous
        # [B*n, L, H, d_h] output, which views to [B, n, L, d_m] for free.
        attn_out = attn_out.transpose(1, 2)  # [B*n, L, H, d_h] non-contig view
        if self.pf2_gating:
            attn_out = self.mix_proj(attn_out.reshape(B * n, L, d_m))
            attn_out = (g * attn_out).view(B, n, L, d_m)
        else:
            g_5d = g.view(B * n, L, self.n_heads, self.head_dim)
            attn_out = (g_5d * attn_out).view(B, n, L, d_m)

        out = self.out_proj(attn_out)
        return out.permute(0, 3, 1, 2).contiguous()  # [B, d_m, n, L]


class Transition(nn.Module):
    def __init__(self, dim: int, activation: str = "gelu", use_layernorm: bool = True) -> None:
        super().__init__()
        self.dim = dim

        self.norm = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.expand = nn.Linear(dim, dim * 4)
        self.act = _make_activation(activation)
        self.contract = nn.Linear(dim * 4, dim)

    def default_fwd(self, input):
        return self.contract(self.act(self.expand(self.norm(input))))

    def forward(self, input):
        return self.default_fwd(input)


class MSATransition(Transition):
    def __init__(self, dim: int, activation: str = "gelu") -> None:
        super().__init__(dim, activation=activation)

    def forward(self, input):
        # [B,d_m,n,L] -> [B,n,L,d_m]
        msa = input.transpose(-1, -3)
        msa = self.default_fwd(msa)

        # back to [B,d_m,n,L]
        return msa.transpose(-1, -3).contiguous()


class OuterProductMean(nn.Module):
    def __init__(
        self, msa_dim: int, pair_dim: int, inner_dim: int, symmetric: bool = False,
        use_layernorm: bool = True, mup: bool = True,
    ) -> None:
        super().__init__()
        self.symmetric = symmetric
        self.mup = mup
        self.norm = nn.LayerNorm(msa_dim) if use_layernorm else nn.Identity()
        self.a_proj = nn.Linear(msa_dim, inner_dim)
        if not symmetric:
            self.b_proj = nn.Linear(msa_dim, inner_dim)
        else:
            _register_tril_buffers(self)
        self.o_proj = nn.Linear(inner_dim**2, pair_dim)

        self.pre_einsum = lambda _: None
        self.einsum = lambda _: None
        self.pre_reshape = lambda _: None
        self.reshape = lambda _: None
        self.pre_indexab = lambda _: None
        self.indexab = lambda _: None

    def register_hooks(self, prefix, memfile, steps, dtypes):
        self.pre_einsum = make_mem_hook(
            memfile, f"{prefix}.einsum", "pre_fwd", steps, dtypes
        )
        self.einsum = make_mem_hook(memfile, f"{prefix}.einsum", "fwd", steps, dtypes)
        self.pre_reshape = make_mem_hook(
            memfile, f"{prefix}.reshape", "pre_fwd", steps, dtypes
        )
        self.reshape = make_mem_hook(memfile, f"{prefix}.reshape", "fwd", steps, dtypes)
        self.pre_indexab = make_mem_hook(
            memfile, f"{prefix}.indexab", "pre_fwd", steps, dtypes
        )
        self.indexab = make_mem_hook(memfile, f"{prefix}.indexab", "fwd", steps, dtypes)

    def forward(self, msa):
        # MSA: [B,d_m,n,L]
        _, _, n, L = msa.size()

        # [B,L,n,d_m]
        msa = self.norm(msa.transpose(-1, -3))

        # [B,n,L,d_in]
        a = self.a_proj(msa).transpose(-2, -3)

        self.pre_einsum(None)
        # Under µP, /sqrt(L) keeps the contraction sum_l a[l]·b[l] = O(sqrt(L))
        # at O(1) per entry. The paper (µP off) takes the mean over columns, /L.
        L_norm = math.sqrt(L) if self.mup else L
        if self.symmetric:
            # [B,n,n,d_in,d_in]
            outer_prod = torch.einsum("...nld,...NlD->...nNdD", a, a) / L_norm
            i, j = _tril_slice(self, n)
            # [B,nC2,d_in,d_in]
            outer_prod = outer_prod[:, i, j, :, :]
        else:
            # [B,n,n,d_in,d_in]
            b = self.b_proj(msa).transpose(-2, -3)
            outer_prod = torch.einsum("...nld,...NlD->...nNdD", a, b) / L_norm
        self.einsum(None)

        self.pre_reshape(None)
        # [B,n,n,d_in*d_in] (or [B,nC2,d_in*d_in])
        o = outer_prod.reshape(outer_prod.shape[:-2] + (-1,))
        self.reshape(None)

        return self.o_proj(o)


class ConcatenationPairUpdate(nn.Module):
    def __init__(
        self, msa_dim: int, pair_dim: int, inner_dim: int, symmetric: bool = False
    ) -> None:
        super().__init__()
        self.symmetric = symmetric
        self.norm = nn.LayerNorm(msa_dim)
        self.a_proj = nn.Linear(msa_dim, inner_dim)
        self.b_proj = nn.Linear(msa_dim, inner_dim)
        self.o_mlp = nn.Sequential(
            nn.Linear(2 * inner_dim, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, pair_dim),
        )
        if symmetric:
            _register_tril_buffers(self)

    def forward(self, msa):
        # MSA: [B,d_m,n,L]
        _, _, n, _ = msa.size()

        # [B,L,n,d_m]
        msa = self.norm(msa.transpose(-1, -3))

        # [B,n,d_in]
        a = self.a_proj(msa).mean(dim=1)
        b = self.b_proj(msa).mean(dim=1)

        if self.symmetric:
            ia, ib = _tril_slice(self, n)
            a = a[:, ia, :]
            b = b[:, ib, :]
        else:
            a = a[:, :, None, :]
            b = b[:, None, :, :]

        # [B,n,n,2*d_in]
        c = torch.cat([a + b, (a - b).abs()], dim=-1)

        # [B,n,n,d_p]
        out = self.o_mlp(c)

        return out


class PairAttention(nn.Module):
    def __init__(
        self,
        pair_dim: int,
        n_heads: int,
        symmetric: bool = False,
        mup: bool = True,
        pf2_gating: bool = False,
    ) -> None:
        super().__init__()
        self.pair_dim = pair_dim
        self.n_heads = n_heads
        self.symmetric = symmetric
        self.mup = mup
        # Phyloformer 2's released code uses a raw linear gate here; Algorithm 8
        # specifies a sigmoid.
        self.pf2_gating = pf2_gating

        self.norm = nn.LayerNorm(pair_dim)
        self.mha = nn.MultiheadAttention(
            pair_dim, n_heads, batch_first=True, bias=False
        )
        # µP attention scale: nn.MultiheadAttention applies 1/sqrt(d_head) inside,
        # so dividing K by sqrt(d_head) gives an effective scale of 1/d_head, matching
        # the µP convention used by the other attention modules in this stack.
        self._k_scale = (1.0 / math.sqrt(pair_dim // n_heads)) if mup else 1.0
        self.g_proj = nn.Linear(pair_dim, pair_dim)
        self.out_proj = nn.Linear(pair_dim, pair_dim)

    def forward_sym(self, pairs):
        # Pairs: [B,nC2,d_p]

        r = self.norm(pairs)
        g = self.g_proj(r) if self.pf2_gating else F.sigmoid(self.g_proj(r))
        k = r * self._k_scale if self.mup else r
        out, _ = self.mha(r, k, r, need_weights=False)
        out = g * out

        return self.out_proj(out)

    def forward_assym(self, pairs):
        # Pairs: [B,n,n,d_p]
        B, n, _, d_p = pairs.size()

        r = self.norm(pairs).reshape(B * n, n, d_p)
        g = self.g_proj(r) if self.pf2_gating else F.sigmoid(self.g_proj(r))
        k = r * self._k_scale if self.mup else r
        out, _ = self.mha(r, k, r, need_weights=False)
        out = (g * out).reshape(B, n, n, d_p)

        return self.out_proj(out)

    def forward(self, pairs):
        if self.symmetric:
            # [B,nC2,d_p]
            return self.forward_sym(pairs)
        else:
            # [B,n,n,d_p]
            return self.forward_assym(pairs)


class EvoPFBlock(nn.Module):
    def __init__(
        self,
        msa_dim: int,
        pair_dim: int,
        n_heads: int,
        use_opm: bool = True,
        symmetric: bool = False,
        use_rope: bool = True,
        n_blocks: int = 1,
        mup: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.msa_dim = msa_dim
        self.pairs_dim = pair_dim
        self.n_heads = n_heads
        self.use_opm = use_opm
        # CompleteP residual-branch scaling: each branch contributes 1/n_blocks
        # of the carry when mup is on; otherwise full-strength residual adds.
        self.beta = (1.0 / max(n_blocks, 1)) if mup else 1.0

        # EvoPF blocks are only ever built by the Phyloformer 2 baseline (every
        # model of ours runs species_encoder.n_blocks=0), so they take the gating
        # of Phyloformer 2's released code. RowAttenGated and PairAttention are
        # also constructed directly by our SPD trunk and covariance heads, which
        # follow the published algorithms -- hence a flag rather than a rewrite.

        # MSA stack
        self.row_atten = RowAttenGated(msa_dim, n_heads, use_rope=use_rope, mup=mup,
                                       pf2_gating=True)
        self.col_atten = ColAttenPairBias(msa_dim, pair_dim, n_heads, symmetric, mup=mup)
        self.msa_trans = MSATransition(msa_dim, activation=activation)

        # Pair stack
        if self.use_opm:
            self.pair_update = OuterProductMean(msa_dim, pair_dim, 32, symmetric, mup=mup)
        else:
            self.pair_update = ConcatenationPairUpdate(msa_dim, pair_dim, 32, symmetric)
        self.pair_atten = PairAttention(pair_dim, n_heads, symmetric, mup=mup,
                                        pf2_gating=True)
        self.pair_trans = Transition(pair_dim, activation=activation)

    def forward(self, msa, pairs):
        # MSA: [B,d_m,n,L]
        # Pairs: [B,n,n,d_p]
        beta = self.beta

        ## MSA STACK
        msa = self.col_atten(msa, pairs) * beta + msa
        msa = self.row_atten(msa) * beta + msa
        msa = self.msa_trans(msa) * beta + msa

        ## PAIR STACK
        pairs = self.pair_update(msa) * beta + pairs
        pairs = self.pair_atten(pairs) * beta + pairs
        pairs = self.pair_trans(pairs) * beta + pairs

        return msa, pairs


class PairEmbedder(nn.Module):
    def __init__(self, vocab_size, pair_dim: int, symmetric: bool = False,
                 activation: str = "gelu") -> None:
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Conv2d(
                in_channels=vocab_size, out_channels=pair_dim, kernel_size=1, stride=1
            ),
            _make_activation(activation),
        )
        self.symmetric = symmetric

    def forward_sym(self, msa):
        msa_emb = self.embedder(msa)  # [b,d_p,L,n]
        s2p = seq2pairs(msa.size(-1), lower=True, diff=False).to(msa)  # [nC2,n]
        pair_sums = msa_emb @ s2p.T  # [b,d_p,L,nC2]
        pairs = pair_sums.mean(dim=-2)  # [b,d_p,nC2]

        return pairs.permute(0, 2, 1).contiguous()  # [d,nC2,d_p]

    def forward_assym(self, msa):
        msa_emb = self.embedder(msa)  # [b,d_p,L,n]
        outer_sum = msa_emb.unsqueeze(-1) + msa_emb.unsqueeze(-2)  # [b,d_p,L,n,n]
        pairs = outer_sum.mean(dim=-3)  # [b,d_p,n,n]

        return pairs.permute(0, 2, 3, 1).contiguous()  # [b,n,n,d_p]

    def forward(self, msa):
        if self.symmetric:
            # [d,nC2,d_p]
            return self.forward_sym(msa)
        else:
            # [b,n,n,d_p]
            return self.forward_assym(msa)


class DistanceMLP(nn.Module):
    def __init__(self, pair_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pair_dim, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, 1),
            nn.Softplus(),
        )

    def forward(self, pairs):
        # N = (n,n) or nC2
        # [b,N,d_p] -> [b,N,1]
        return self.mlp(pairs)


class ParamsFromPair(nn.Module):
    def __init__(
        self,
        d_msa: int,
        n_params: int,
        symmetric: bool = False,
        pos: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.pos = pos
        self.eps = eps
        self.bil = nn.Bilinear(
            in1_features=d_msa, in2_features=d_msa, out_features=n_params
        )
        if symmetric:
            parametrize.register_parametrization(self.bil, "weight", Symmetric())

    def forward(self, seq1: Tensor, seq2: Tensor) -> Tensor:
        out = torch.einsum('...j,ijk,...k->...i', seq1, self.bil.weight, seq2) + self.bil.bias
        if self.pos:
            return F.softplus(out) + self.eps
        return out


class QSampler(nn.Module):
    def __init__(self, h_dim: int):
        super().__init__()
        self.h_dim = h_dim
        self.bil = ParamsFromPair(h_dim, 1, pos=False, symmetric=True)
        nn.init.xavier_uniform_(self.bil.bil.parametrizations.weight.original)
        nn.init.constant_(self.bil.bil.bias, 0.0)

    def forward(self, seqs1: Tensor, seqs2: Tensor):
        return self.bil(seqs1, seqs2)


class EvoPF(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_blocks: int = 6,
        n_heads: int = 4,
        h_dim: int = 64,
        pair_dim: int = 256,
        use_opm: bool = True,
        distance_mlp: bool = False,
        symmetric: bool = False,
        use_rope: bool = True,
        mup: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        self.symmetric = symmetric
        self.msa_embedder = MSAEmbedder(vocab_size, h_dim, activation=activation, mup=mup)
        self.pair_embedder = PairEmbedder(vocab_size, pair_dim, symmetric,
                                          activation=activation)
        self.evoblocks = nn.ModuleList(
            [
                EvoPFBlock(
                    h_dim,
                    pair_dim,
                    n_heads,
                    use_opm,
                    symmetric,
                    use_rope=use_rope,
                    n_blocks=n_blocks,
                    mup=mup,
                    activation=activation,
                )
                for _ in range(n_blocks)
            ]
        )

        self.pairs_to_dm = (
            DistanceMLP(pair_dim) if distance_mlp else nn.Linear(pair_dim, 1)
        )

        self.e_norm = nn.GroupNorm(8, h_dim)
        self.parent_embedder = ParamsFromPair(h_dim, h_dim, pos=False, symmetric=True)
        self.q_embedder = QSampler(h_dim)

    def embed_parent_node(self, left: Tensor, right: Tensor):
        # [B,d_m] x 2 -> [B,d_m]
        return self.parent_embedder(left, right)

    def embed_input(self, input: Tensor):
        # [b,d_m,n,L]
        msa_emb = self.msa_embedder(input)
        # [b,n,n,d_p]
        pairs = self.pair_embedder(input)

        return msa_emb, pairs

    def forward(self, input, pool=True):
        msa, pairs = self.embed_input(input)

        for block in self.evoblocks:
            msa, pairs = block(msa, pairs)

        # Compute distances
        dm = self.pairs_to_dm(pairs).squeeze(-1)
        if not self.symmetric:
            dm = (dm + dm.transpose(-1, -2)) / 2.0

        if pool:
            msa_emb = msa.mean(-1)
        else:
            msa_emb = msa  # [batch, h_dim, n_species, seq_len]
        return dm, msa_emb
