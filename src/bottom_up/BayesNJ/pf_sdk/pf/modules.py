import torch  # type:ignore
from torch import nn


def _make_activation(name: str) -> nn.Module:
    """Resolve 'gelu' or 'relu' to the corresponding nn module. Used to swap
    EvoPF's transition/embedder activations between code default (GELU) and
    the activation used in the paper's Algorithms S.2/S.3/S.6/S.9 (ReLU)."""
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name!r} (expected 'gelu' or 'relu')")


@torch.compiler.disable
def seq2pairs(n, lower=False, diff=False):
    """
    Gives a matrix which will transform from sequence space to pair space:

     - If lower=True, then the order of the pairs corresponds to the order
     of the lower triangular vector of the pairwise matrix. By default the
     order corresponds to the upper triangular vector

     - If diff=True, then multiplying this matrix with the MSA embedding
     will yield the difference between embeddings instead of the sum by
     default (e.g. useful if computing pairwise distances)
    """
    n_pairs = n * (n - 1) // 2
    seq2pair = torch.zeros(n_pairs, n)

    # Compute where to put 1
    row_idx = torch.arange(n_pairs).long()

    idx = torch.triu_indices(n, n, offset=1)
    if lower:
        idx = torch.tril_indices(n, n, offset=-1)

    # Define fill values
    fill = torch.Tensor([1])
    fill2 = fill * (-1 if diff else 1)

    # Fill the matrix and return
    seq2pair = seq2pair.index_put((row_idx, idx[0]), fill)
    return seq2pair.index_put((row_idx, idx[1]), fill2)


class MSAEmbedder(nn.Module):
    def __init__(self, vocab_size, embed_dim: int, activation: str = "gelu",
                 mup: bool = True) -> None:
        super().__init__()
        self.mup = mup
        self.act = _make_activation(activation)
        if mup:
            # A 1x1 conv over a one-hot channel axis is exactly an embedding
            # lookup, so under µP we use a genuine nn.Embedding: O(1) init
            # (N(0,1)) and the µP optimizer treats it correctly (lr_mult=1, eps
            # via embed_dim) instead of letting a Conv2d weight fall through to
            # lr_mult=1/eps with d_out=1.
            self.embed = nn.Embedding(vocab_size, embed_dim)
        else:
            # Paper-exact (Algorithm S.2): 1x1 conv over the one-hot vocab axis,
            # with default Conv2d init (incl. bias). Matches Phyloformer 2's
            # released code and the paper when µP is off.
            self.conv = nn.Conv2d(vocab_size, embed_dim, kernel_size=1, stride=1)

    def forward(self, input):
        # input is one-hot [B, vocab, L, n] (kept for interface compatibility
        # with pair_embedder, which consumes the same tensor).
        if self.mup:
            # Recover token indices along the vocab/channel axis.
            idx = input.argmax(dim=1)            # [B, L, n]
            out = self.act(self.embed(idx))      # [B, L, n, D]
            # [B,D,n,L]  .contiguous() normalizes strides so torch.compile
            # guards don't fire on stride variation as L and n change.
            return out.permute(0, 3, 2, 1).contiguous()

        # Conv2d path: [B, vocab, L, n] -> [B, D, L, n] -> [B, D, n, L]
        out = self.act(self.conv(input))         # [B, D, L, n]
        return out.transpose(-1, -2).contiguous()
