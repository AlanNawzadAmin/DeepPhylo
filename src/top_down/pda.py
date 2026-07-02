"""PDA (beta-splitting) probability computations."""

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=4096)
def _pda_denominator_term_scalar(L, M, beta=-1.5):
    out = 0.0
    for k in range(max(M - 1, 0)):
        prod = 1.0
        for l in range(k + 1):
            prod *= (L + beta + 1 + l) / (L + 2 * beta + 3 + l)
        out += prod
    return out


def _pda_denominator_term_tensor(A, M, beta=-1.5):
    if A.numel() == 0:
        return torch.empty_like(A, dtype=torch.float)
    pairs = torch.stack((A.long(), M.long()), dim=1)
    unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    values = torch.tensor(
        [
            _pda_denominator_term_scalar(int(a.item()), int(m.item()), beta=beta)
            for a, m in unique_pairs
        ],
        device=A.device,
        dtype=torch.float,
    )
    return values[inverse]


def _pda_left_prob_from_counts(L, R, M, beta=-1.5):
    p_left = torch.empty_like(L, dtype=torch.float)

    both_zero = (L == 0) & (R == 0)
    p_left[both_zero] = 0.5

    right_zero = (R == 0) & ~both_zero
    if right_zero.any():
        denom = _pda_denominator_term_tensor(L[right_zero], M[right_zero], beta=beta)
        p_left[right_zero] = 1 - 1 / (1 + denom)

    left_zero = (L == 0) & ~both_zero
    if left_zero.any():
        denom = _pda_denominator_term_tensor(R[left_zero], M[left_zero], beta=beta)
        p_left[left_zero] = 1 / (1 + denom)

    generic = ~(both_zero | right_zero | left_zero)
    if generic.any():
        Lg = L[generic].float()
        Rg = R[generic].float()
        p_left[generic] = (Lg + beta + 1) / (Lg + Rg + 2 * beta + 2)

    return p_left


def _neg_log_split_prob(L, R, beta):
    """
    -log p(L, R) for the beta-splitting model.
    For beta=-3/2 (PDA), b1=-0.5: lgamma(L-0.5), lgamma(R-0.5) are fine for L,R >= 1.
    The normalisation alpha_N = B(b1,b1) - 2*B(N+b1,b1) is computed carefully.
    """
    b1 = beta + 1
    device = L.device
    b1t = L.new_full((), b1)

    log_num = torch.lgamma(L + b1) + torch.lgamma(R + b1) - torch.lgamma(L + R + 2 * b1)

    N = L + R
    if b1 > 0:
        log_B_b1b1 = 2 * torch.lgamma(b1t) - torch.lgamma(2 * b1t)
        log_B_Nb1 = torch.lgamma(N + b1) + torch.lgamma(b1t) - torch.lgamma(N + 2 * b1)
        log_alpha = log_B_b1b1 + torch.log1p(-2 * torch.exp(log_B_Nb1 - log_B_b1b1))
    else:
        log_abs_g_b1 = torch.log(torch.tensor(2.0, device=device)) + torch.lgamma(torch.tensor(0.5, device=device))
        log_alpha = (
            torch.log(torch.tensor(2.0, device=device))
            + torch.lgamma(N + b1)
            + log_abs_g_b1
            - torch.lgamma(N - 1.0)
        )

    return -(log_num - log_alpha)


def _compute_qA_table(sA_grid, sB_grid, M_grid, valid, beta=-1.5):
    """Vectorised q_A(s_A, s_B) over an (N x L+1) grid."""
    qA = torch.full_like(sA_grid, 0.5, dtype=torch.float64)
    if not valid.any():
        return qA

    sA = sA_grid[valid].double()
    sB = sB_grid[valid].double()
    M = M_grid[valid].double()

    generic = (sA > 0) & (sB > 0)
    if generic.any():
        qA_flat = qA[valid]
        qA_flat[generic] = (sA[generic] + beta + 1) / (sA[generic] + sB[generic] + 2 * beta + 2)
        qA[valid] = qA_flat

    sB_zero = (sB == 0) & (sA > 0)
    if sB_zero.any():
        sA_bz = sA[sB_zero].long().cpu().tolist()
        M_bz = M[sB_zero].long().cpu().tolist()
        phi_vals = torch.tensor(
            [1.0 - 1.0 / (1.0 + _pda_denominator_term_scalar(si, mi, beta))
             for si, mi in zip(sA_bz, M_bz)],
            dtype=torch.float64,
        )
        qA_flat = qA[valid]
        qA_flat[sB_zero] = phi_vals.to(qA_flat.device)
        qA[valid] = qA_flat

    sA_zero = (sA == 0) & (sB > 0)
    if sA_zero.any():
        sB_az = sB[sA_zero].long().cpu().tolist()
        M_az = M[sA_zero].long().cpu().tolist()
        phi_vals = torch.tensor(
            [1.0 / (1.0 + _pda_denominator_term_scalar(si, mi, beta))
             for si, mi in zip(sB_az, M_az)],
            dtype=torch.float64,
        )
        qA_flat = qA[valid]
        qA_flat[sA_zero] = phi_vals.to(qA_flat.device)
        qA[valid] = qA_flat

    return qA


def _compute_qA_scalar(sA, sB, M, beta=-1.5):
    """Scalar q_A(s_A, s_B): PDA predictive probability for left."""
    if sA == 0 and sB == 0:
        return 0.5
    if sA > 0 and sB > 0:
        return (sA + beta + 1) / (sA + sB + 2 * beta + 2)
    if sB == 0:
        denom = _pda_denominator_term_scalar(sA, M, beta)
        return 1.0 - 1.0 / (1.0 + denom)
    denom = _pda_denominator_term_scalar(sB, M, beta)
    return 1.0 / (1.0 + denom)


def _compute_I_scalar(a, b, L, R, M, beta=-1.5, eps=1e-6):
    """Scalar per-element KL  I(a, b)."""
    assert a + b == M
    if M == 0 or (a == L and b == R):
        return 0.0
    sA, sB = L - a, R - b
    qA = _compute_qA_scalar(sA, sB, M, beta)
    qA = max(eps, min(1.0 - eps, qA))
    val = 0.0
    if a > 0:
        val += a / M * math.log(1.0 / qA)
    if b > 0:
        val += b / M * math.log(1.0 / (1.0 - qA))
    return val
