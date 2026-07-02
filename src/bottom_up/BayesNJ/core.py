import torch
import torch.nn.functional as F


def argmin(x, axis=-1):
    return F.one_hot(torch.argmin(x, dim=axis), list(x.shape)[axis]).float()


def argmax(x, axis=-1):
    return F.one_hot(torch.argmax(x, dim=axis), list(x.shape)[axis]).float()


def batch_mask_from_ignored(ignored):
    return torch.matmul(
        (~ignored[:, :, None]).float(), (~ignored[:, None, :]).float()
    ).bool()


def batch_init_NJ(embs, rooted: bool = False):
    device = embs.device
    batch_size, _, n_leaves = embs.shape
    max_nodes = 2 * n_leaves - 1

    # Splits matrix: (batch, max_nodes, n_leaves)
    splits = torch.zeros(batch_size, max_nodes, n_leaves, device=device)
    splits[:, :n_leaves, :n_leaves] = torch.eye(
        n_leaves, device=device, dtype=torch.float
    )

    # Branch lengths: (batch, max_nodes)
    brlens = torch.zeros(batch_size, max_nodes, device=device, dtype=torch.float)

    # Mask to track ignored nodes: True = ignored, False = active
    ignored_nodes = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=device)
    ignored_nodes[:, :n_leaves] = False

    return None, splits, brlens, ignored_nodes, n_leaves, max_nodes


def batch_sample_Q(model, embs: torch.Tensor, ignored):
    batch_size, n_nodes, _ = embs.shape
    Q = torch.zeros(batch_size, n_nodes, n_nodes).to(embs)
    ix, jx = torch.tril_indices(n_nodes, n_nodes, -1, device=embs.device)
    mask_2d = batch_mask_from_ignored(ignored)

    # Compute Q matrix
    vals = model.q_embedder(embs[:, ix, :], embs[:, jx, :])
    Q[:, ix, jx] = vals.squeeze(-1)
    Q = Q + Q.transpose(-1, -2)

    # Mask ignored ignored_nodes
    mask = (~mask_2d) | torch.eye(n_nodes, device=mask_2d.device).bool()
    Q = Q.masked_fill(mask, torch.inf)

    return Q
