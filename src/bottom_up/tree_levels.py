import dgl
import torch


def precompute_tree_levels(graph: dgl.DGLGraph):
    """Group internal nodes by tree level with their children indices.

    Returns a list of (node_ids, left_child_ids, right_child_ids) tuples,
    ordered bottom-up. All tensors are on CPU (long).
    """
    n_leaves = int(graph.in_degrees().eq(0).sum().item())
    levels = []
    for nodes in dgl.topological_nodes_generator(graph):
        internal = nodes[nodes >= n_leaves]
        if internal.numel() == 0:
            continue
        left_ids = []
        right_ids = []
        for nid in internal.tolist():
            children = graph.predecessors(nid).sort()[0]
            left_ids.append(children[0].item())
            right_ids.append(children[1].item())
        levels.append((
            internal.long(),
            torch.tensor(left_ids, dtype=torch.long),
            torch.tensor(right_ids, dtype=torch.long),
        ))
    return levels
