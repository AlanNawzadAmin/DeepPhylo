import dgl
import torch

from ..spd_attn import compute_spd_nodes


def get_hosts(graph, leaf_idx, leaf_idx_within_tree):
    node_to_tree_id = torch.repeat_interleave(
        torch.arange(graph.batch_size, device=graph.device), graph.batch_num_nodes()
    )
    leaf_to_tree_id = node_to_tree_id[leaf_idx]
    tree_match_mask = node_to_tree_id[:, None] == leaf_to_tree_id[None, :]
    host_match_mask = tree_match_mask & graph.ndata["leafset"][
        :, leaf_idx_within_tree
    ].eq(0)
    assert host_match_mask.sum(dim=0).eq(1).all()
    host_idx = host_match_mask.max(dim=0).indices
    return host_idx


def batch_spd_matrix(
    spd_nodes_list, n_leaves_per_tree, n_obs_per_graph, host_idx_in_obs, leaf_to_tree_id
):
    """Construct batched SPD matrix for attention from list of per-graph SPD matrices."""
    device = n_obs_per_graph.device
    batch_size = len(n_leaves_per_tree)
    spd_dim = max(n_leaves_per_tree + n_obs_per_graph)
    spd = torch.full((batch_size, spd_dim, spd_dim), -1, dtype=torch.long, device=device)
    offset = torch.cat([torch.tensor([0], device=device), n_obs_per_graph.cumsum(0)[:-1]])
    host_idx_in_obs_b = torch.split(
        host_idx_in_obs - offset[leaf_to_tree_id], n_leaves_per_tree.tolist()
    )
    for i, s in enumerate(spd_nodes_list):
        nl = n_leaves_per_tree[i]
        no = n_obs_per_graph[i]
        h = host_idx_in_obs_b[i]
        spd[i, nl : nl + no, nl : nl + no] = s
        spd[i, :nl, nl : nl + no] = s[h]
        spd[i, nl : nl + no, :nl] = s[:, h]
        spd[i, :nl, :nl] = s[h][:, h]
    return spd


def build_obs_graph_aux(
    obs_graphs, tree_to_n_leaves, n_obs_per_graph, host_idx_in_obs, leaf_to_tree_id
):
    spd_nodes_list = [compute_spd_nodes(g) for g in obs_graphs]
    spd = batch_spd_matrix(
        spd_nodes_list,
        tree_to_n_leaves,
        n_obs_per_graph,
        host_idx_in_obs,
        leaf_to_tree_id,
    )
    obs_graph = dgl.batch(obs_graphs)
    left_child, right_child = get_child_lookups(obs_graph)
    no_observed_children = (left_child == -1) & (right_child == -1)
    return spd, no_observed_children, left_child, right_child


def compute_depths(graph):
    rev_graph = dgl.reverse(graph)
    rev_graph.ndata["d"] = rev_graph.in_degrees().eq(0).long()
    dgl.prop_nodes_topo(
        rev_graph,
        message_func=lambda e: {"m": e.src["d"]},
        reduce_func=lambda n: {"d": torch.sum(n.mailbox["m"], dim=1) + 1},
    )
    return rev_graph.ndata["d"] - 1


def compute_target(graph, hosts, leaf_id_in_graph):
    left_children, right_children = get_child_lookups(graph)
    left_children = left_children[hosts]
    right_children = right_children[hosts]
    should_go_left = graph.ndata["leafset"][left_children, leaf_id_in_graph].ge(0) & (
        left_children != -1
    )
    should_go_right = graph.ndata["leafset"][right_children, leaf_id_in_graph].ge(0) & (
        right_children != -1
    )
    no_observed_children = (
        ~graph.ndata["observed"][left_children]
        & ~graph.ndata["observed"][right_children]
        & (left_children != -1)
        & (right_children != -1)
    )

    current_depths = graph.ndata["d"][hosts]
    true_depths = graph.ndata["d"][graph.in_degrees().eq(0)]
    true_delta_depth = (true_depths - current_depths).float()
    target = torch.zeros((hosts.shape[0], 2), device=graph.device)
    target[should_go_left, 0] = true_delta_depth[should_go_left]
    target[should_go_right, 1] = true_delta_depth[should_go_right]
    target[no_observed_children] = (
        true_delta_depth[no_observed_children].unsqueeze(-1) / 2
    )
    return target


def get_child_lookups(graph):
    children, parents = graph.edges()
    directions = graph.edata["direction"]
    left_children = torch.full(
        (graph.num_nodes(),), -1, device=graph.device, dtype=torch.long
    )
    left_children[parents[directions == 0]] = children[directions == 0]
    right_children = torch.full(
        (graph.num_nodes(),), -1, device=graph.device, dtype=torch.long
    )
    right_children[parents[directions == 1]] = children[directions == 1]
    return left_children, right_children
