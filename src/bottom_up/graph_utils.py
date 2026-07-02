from itertools import combinations

import dgl
import torch


def ordered_exposed_nodes(graph):
    exposed_nodes = torch.where(graph.out_degrees().eq(0))[0]
    return exposed_nodes[torch.argsort(graph.ndata[dgl.NID][exposed_nodes])]


def ordered_candidate_pairs(graph):
    exposed_nodes = ordered_exposed_nodes(graph)
    return [tuple(pair) for pair in combinations(exposed_nodes.tolist(), 2)]
