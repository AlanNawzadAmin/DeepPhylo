import math
from io import StringIO

import dgl
import networkx as nx
import torch
import torch.distributed as dist
from Bio import Phylo
from ete3 import Tree
from torch.utils.data import IterableDataset, get_worker_info

from ..sampling_utils import ensure_newick_semicolon, sample_n_leaves_weighted
from .random_tree import get_random_tree
from .simulate_ali import simulate_alignment


def _log_double_factorial_weights(n_min: int, n_max: int) -> torch.Tensor:
    """Compute ln((2N-3)!!) = sum_{k=4}^{N} ln(2k-5) for each N in [n_min, n_max].

    Returns a tensor of length (n_max - n_min + 1) with the weights.
    Values for N <= 3 are clamped to 0 (single topology, no difficulty).
    """
    weights = []
    for n in range(n_min, n_max + 1):
        if n <= 3:
            weights.append(0.0)
        else:
            weights.append(sum(math.log(2 * k - 5) for k in range(4, n + 1)))
    return torch.tensor(weights)


def newick_to_dgl(newick_str):
    tree = Phylo.read(StringIO(newick_str), "newick")
    nx_graph = Phylo.to_networkx(tree)
    nx_int = nx.convert_node_labels_to_integers(nx_graph, label_attribute="clade")
    leaf_map = {}
    for node_id, data in nx_int.nodes(data=True):
        clade = data["clade"]
        if clade.is_terminal():
            leaf_map[str(clade.name)] = node_id
    return dgl.graph(list(nx_int.edges())).reverse(), leaf_map


def ete3_and_seqs_to_dgl(ete_tree, seqs, feature_width=None):
    newick = ensure_newick_semicolon(ete_tree.write(format=9))
    graph, leaf_map = newick_to_dgl(newick)
    n_leaves = seqs.shape[0]
    seq_width = seqs.shape[1]
    width = seq_width if feature_width is None else int(feature_width)
    feat = torch.zeros(graph.num_nodes(), width, dtype=seqs.dtype)
    inds = [leaf_map[str(i)] for i in range(n_leaves)]
    feat[inds, :seq_width] = seqs
    graph.ndata["x"] = feat

    leaf_name = torch.full((graph.num_nodes(),), -1, dtype=torch.long)
    leaf_name[inds] = torch.arange(n_leaves, dtype=torch.long)
    graph.ndata["leaf_name"] = leaf_name
    return graph


def label_tree_nodes(ete_tree, taxa_names):
    name_to_idx = {name: i for i, name in enumerate(taxa_names)}
    internal_idx = len(taxa_names)
    for node in ete_tree.traverse():
        if node.name in name_to_idx:
            node.name = int(name_to_idx[node.name])
        else:
            node.name = int(internal_idx)
            internal_idx += 1


class SimulatedDataset(IterableDataset):
    def __init__(self, cfg, n_min, n_max, d_min, d_max, size=None, seed=0, d_skew=1,
                 weight_n_by_difficulty=False, d_target=None):
        super().__init__()
        self.cfg = cfg
        self.n_min = n_min
        self.n_max = n_max
        self.d_min = d_min
        self.d_max = d_max
        self.size = None if size is None else int(size)
        self.seed = int(seed)
        self.d_skew = d_skew
        self.weight_n_by_difficulty = weight_n_by_difficulty
        # If set, force the simulator's target length to this value (bypassing
        # the [d_min, d_max] sample). `d_max` still bounds acceptance so indel
        # expansion up to the true max is allowed.
        self.d_target = d_target

    def __iter__(self):
        info = get_worker_info()
        num_workers, worker_id = (info.num_workers, info.id) if info else (1, 0)
        if dist.is_available() and dist.is_initialized():
            rank, world_size = dist.get_rank(), dist.get_world_size()
        else:
            rank, world_size = 0, 1
        total_shards = world_size * num_workers
        shard_id = rank * num_workers + worker_id
        per_shard = None if self.size is None else (self.size + total_shards - 1) // total_shards

        idx = shard_id
        produced = 0
        while per_shard is None or produced < per_shard:
            shape_seed = self.seed + produced * num_workers + worker_id
            yield self._sample(self.seed + idx, shape_seed)
            idx += total_shards
            produced += 1

    def _sample(self, seed, shape_seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        shape_gen = torch.Generator()
        shape_gen.manual_seed(shape_seed)
        if self.weight_n_by_difficulty:
            n_leaves, n_is_weight = sample_n_leaves_weighted(
                self.n_min, self.n_max, generator=shape_gen
            )
        else:
            n_leaves = torch.randint(
                self.n_min, self.n_max + 1, (1,), generator=shape_gen
            ).item()
            n_is_weight = 1.0
        if self.d_target is not None:
            d = int(self.d_target)
        else:
            d = int(
                (
                    self.d_min
                    + (self.d_max - self.d_min)
                    * (torch.rand(1, generator=shape_gen) ** (self.d_skew))
                ).floor()
            )
        while True:
            tree = get_random_tree(self.cfg.tree, n_leaves, seed=seed)
            seqs, taxa_names = simulate_alignment(tree, self.cfg.ali, d, seed=seed)
            if not isinstance(seqs, torch.Tensor):
                seqs = torch.tensor(seqs, dtype=torch.float32)
            if seqs.shape[1] < self.d_min or seqs.shape[1] > self.d_max:
                seed = seed + 1  # Sadly this risks repeating training data
                print("Retrying")
                continue

            perm = torch.randperm(len(taxa_names), generator=generator)
            seqs = seqs[perm]
            taxa_names = [taxa_names[i] for i in perm.tolist()]

            ete_tree = Tree(tree)
            label_tree_nodes(ete_tree, taxa_names)
            return ete_tree, seqs, n_is_weight
