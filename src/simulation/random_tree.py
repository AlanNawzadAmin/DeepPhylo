import random

import numpy as np
from dendropy import TaxonNamespace
from dendropy.simulate import treesim


def apply_lognormal(tree, mean_rate=1, log_sd=0.3, auto_correlated=False, rng=None):
    """multiplies branch lengths by "relaxed clock" rates"""
    tree.seed_node.rate = mean_rate
    for node in tree.preorder_node_iter():
        if node.parent_node is not None:
            if auto_correlated:
                parent_rate = np.log(node.parent_node.rate)
            else:
                parent_rate = (
                    np.log(mean_rate) - (log_sd**2) / 2
                )  # so that E[rate] = mean_rate
            node.rate = rng.lognormal(parent_rate, log_sd)
            node.edge.length = node.edge.length * node.rate
    return tree


def sample_pda_newick(n, rate=1.0, rng=None):
    """Ford alpha = 1/2 with exp(rate) branch lengths"""
    tree = ["t0", "t1"]

    def get_edges(t, parent=None, idx=None):
        """Returns list of (parent, idx, subtree) for each edge: parent -> subtree"""
        edges = [(parent, idx, t)]
        if isinstance(t, list):
            edges += get_edges(t[0], t, 0)
            edges += get_edges(t[1], t, 1)
        return edges

    for i in range(2, n):
        edges = get_edges(tree)
        parent, idx, subtree = rng.choice(edges)
        new_node = [subtree, f"t{i}"]  # replace subtree with this
        if parent is None:
            tree = new_node
        else:
            parent[idx] = new_node

    def to_newick(t, first=False, rng=rng):
        if isinstance(t, str):
            return t
        if first:
            l1 = rng.expovariate(rate) / 2
            l2 = l1
        else:
            l1 = rng.expovariate(rate)
            l2 = rng.expovariate(rate)
        return f"({to_newick(t[0])}:{l1:.6f},{to_newick(t[1])}:{l2:.6f})"

    return to_newick(tree, first=True, rng=rng) + ";"


def get_random_tree(cfg, n_leaves, seed=None):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if (
        cfg.name == "birth_death"
    ):  # rejection samples until it gets to N species; not CPP
        tree = treesim.birth_death_tree(
            birth_rate=cfg.hyper.birth,
            death_rate=cfg.hyper.death,
            num_extant_tips=n_leaves,
            rng=rng,
        )
        tree = apply_lognormal(
            tree, 1, cfg.branch.std, cfg.branch.auto_correlate, rng=np_rng
        )
        tree = tree.as_string(schema="newick", suppress_rooting=True)
    elif cfg.name == "kingman":
        tree = treesim.pure_kingman_tree(
            TaxonNamespace(np.arange(n_leaves)), pop_size=cfg.hyper.pop_size, rng=rng
        )
        tree = apply_lognormal(
            tree, 1, cfg.branch.std, cfg.branch.auto_correlate, rng=np_rng
        )
        tree = tree.as_string(schema="newick", suppress_rooting=True)
    elif cfg.name == "pda":
        rate = cfg.branch.rate * (1 if not cfg.branch.compound else rng.expovariate(1))
        tree = sample_pda_newick(n_leaves, rate, rng)
    return tree
