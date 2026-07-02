"""Runtime tree construction helpers for MiddleOut sampling.

Split out of the former monolithic ``unified.py``."""

from __future__ import annotations

from ete3 import TreeNode
from .data import ensure_numeric_node_names


class _SamplingTreeBuilder:
    def __init__(self):
        self.root = None
        self.name_dict = {}
        self.edge_index: list[list[int]] = []
        self.next_internal_id = 0

    def init_triplet(self, seed_triplet: list[int], max_leaf_id: int | None = None):
        max_id = (
            max(seed_triplet)
            if max_leaf_id is None
            else max(max_leaf_id, max(seed_triplet))
        )
        root_name = 2 * max_id + 1
        self.root = TreeNode(name=root_name)
        self.name_dict = {root_name: self.root}
        self.next_internal_id = root_name + 1
        for leaf_id in seed_triplet:
            leaf = TreeNode(name=leaf_id)
            self.root.add_child(leaf)
            self.name_dict[leaf_id] = leaf
        self._reindex()

    def _reindex(self):
        nodes = list(self.root.traverse())
        max_id = max(int(node.name) for node in nodes)
        edge_index = [[-1, -1, -1] for _ in range(max_id + 1)]
        for node in nodes:
            node_id = int(node.name)
            if node.is_root():
                children = [int(c.name) for c in node.children]
                edge_index[node_id] = children
            elif node.is_leaf():
                edge_index[node_id] = [int(node.up.name), -1, -1]
            else:
                children = [int(c.name) for c in node.children]
                edge_index[node_id] = [int(node.up.name), children[0], children[1]]
        self.edge_index = edge_index
        self.name_dict = {int(node.name): node for node in nodes}

    def add_leaf(self, edge_child: int, leaf_id: int):
        anchor_node = self.name_dict[edge_child]
        parent_node = anchor_node.up
        new_pendant_id = self.next_internal_id
        self.next_internal_id += 1
        new_pendant_node = TreeNode(name=new_pendant_id)
        new_leaf_node = TreeNode(name=leaf_id)
        parent_node.remove_child(anchor_node)
        new_pendant_node.add_child(anchor_node)
        new_pendant_node.add_child(new_leaf_node)
        parent_node.add_child(new_pendant_node)
        self._reindex()


def augmented_tree_to_rooted_tree(
    augmented_tree: TreeNode, root_token_id: int
) -> TreeNode:
    augmented_tree = ensure_numeric_node_names(augmented_tree.copy())
    adjacency: dict[int, list[int]] = {}
    for node in augmented_tree.traverse():
        node_id = int(node.name)
        adjacency.setdefault(node_id, [])
        for child in node.children:
            child_id = int(child.name)
            adjacency.setdefault(child_id, [])
            adjacency[node_id].append(child_id)
            adjacency[child_id].append(node_id)

    if root_token_id not in adjacency or len(adjacency[root_token_id]) != 1:
        raise RuntimeError("Augmented tree is missing a valid root token leaf")
    root_anchor = adjacency[root_token_id][0]
    adjacency[root_anchor].remove(root_token_id)
    del adjacency[root_token_id]

    def build_subtree(node_id: int, parent_id: int | None):
        children = [nbr for nbr in adjacency[node_id] if nbr != parent_id]
        node = TreeNode(name=node_id)
        for child_id in children:
            node.add_child(build_subtree(child_id, node_id))
        return node

    rooted_tree = TreeNode(name=root_anchor)
    for child_id in adjacency[root_anchor]:
        rooted_tree.add_child(build_subtree(child_id, root_anchor))
    return rooted_tree
