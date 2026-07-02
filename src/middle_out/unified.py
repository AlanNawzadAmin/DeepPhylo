"""Backward-compatible re-export shim.


``unified.py`` was split into data/architecture/sampling/module. This
module re-exports every public + private name so existing imports and the
Hydra ``src.middle_out.unified.*`` _target_s keep working."""

from .data import (  # noqa: F401
    MiddleOutStepExample,
    MiddleOutTrainStep,
    MiddleOutTrainExample,
    ensure_numeric_node_names,
    normalize_rooted_tree_ids,
    augment_rooted_tree_with_token,
    canonical_unroot,
    remap_tree_ids,
    tree_to_edge_index,
    prune_tree,
    compute_node_leaf_masks,
    build_leafmask_lookup,
    resolve_builder_edge_child,
    precompute_leaf_sibling_masks,
    compute_dense_targets,
    _squaretrick_core,
    squaretrick,
    edge_index_to_spd,
    build_global_node_basis,
    _build_train_step,
    preprocess_train_example,
    train_map_fn,
    train_collate_fn,
)
from .architecture import (  # noqa: F401
    build_node_embeddings_from_basis,
    _extend_attn_inputs,
    SPDAttentionTreeModelWithFeatures,
    SPDAttentionCovTreeModel,
    MiddleOutMergePredictor,
)
from .sampling import (  # noqa: F401
    _SamplingTreeBuilder,
    augmented_tree_to_rooted_tree,
)
from .module import (  # noqa: F401
    UnifiedMiddleOutModule,
)

__all__ = [
    "MiddleOutStepExample",
    "MiddleOutTrainStep",
    "MiddleOutTrainExample",
    "ensure_numeric_node_names",
    "normalize_rooted_tree_ids",
    "augment_rooted_tree_with_token",
    "canonical_unroot",
    "remap_tree_ids",
    "tree_to_edge_index",
    "prune_tree",
    "compute_node_leaf_masks",
    "build_leafmask_lookup",
    "resolve_builder_edge_child",
    "precompute_leaf_sibling_masks",
    "compute_dense_targets",
    "_squaretrick_core",
    "squaretrick",
    "edge_index_to_spd",
    "build_global_node_basis",
    "_build_train_step",
    "preprocess_train_example",
    "train_map_fn",
    "train_collate_fn",
    "build_node_embeddings_from_basis",
    "_extend_attn_inputs",
    "SPDAttentionTreeModelWithFeatures",
    "SPDAttentionCovTreeModel",
    "MiddleOutMergePredictor",
    "_SamplingTreeBuilder",
    "augmented_tree_to_rooted_tree",
    "UnifiedMiddleOutModule",
]
