"""QuartFormer: quartet-puzzling phylogenetic inference.

A reimplementation of QuartFormer (Su & Jiao, bioRxiv 2025.04.05.646867) fitted
to the PhyloPFN simulated datasets (JC / Cat / Indels). The method predicts, for
every 4-taxon subset, a distribution over the three unrooted quartet topologies,
lets quartets that share >=3 taxa exchange information via masked ("sparse")
attention, and assembles a full tree from the weighted quartets by quartet
puzzling.

Public entry point is ``QuartFormerModule`` (a LightningModule) instantiated via
``configs/model/quartformer.yaml``.
"""

from .module import QuartFormerModule

__all__ = ["QuartFormerModule"]
