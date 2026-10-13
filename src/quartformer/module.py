"""QuartFormer LightningModule.

Training objective (as in the QuartFormer paper): 3-way cross-entropy over the
correct unrooted quartet topology. Each training step samples several `block_size`
-taxon blocks from one simulated tree, enumerates all C(block_size,4) quartets per
block, and classifies each against the label induced by the true tree. Blocks are
batched through the network in one forward for GPU efficiency.

The per-tree loss is scaled by the difficulty importance weight (as in the other
models) so the effective distribution over tree sizes matches — the dataset
oversamples large N (weight_n_by_difficulty) and the weight corrects back.

Decoding is by quartet puzzling (see puzzling.py) into a full tree. A cheap RF
proxy is also logged during validation on a few trees.
"""

import numpy as np
import torch
import torch.nn.functional as F
from ete3 import Tree

from ..model import PhylogeneticTreeDiffusionModel
from ..sampling_utils import ensure_newick_semicolon
from .model import QuartFormerNet
from .puzzling import QuartetOracle, quartet_puzzling
from .quartets import (
    all_pairs_topo_distance,
    coverage_blocks,
    enumerate_local_quartets,
    label_quartets,
    multihot_species,
    sample_blocks,
    shared_taxa_mask,
)


class QuartFormerModule(PhylogeneticTreeDiffusionModel):
    def __init__(
        self,
        *,
        lr,
        mode,
        block_size=16,
        n_blocks_train=8,
        n_blocks_val=8,
        importance_weight=True,
        token_dim=256,
        n_heads=8,
        n_layers=3,
        ffn_dim=512,
        embed_dim=16,
        feat_hidden=256,
        feat_dim=128,
        site_chunk=512,
        # Sites are i.i.d., so a random subset is an unbiased estimate of the
        # pattern-frequency feature and bounds backprop memory. Eval runs under
        # no_grad with site chunking and uses the full alignment.
        max_sites_train=512,
        eval_max_sites=None,
        weight_decay=1e-5,
        ema_decay=0.9998,
        warmup_steps=1000,
        mup_mha_fix=None,
        mup_optimizer=True,
        # The µP optimizer sets eps ~1e-16, small enough that an underflowing
        # second moment makes Adam's normalized update explode. Floor it.
        min_eps=1e-8,
        # eval / puzzling knobs
        eval_min_per_taxon=40,
        eval_n_orderings=50,
        eval_max_quartets_per_insertion=400,
        eval_block_batch=16,
        # cheap in-training RF monitor
        monitor_rf=True,
        monitor_rf_n_trees=3,
        monitor_rf_min_per_taxon=12,
        monitor_rf_n_orderings=10,
    ):
        super().__init__(lr, ema_decay=ema_decay, weight_decay=weight_decay,
                         warmup_steps=warmup_steps, mup_mha_fix=mup_mha_fix,
                         mup_optimizer=mup_optimizer)
        self.save_hyperparameters()
        self.min_eps = min_eps
        self.block_size = block_size
        self.n_blocks_train = n_blocks_train
        self.n_blocks_val = n_blocks_val
        self.importance_weight = importance_weight
        self.max_sites_train = max_sites_train
        self.eval_max_sites = eval_max_sites
        self.eval_min_per_taxon = eval_min_per_taxon
        self.eval_n_orderings = eval_n_orderings
        self.eval_max_quartets_per_insertion = eval_max_quartets_per_insertion
        self.eval_block_batch = eval_block_batch
        self.monitor_rf = monitor_rf
        self.monitor_rf_n_trees = monitor_rf_n_trees
        self.monitor_rf_min_per_taxon = monitor_rf_min_per_taxon
        self.monitor_rf_n_orderings = monitor_rf_n_orderings
        self._val_rfs = []

        vocab_size = 7 if mode == "dna" else 23
        self.vocab_size = vocab_size
        self.net = QuartFormerNet(
            vocab_size=vocab_size, block_size=block_size, token_dim=token_dim,
            n_heads=n_heads, n_layers=n_layers, ffn_dim=ffn_dim,
            embed_dim=embed_dim, feat_hidden=feat_hidden, feat_dim=feat_dim,
            site_chunk=site_chunk,
        )

        # Fixed block-local structures (identical for every full-size block).
        lq = enumerate_local_quartets(block_size)
        self.register_buffer("local_quartets", lq, persistent=False)
        self.register_buffer(
            "species_multihot", multihot_species(lq, block_size), persistent=False)
        mask = shared_taxa_mask(lq)  # (Q,Q) bool, True=attend
        bias = torch.zeros_like(mask, dtype=torch.float32)
        bias.masked_fill_(~mask, float("-inf"))
        self.register_buffer("attn_bias", bias, persistent=False)
        self._block_cache = {}

    def configure_optimizers(self):
        out = super().configure_optimizers()
        # Floor the (tiny µP) Adam eps to keep Adam in its stable regime. Only
        # matters on the µP path; plain AdamW already uses eps=1e-8.
        opt = out["optimizer"] if isinstance(out, dict) else out
        n_floored = 0
        for g in opt.param_groups:
            if g.get("eps", self.min_eps) < self.min_eps:
                g["eps"] = self.min_eps
                n_floored += 1
        if n_floored:
            print(f"[quartformer] floored Adam eps to {self.min_eps:g} for "
                  f"{n_floored} param groups (µP eps was ~1e-16).")
        return out

    # ---- shared helpers -------------------------------------------------
    def _block_tensors(self, k, device, dtype):
        """(local_quartets, species_multihot[width=block_size], attn_bias) for a
        block of k taxa (k <= block_size). Cached per k."""
        if k == self.block_size:
            return (self.local_quartets, self.species_multihot.to(dtype),
                    self.attn_bias.to(dtype))
        if k not in self._block_cache:
            lq = enumerate_local_quartets(k)
            Q = lq.shape[0]
            sp = torch.zeros(Q, self.block_size)
            sp.scatter_(1, lq, 1.0)
            mask = shared_taxa_mask(lq)
            bias = torch.zeros_like(mask, dtype=torch.float32)
            bias.masked_fill_(~mask, float("-inf"))
            self._block_cache[k] = (lq, sp, bias)
        lq, sp, bias = self._block_cache[k]
        return (lq.to(device), sp.to(device=device, dtype=dtype),
                bias.to(device=device, dtype=dtype))

    def _subsample_sites(self, seqs, max_sites, generator=None):
        """Return seqs with at most `max_sites` columns (random subset). Sites
        are i.i.d. so this is an unbiased subsample of the alignment."""
        L = seqs.shape[1]
        if not max_sites or L <= max_sites:
            return seqs
        idx = torch.randperm(L, generator=generator, device=seqs.device)[:max_sites]
        return seqs[:, idx]

    def _unpack(self, batch):
        """Return (tree, seqs[N,L], weight) for the single-item batch. Handles
        both the 3-tuple training collate and the 2-tuple sampling collate."""
        if len(batch) == 3:
            seqs, trees, weights = batch
            weight = float(weights[0])
        else:
            seqs, trees = batch
            weight = 1.0
        assert seqs.shape[0] == 1, "QuartFormer expects batch_size=1"
        return trees[0], seqs[0], weight

    def _forward_blocks(self, seqs_long, blocks, k):
        """Run the net on a (B, k) block index tensor. Returns logits (B, Q, 3)
        and the (B, Q, 4) global-taxon quartets (for labeling)."""
        device = seqs_long.device
        dtype = self.net.classifier.weight.dtype
        lq, sp, bias = self._block_tensors(k, device, dtype)
        blocks = blocks.to(device)
        block_seqs = seqs_long[blocks]                 # (B, k, L)
        logits = self.net(block_seqs, lq, sp, bias)    # (B, Q, 3)
        global_quartets = blocks[:, lq]                # (B, Q, 4)
        return logits, global_quartets

    def _block_loss(self, seqs, dist, blocks, k, weight):
        seqs_long = seqs.long()
        logits, global_quartets = self._forward_blocks(seqs_long, blocks, k)
        B, Q, _ = logits.shape
        labels = label_quartets(
            global_quartets.reshape(-1, 4).cpu().numpy(), dist
        ).to(logits.device)
        flat = logits.reshape(-1, 3).float()
        loss = F.cross_entropy(flat, labels)
        if self.importance_weight:
            loss = loss * weight
        acc = (flat.argmax(-1) == labels).float().mean()
        return loss, acc

    # ---- lightning API --------------------------------------------------
    def training_step(self, batch, batch_idx):
        tree, seqs, weight = self._unpack(batch)
        n = seqs.shape[0]
        assert n >= self.block_size, (
            f"n_leaves={n} < block_size={self.block_size}")
        dist = all_pairs_topo_distance(tree, n)
        blocks = sample_blocks(n, self.block_size, self.n_blocks_train)
        seqs = self._subsample_sites(seqs, self.max_sites_train)
        loss, acc = self._block_loss(seqs, dist, blocks, self.block_size, weight)
        self.log("train_loss", loss, prog_bar=True, batch_size=1)
        self.log("train_acc", acc, prog_bar=True, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        tree, seqs, weight = self._unpack(batch)
        n = seqs.shape[0]
        k = min(self.block_size, n)
        dist = all_pairs_topo_distance(tree, n)
        gen = torch.Generator().manual_seed(batch_idx)
        blocks = sample_blocks(n, k, self.n_blocks_val, generator=gen)
        seqs_sub = self._subsample_sites(seqs, self.max_sites_train)
        loss, acc = self._block_loss(seqs_sub, dist, blocks, k, weight)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=1)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True, batch_size=1)

        # Cheap RF proxy on the first few val trees (real end-to-end signal at
        # this tree's own N; wrapped so it can never break validation).
        if self.monitor_rf and batch_idx < self.monitor_rf_n_trees:
            try:
                nrf = self._quick_rf(tree, seqs, seed=batch_idx)
                self._val_rfs.append(nrf)
            except Exception as e:  # noqa: BLE001
                if batch_idx == 0:
                    print(f"[quartformer] RF monitor skipped: {e}", flush=True)
        return loss

    def on_validation_epoch_end(self):
        if self._val_rfs:
            mean_rf = float(np.mean(self._val_rfs))
            self.log("val_rf", mean_rf, prog_bar=True, sync_dist=True, batch_size=1)
            self._val_rfs = []

    # ---- inference ------------------------------------------------------
    @torch.no_grad()
    def predict_quartets(self, seqs, *, seed=0, min_per_taxon=None):
        """Run the model over coverage blocks and return a populated
        QuartetOracle. seqs: (N, L) on the model device."""
        min_per_taxon = self.eval_min_per_taxon if min_per_taxon is None else min_per_taxon
        gen_sites = torch.Generator(device=seqs.device).manual_seed(seed)
        seqs = self._subsample_sites(seqs, self.eval_max_sites, generator=gen_sites)
        n = seqs.shape[0]
        k = min(self.block_size, n)
        lq, _, _ = self._block_tensors(k, seqs.device, self.net.classifier.weight.dtype)
        lq_np = lq.cpu().numpy()
        seqs_long = seqs.long()
        gen = torch.Generator().manual_seed(seed)
        blocks = coverage_blocks(n, k, min_per_taxon, generator=gen)
        oracle = QuartetOracle()
        bb = self.eval_block_batch
        for start in range(0, blocks.shape[0], bb):
            chunk = blocks[start:start + bb]
            logits, _ = self._forward_blocks(seqs_long, chunk, k)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()  # (b,Q,3)
            for bi in range(chunk.shape[0]):
                oracle.add_block(chunk[bi].cpu().numpy(), lq_np, probs[bi])
        return oracle

    def _quick_rf(self, tree, seqs, *, seed=0):
        oracle = self.predict_quartets(
            seqs, seed=seed, min_per_taxon=self.monitor_rf_min_per_taxon)
        n = seqs.shape[0]
        pred = quartet_puzzling(
            list(range(n)), oracle,
            n_orderings=self.monitor_rf_n_orderings, seed=seed,
            max_quartets_per_insertion=self.eval_max_quartets_per_insertion,
        )
        t_true = Tree(ensure_newick_semicolon(tree.write(format=9)), format=1)
        t_pred = Tree(ensure_newick_semicolon(pred), format=1)
        rf, max_rf = t_true.robinson_foulds(t_pred, unrooted_trees=True)[:2]
        return rf / max_rf if max_rf else 0.0

    @torch.no_grad()
    def sample_from_batch(self, batch, *, batch_index=None, datamodule=None,
                          temperature=1.0, seed=None, **kwargs):
        """temperature > 0 draws a posterior tree: quartet resolutions are sampled
        from the model's distribution and one insertion order is used. temperature
        0 gives the point estimate: argmax quartets, consensus over
        `eval_n_orderings` puzzling passes."""
        tree, seqs, _ = self._unpack(batch)
        n = seqs.shape[0]
        if seed is None:
            # Each draw needs its own insertion orders; the point estimate is fixed.
            seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item()) if temperature > 0 else 0
        oracle = self.predict_quartets(seqs, seed=seed)
        sampled_newick = quartet_puzzling(
            list(range(n)), oracle,
            n_orderings=1 if temperature > 0 else self.eval_n_orderings,
            seed=seed,
            temperature=temperature,
            max_quartets_per_insertion=self.eval_max_quartets_per_insertion,
        )
        return {
            "true_tree_newick": ensure_newick_semicolon(tree.write(format=9)),
            "sampled_tree_newick": ensure_newick_semicolon(sampled_newick),
        }
