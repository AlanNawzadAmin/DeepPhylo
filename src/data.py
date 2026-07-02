import inspect

import hydra
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

from .simulation.dataset import ete3_and_seqs_to_dgl


def collate_fn_seqs_and_trees(batch):
    seqs = torch.stack([b[1] for b in batch])
    trees = [b[0] for b in batch]  # keep list of trees
    return seqs, trees


def dummy_map_fn(sample):
    return sample


def dgl_map_fn(sample):
    tree, seqs, n_is_weight = sample
    return ete3_and_seqs_to_dgl(tree, seqs), n_is_weight


class MapDataset(IterableDataset):
    def __init__(self, dataset, fn):
        self.dataset = dataset
        self.fn = fn
        try:
            self.fn_accepts_idx = "idx" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            self.fn_accepts_idx = False

    def __iter__(self):
        for idx, item in enumerate(self.dataset):
            if self.fn_accepts_idx:
                yield self.fn(item, idx=idx)
            else:
                yield self.fn(item)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_cfg,
        seed,
        val_size=500,
        batch_size=1,
        num_workers=0,
        collate_fn=None,
        map_fn=None,
    ):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.val_size = val_size
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.collate_fn = hydra.utils.instantiate(collate_fn)
        self.map_fn = hydra.utils.instantiate(map_fn)

    def setup(self, stage=None):
        self.train_dataset = MapDataset(
            dataset=hydra.utils.instantiate(
                self.dataset_cfg,
                seed=self.seed + self.val_size,
            ),
            fn=self.map_fn,
        )

        self.val_dataset = MapDataset(
            dataset=hydra.utils.instantiate(
                self.dataset_cfg,
                size=self.val_size,
                seed=self.seed,
            ),
            fn=self.map_fn,
        )

    def train_dataloader(self):
        loader_kwargs = {}
        if torch.cuda.is_available():
            loader_kwargs["pin_memory"] = True
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            **loader_kwargs,
        )

    def val_dataloader(self):
        loader_kwargs = {}
        if torch.cuda.is_available():
            loader_kwargs["pin_memory"] = True
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            **loader_kwargs,
        )
