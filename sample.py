import fcntl
import pickle
import time
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import seed_everything
from tqdm import tqdm

from src.utils import resolve_checkpoint


SAMPLING_COLLATE_FN = {"_target_": "src.data.collate_fn_seqs_and_trees", "_partial_": True}
SAMPLING_MAP_FN = {"_target_": "src.data.dummy_map_fn", "_partial_": True}


def _majority_consensus_newick(newicks, min_freq=0.5):
    """Majority-rule consensus over Newick strings via dendropy."""
    import dendropy

    tns = dendropy.TaxonNamespace()
    dn_trees = dendropy.TreeList(taxon_namespace=tns)
    for nwk in newicks:
        if not nwk.endswith(";"):
            nwk = nwk + ";"
        dn_trees.append(
            dendropy.Tree.get(data=nwk, schema="newick", taxon_namespace=tns)
        )
    consensus = dn_trees.consensus(min_freq=min_freq)
    out = consensus.as_string(
        schema="newick",
        suppress_internal_node_labels=True,
        suppress_edge_lengths=True,
        suppress_rooting=True,
    ).strip()
    if not out.endswith(";"):
        out = out + ";"
    return out


def load_train_config(run_name: str) -> DictConfig:
    config_path = Path(get_original_cwd()) / "checkpoints" / run_name / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.yaml found in {config_path.parent}. "
            f"Was this run trained after the config-saving change to train.py?"
        )
    train_cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(train_cfg)
    return train_cfg


def prepare_cfg(cfg: DictConfig):
    """Apply run config and set sampling-specific data overrides. Modifies cfg in place."""
    run = cfg.get("run", None)
    if run is not None:
        train_cfg = load_train_config(run)
        cfg.data = train_cfg.data
        if "model" in cfg:
            cfg.model = train_cfg.model
        if "restart" in cfg:
            cfg.restart = run

    cfg.data.collate_fn = SAMPLING_COLLATE_FN
    cfg.data.map_fn = SAMPLING_MAP_FN
    cfg.data.batch_size = 1
    cfg.data.num_workers = 0

    # Sample the validation split, never the training stream. DataModule reserves
    # seeds [data.seed, data.seed + data.val_size) for validation and trains on
    # everything above; `seed` picks the band inside the reserved range, so the
    # trees are identical across runs whatever val_size a training config baked in.
    held_out_start = cfg.data.seed
    held_out_stop = cfg.data.seed + cfg.data.val_size
    n_trees = cfg.sampling.trees_per_n
    if cfg.seed < held_out_start or cfg.seed + n_trees > held_out_stop:
        raise ValueError(
            f"sampling seeds [{cfg.seed}, {cfg.seed + n_trees}) fall outside the "
            f"run's held-out range [{held_out_start}, {held_out_stop})"
        )
    cfg.data.seed = cfg.seed
    cfg.data.val_size = n_trees


def _existing_cell_counts(samples_path: Path) -> dict:
    """Unique-tree count per (n_leaves, msa_length) in an existing pkl.

    Dedup key matches the loader: (n, d, i, true_newick, sampled_newick).
    """
    if not samples_path.exists():
        return {}
    with samples_path.open("rb") as f:
        existing = pickle.load(f)
    seen = set()
    counts: dict = {}
    for entry in existing:
        key = (entry[0], entry[1], entry[2], entry[3], entry[4])
        if key in seen:
            continue
        seen.add(key)
        cell = (entry[0], entry[1])
        counts[cell] = counts.get(cell, 0) + 1
    return counts


def iter_batches(cfg: DictConfig):
    """Yield (n_leaves, msa_length, i, batch, datamodule) over the sampling grid."""
    skip_cell = cfg.sampling.get("skip_cell_if_existing", None)
    cell_counts: dict = {}
    if skip_cell is not None:
        samples_path = Path(get_original_cwd()) / "samples" / cfg.sampling.path
        cell_counts = _existing_cell_counts(samples_path)

    for n_leaves in cfg.sampling.n_leaves:
        for msa_length in cfg.sampling.msa_lengths:
            if skip_cell is not None:
                count = cell_counts.get((n_leaves, msa_length), 0)
                if count >= skip_cell:
                    print(
                        f"Skipping cell n={n_leaves} d={msa_length}: "
                        f"already has {count} unique trees (>= {skip_cell})"
                    )
                    continue

            print(f"Processing trees with {n_leaves} leaves and MSA length {msa_length}")
            cfg.data.dataset_cfg.n_min = n_leaves
            cfg.data.dataset_cfg.n_max = n_leaves
            # Force simulator to target exactly msa_length, but keep d_max (the
            # rejection ceiling) at its original value so indel expansion up to
            # the dataset-wide max is accepted instead of retried.
            with open_dict(cfg.data.dataset_cfg):
                cfg.data.dataset_cfg.d_target = msa_length

            datamodule = hydra.utils.instantiate(cfg.data)
            datamodule.setup("fit")

            for i, batch in enumerate(
                tqdm(
                    datamodule.val_dataloader(),
                    total=cfg.sampling.trees_per_n,
                    desc=f"n={n_leaves} d={msa_length}",
                )
            ):
                yield n_leaves, msa_length, i, batch, datamodule
                if i + 1 >= cfg.sampling.trees_per_n:
                    break


def save_results(results, cfg: DictConfig, *, verbose: bool = True):
    samples_path = Path(get_original_cwd()) / "samples" / cfg.sampling.path
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.sampling.get("append", True):
        # Lock-protected read-extend-write so concurrent shards can share one pkl.
        lock_path = samples_path.with_suffix(samples_path.suffix + ".lock")
        with lock_path.open("w") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            existing = []
            if samples_path.exists():
                with samples_path.open("rb") as f:
                    existing = pickle.load(f)
            existing.extend(results)
            with samples_path.open("wb") as f:
                pickle.dump(existing, f)
        if verbose:
            print(f"Appended {len(results)} trees to {samples_path} (total: {len(existing)})")
    else:
        with samples_path.open("wb") as f:
            pickle.dump(results, f)
        if verbose:
            print(f"Saved {len(results)} trees to {samples_path}")


def run_sampling(cfg: DictConfig):
    seed_everything(cfg.seed)
    prepare_cfg(cfg)
    print(OmegaConf.to_yaml(cfg))

    skip_if_existing = cfg.sampling.get("skip_if_existing", None)
    if skip_if_existing is not None:
        samples_path = Path(get_original_cwd()) / "samples" / cfg.sampling.path
        if samples_path.exists():
            with samples_path.open("rb") as f:
                existing = len(pickle.load(f))
            if existing >= skip_if_existing:
                print(
                    f"Skipping: {samples_path} already has {existing} samples "
                    f"(>= skip_if_existing={skip_if_existing})"
                )
                return

    model = hydra.utils.instantiate(cfg.model)
    ckpt_path = resolve_checkpoint(cfg.get("restart", None))
    if ckpt_path:
        print(f"Loading model from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("Using randomly-initialized model (no checkpoint loaded).")
    if torch.cuda.is_available():
        model.cuda()
    if cfg.sampling.get("use_ema", True):
        model.eval()
    else:
        # EMA tensors are bf16 (cloned from bf16-true params), so lerp_(v, 0.0002)
        # rounds to zero and EMA stays at init. Bypass the override to keep the
        # trained weights live.
        torch.nn.Module.eval(model)

    if cfg.sampling.get("fp32_inference", False):
        print("fp32_inference: casting model to fp32")
        model.float()

    if not hasattr(model, "sample_from_batch"):
        raise AttributeError(
            f"{type(model).__name__} does not implement sample_from_batch"
        )

    model_device = next(model.parameters()).device

    def move_batch(batch):
        seqs, trees = batch
        return seqs.to(model_device), trees

    sampling_kwargs = {"temperature": cfg.sampling.get("temperature", 1.0)}
    beam_size = cfg.sampling.get("beam_size", 1)
    if beam_size != 1:
        sampling_kwargs["beam_size"] = beam_size
    corrector_steps = cfg.sampling.get("corrector_steps", 0)
    if corrector_steps != 0:
        sampling_kwargs["corrector_steps"] = corrector_steps
    n_samples = cfg.sampling.get("n_samples", 1)
    if n_samples != 1:
        sampling_kwargs["n_samples"] = n_samples
    if cfg.sampling.get("layer_corrector", False):
        sampling_kwargs["layer_corrector"] = True
        sampling_kwargs["layer_drop_k"] = cfg.sampling.get("layer_drop_k", 5)
        sampling_kwargs["layer_lift_k"] = cfg.sampling.get("layer_lift_k", 4)
    consensus_n_samples = cfg.sampling.get("consensus_n_samples", 0)
    consensus_min_freq = cfg.sampling.get("consensus_min_freq", 0.5)

    def _sample_one(batch, batch_index, datamodule):
        if consensus_n_samples > 1:
            newicks = []
            true_newick = None
            for _ in range(consensus_n_samples):
                s = model.sample_from_batch(
                    batch, batch_index=batch_index, datamodule=datamodule,
                    **sampling_kwargs,
                )
                true_newick = s["true_tree_newick"]
                newicks.append(s["sampled_tree_newick"])
            return {
                "true_tree_newick": true_newick,
                "sampled_tree_newick": _majority_consensus_newick(
                    newicks, min_freq=consensus_min_freq,
                ),
            }
        return model.sample_from_batch(
            batch, batch_index=batch_index, datamodule=datamodule,
            **sampling_kwargs,
        )

    append_mode = cfg.sampling.get("append", True)

    results = []
    for n_leaves, msa_length, i, batch, datamodule in iter_batches(cfg):
        batch = move_batch(batch)
        t0 = time.time()
        sample = _sample_one(batch, i, datamodule)
        elapsed = time.time() - t0
        new_result = (
            n_leaves,
            msa_length,
            i,
            sample["true_tree_newick"],
            sample["sampled_tree_newick"],
            elapsed,
        )
        results.append(new_result)
        # Append mode: write only the new entry. Overwrite mode: rewrite cumulative.
        save_results(
            [new_result] if append_mode else results, cfg, verbose=False,
        )

    if not append_mode:
        save_results(results, cfg)
    else:
        print(f"Done: appended {len(results)} trees to samples/{cfg.sampling.path}")


@hydra.main(config_path="configs", config_name="sample", version_base="1.1")
def main(cfg: DictConfig):
    run_sampling(cfg)


if __name__ == "__main__":
    main()
