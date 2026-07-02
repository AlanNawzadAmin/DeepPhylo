import datetime
import os
import warnings

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import hydra
import pytorch_lightning as pl
import torch
import wandb

from dgl.base import DGLWarning
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict

_orig_torch_load = torch.load
torch.load = lambda *args, **kwargs: _orig_torch_load(*args, **{**kwargs, "weights_only": False})
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import src as src
from src.utils import resolve_checkpoint


@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def main(cfg: DictConfig):
    warnings.filterwarnings("ignore", category=DGLWarning)
    torch.set_float32_matmul_precision("medium")
    seed_everything(cfg.seed)
    print(cfg)

    callbacks = []
    wandb_logger = None

    # DDP-safe run name. Every rank must compute the same string so they all
    # write to the same ckpt_dir; otherwise only rank 0 has the ModelCheckpoint
    # callback and `on_validation_end` collectives desync.
    #   - SLURM job: SLURM_JOB_ID is set on every rank → use it.
    #   - Local DDP / single-rank: generate once and stash in env. Lightning's
    #     subprocess launcher inherits parent env, so children see the same
    #     RUN_NAME and skip regeneration.
    #   wandb.run.name is rank-0-only (`@rank_zero_experiment`), so we don't
    #   use it for naming.
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        run_name = f"slurm-{slurm_job_id}"
    elif os.environ.get("RUN_NAME"):
        run_name = os.environ["RUN_NAME"]
    else:
        run_name = f"local-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        os.environ["RUN_NAME"] = run_name

    if cfg.use_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        # Optional human-friendly W&B identity (e.g. ablation sweeps): WANDB_RUN_NAME
        # overrides the display name and WANDB_RUN_GROUP assigns a shared group, both
        # without touching the DDP-safe run_name used for ckpt dirs. Unset => prior
        # behavior (name=run_name, no group).
        wandb_display_name = os.environ.get("WANDB_RUN_NAME") or run_name
        wandb_group = os.environ.get("WANDB_RUN_GROUP")
        wandb_logger = WandbLogger(
            project="PhyloPFN",
            entity="eb4727-nyu",
            name=wandb_display_name,
            group=wandb_group,
        )
        _ = wandb_logger.experiment  # force initialization on rank 0
        if wandb.run is not None:
            wandb.run.config.update(
                OmegaConf.to_container(cfg, resolve=True), allow_val_change=True
            )

    # Checkpoint always — independent of wandb. ckpt_dir matches across ranks
    # because run_name is computed deterministically from the env.
    ckpt_dir = os.path.join(get_original_cwd(), "checkpoints", run_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="step_{step}",
        every_n_train_steps=2000,
        save_top_k=-1,
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(ckpt_dir, "config.yaml"))
    callbacks.append(checkpoint_callback)

    ckpt_path = resolve_checkpoint(cfg.restart)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")

    model = hydra.utils.instantiate(cfg.model)

    n_gpus = max(1, torch.cuda.device_count())
    per_rank_batch = cfg.data.batch_size
    with open_dict(cfg.train):
        # Derive accumulate from the configured effective batch (= unique trees
        # per opt step). Keeps per-opt-step gradient identical regardless of
        # GPU count or per-rank batch.
        cfg.train.accumulate_grad_batches = max(
            1, cfg.effective_batch_size // (n_gpus * per_rank_batch)
        )
        # val_check_interval is in micro-batches, so scale with accumulate to
        # keep validations at the same optimizer-step cadence as the 2-GPU
        # baseline (accumulate=16) the value was originally tuned for.
        cfg.train.val_check_interval = int(
            cfg.train.val_check_interval * cfg.train.accumulate_grad_batches / 16
        )

    trainer = pl.Trainer(
        accelerator="auto",
        strategy="ddp",
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=5,
        enable_progress_bar=True,
        precision=cfg.precision,
        detect_anomaly=cfg.detect_anomaly,
        **cfg.train,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
