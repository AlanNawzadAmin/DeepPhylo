from pathlib import Path

from hydra.utils import get_original_cwd


def resolve_checkpoint(restart: str | None):
    if restart is None:
        return None

    base = Path(get_original_cwd()) / "checkpoints"
    run_dir = base / restart

    if run_dir.is_dir():
        ckpts = sorted(run_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {run_dir}")
        return str(ckpts[-1])

    ckpt_path = base / restart
    if ckpt_path.is_file():
        return str(ckpt_path)

    raise FileNotFoundError(f"Invalid restart path: {restart}")
