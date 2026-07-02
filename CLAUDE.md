# Repo notes for agents

## Heavy directories — do not search or list

These directories contain experiment artifacts (~1.1T total). They are in `.gitignore`, so `Grep`/`Glob` skip them automatically, but `find`, `ls -R`, `du`, etc. will not.

- `checkpoints/` — model checkpoints (~1.1T)
- `outputs/` — Hydra run outputs (~15G)
- `slurm/` — SLURM job logs (~8G)
- `samples/` — pickled tree samples

**Do not** recursively list or traverse these via `Bash`. If you need a specific file, ask for the path.

## Layout

- `train.py` / `sample.py` — entry points (Hydra configs under `configs/`).
- `src/middle_out/` — "Middle out" model. `src/bottom_up/` — "Bottom up" model (and the "Phyloformer 2" baseline head). `src/top_down/` — "Top down" model.
- `src/simulation/` — MSA/tree simulators. `src/evopf.py`, `src/spd_attn.py` — shared encoders.
- `src/bottom_up/BayesNJ/` — vendored external Phyloformer SDK; leave as-is.

Production codenames "artree", "phyloformer", "layerwise" must not appear in code — use Middle out / Bottom up / Top down. "Phyloformer 2" is allowed only as the baseline config/label.
