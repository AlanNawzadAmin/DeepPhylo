import math
from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


class PhylogeneticTreeDiffusionModel(pl.LightningModule):
    def __init__(self, lr, ema_decay=0.9998, weight_decay=1e-5, warmup_steps=1000,
                 mup_mha_fix=None, mup_optimizer=True):
        super().__init__()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        # None = auto: apply unless full_traj=True (canon_bl baselines were
        # trained without this fix; preserve their behavior for reproduction).
        self.mup_mha_fix = mup_mha_fix
        # When False, configure_optimizers falls back to a plain AdamW with one
        # uniform LR — bypasses the per-Linear lr_mult=1/fan_in rescaling and
        # the per-group eps schedule. Use to reproduce pre-µP runs.
        self.mup_optimizer = mup_optimizer
        self._ema_weights = {}  # keys are "_ema." + param_name
        self._saved_training_weights = None

    def _init_ema(self):
        if not self._ema_weights:
            self._ema_weights = {"_ema." + k: v.clone() for k, v in self.state_dict().items()}

    def _update_ema(self):
        self._init_ema()
        for k, v in super().state_dict().items():
            ema_key = "_ema." + k
            ema_v = self._ema_weights[ema_key]
            # Resync device/dtype if either drifted (e.g. checkpoint loaded with
            # bf16 EMA into a model that bf16-true then re-casts, or fp32 EMA
            # carried across a precision change). lerp_ requires both operands
            # to match exactly.
            if ema_v.device != v.device or ema_v.dtype != v.dtype:
                ema_v = ema_v.to(device=v.device, dtype=v.dtype)
                self._ema_weights[ema_key] = ema_v
            ema_v.lerp_(v, 1 - self.ema_decay)

    def eval(self):
        self._init_ema()
        if self._saved_training_weights is None:
            self._saved_training_weights = {k: v.clone() for k, v in self.state_dict().items()}
            super().load_state_dict({k[5:]: v for k, v in self._ema_weights.items()}, strict=False)
        return super().eval()

    def train(self, mode=True):
        if mode and self._saved_training_weights is not None:
            super().load_state_dict(self._saved_training_weights, strict=False)
            self._saved_training_weights = None
        return super().train(mode)

    def on_before_optimizer_step(self, optimizer):
        self._update_ema()

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if grads:
            total_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2)
            self.log("grad_norm", total_norm, prog_bar=False)
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        sd.update(self._ema_weights)
        return sd

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        normalize = lambda k: k.replace("._orig_mod", "")
        normalized = {normalize(k): v for k, v in state_dict.items()}
        live_keys = {normalize(k): k for k in super().state_dict().keys()}

        # Split: _ema.* keys go to EMA, rest go to model. Remap both back to the
        # live model's key form so future _update_ema lookups match.
        ema = {"_ema." + live_keys[k[len("_ema."):]]: v
               for k, v in normalized.items()
               if k.startswith("_ema.") and k[len("_ema."):] in live_keys}
        model_sd = {live_keys[k]: v for k, v in normalized.items()
                    if not k.startswith("_ema.") and k in live_keys}

        result = super().load_state_dict(model_sd, strict=strict, **kwargs)

        # Cast EMA tensors to match live param dtype/device. Without this, an old
        # bf16-true checkpoint loaded into an fp32 (bf16-mixed) model would leave
        # EMA in bf16 while live params are fp32 — future EMA updates would then
        # silently underflow on the bf16 EMA tensors.
        live = super().state_dict()
        if ema:
            self._ema_weights = {
                k: v.to(dtype=live[k[5:]].dtype, device=live[k[5:]].device)
                for k, v in ema.items()
            }
        else:
            self._ema_weights = {"_ema." + k: v.clone() for k, v in live.items()}

        return result

    def _mup_depth(self) -> int:
        depths = [
            len(m.attn_layers)
            for m in self.modules()
            if hasattr(m, "attn_layers") and isinstance(m.attn_layers, nn.ModuleList)
        ]
        return max(depths) if depths else 8

    def configure_optimizers(self):
        if not self.mup_optimizer:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = LambdaLR(optimizer, lr_lambda=self._warmup_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        # µP per-layer LR (1/fan_in on Linears, 1 elsewhere) and per-layer Adam eps
        # following Table 1's eps ∝ 1/(L · d_out) scaling rule. Leading constant
        # `eps_base` is set well below observed gradient magnitudes so Adam stays
        # in its normalized regime (step ≈ LR per element) rather than the linear
        # regime (step ∝ grad/eps).
        L = self._mup_depth()
        # eps deliberately *very* small so Adam stays firmly in its normalized
        # regime (step ≈ LR per element) and never enters the linear regime
        # (step ∝ grad/eps). Per-group eps = eps_base / (L · d_out) then lands
        # around 1e-15–1e-16, far below any observed gradient magnitude.
        eps_base = 1e-12

        # Depth-µP (only when SPDBiasedAttention was built with depth_mup=True):
        # params whose residual branch is scaled by 1/√L in forward (the per-layer
        # lists marked _mup_depth_scaled) get an extra 1/√L on their LR. When the
        # toggle is off nothing is marked, depth_param_ids is empty, and LRs are
        # untouched. Entrance/shared params always keep O(1) LR.
        depth_param_ids = set()
        for mod in self.modules():
            if getattr(mod, "_mup_depth_scaled", False):
                depth_param_ids.update(id(p) for p in mod.parameters())
        depth_factor = 1.0 / math.sqrt(L)

        use_mha_fix = self.mup_mha_fix
        if use_mha_fix is None:
            use_mha_fix = not getattr(self, "full_traj", False)

        meta = {}
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                meta[id(mod.weight)] = (1.0 / mod.in_features, mod.out_features)
                if mod.bias is not None:
                    meta[id(mod.bias)] = (1.0, mod.out_features)
            elif isinstance(mod, nn.Embedding):
                meta[id(mod.weight)] = (1.0, mod.embedding_dim)
            elif isinstance(mod, nn.LayerNorm):
                d = mod.normalized_shape[0] if isinstance(mod.normalized_shape, (tuple, list)) else mod.normalized_shape
                if mod.weight is not None:
                    meta[id(mod.weight)] = (1.0, d)
                if mod.bias is not None:
                    meta[id(mod.bias)] = (1.0, d)
            elif isinstance(mod, nn.MultiheadAttention) and use_mha_fix:
                # Fused QKV is a raw Parameter, not a Linear submodule, so it
                # would otherwise fall through to lr_mult=1. out_proj is a
                # Linear subclass and is picked up by the nn.Linear branch.
                if mod.in_proj_weight is not None:
                    meta[id(mod.in_proj_weight)] = (1.0 / mod.embed_dim, 3 * mod.embed_dim)
                if mod.in_proj_bias is not None:
                    meta[id(mod.in_proj_bias)] = (1.0, 3 * mod.embed_dim)

        groups = defaultdict(list)
        unhandled = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in meta:
                lr_mult, d_out = meta[id(p)]
            else:
                unhandled.append((name, tuple(p.shape)))
                lr_mult, d_out = 1.0, (p.shape[-1] if p.ndim >= 1 else 1)
            if id(p) in depth_param_ids:
                lr_mult = lr_mult * depth_factor
            groups[(lr_mult, d_out)].append(p)

        if unhandled:
            # Multi-dim fall-throughs (marked '!') almost always mean a missing
            # module-type handler above (e.g. Conv2d weights, MHA in_proj_weight,
            # Bilinear weight) and silently get the wrong LR. 1D fall-throughs
            # are usually fine — bare nn.Parameter tokens treated as embeddings.
            print(f"[µP] {len(unhandled)} params fell through to default lr_mult=1:")
            for name, shape in unhandled:
                flag = "!" if len(shape) >= 2 else " "
                print(f"  {flag} {name}: {shape}")

        param_groups = []
        for (lr_mult, d_out), ps in groups.items():
            eps = eps_base / (L * d_out)
            # AdamW applies decoupled WD as lr·wd·θ, so the per-group lr_mult would
            # otherwise scale the effective decay too. Divide wd by lr_mult so the
            # realized decay is a uniform lr·weight_decay·θ across all groups,
            # independent of width (fan_in) and depth.
            param_groups.append({
                "params": ps,
                "lr": self.lr * lr_mult,
                "eps": eps,
                "weight_decay": self.weight_decay / lr_mult,
            })
        for (lr_mult, d_out), ps in sorted(groups.items()):
            n_params = sum(p.numel() for p in ps)
            eps = eps_base / (L * d_out)
            wd = self.weight_decay / lr_mult
            print(f"[µP] lr_mult={lr_mult:>10.5g}  d_out={d_out:>5d}  →  "
                  f"{len(ps):4d} tensors, {n_params:>12,d} params  "
                  f"(LR {self.lr * lr_mult:.3e}, eps {eps:.3e}, wd {wd:.3e})")

        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = LambdaLR(optimizer, lr_lambda=self._warmup_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # adjust each training step
            },
        }

    def _warmup_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        return 1.0
