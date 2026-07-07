from __future__ import annotations

"""
Test-time guidance functions — the piece of CTG that this repo is
currently missing.

CTG's central idea is NOT the diffusion model itself (that part of this
repo, models/diffusion_model.py, is already a perfectly reasonable
conditional DDPM). CTG's contribution is *how sampling is done*: instead
of training a separate model per rule ("stay under speed limit", "don't
collide", "stay behind the lead car"), you write each rule as a small
differentiable cost function of the trajectory, and at every reverse-
diffusion step you nudge the current noisy sample x_t downhill along that
cost's gradient (a classifier-guidance-style correction), before doing the
usual DDPM denoising update.

This module provides:
    - a registry of guidance ("rule") functions
    - `compose_guidance(...)` to combine several of them with weights
    - `guidance_grad(...)` to compute d(cost)/d(x) for use inside sampling

`models/diffusion_model.py:DiffusionModel.guided_sample` (added alongside
the existing `sample`) is the thing that actually calls these during the
reverse process. This file has no dependency on the model, so new rules
can be added/tested independently of it.
"""

from typing import Callable, Dict, List, Optional

import torch

GuidanceFn = Callable[..., torch.Tensor]
GUIDANCE_REGISTRY: Dict[str, GuidanceFn] = {}


def register_guidance(name: str):
    """Decorator to add a cost function to the registry under `name`."""

    def deco(fn: GuidanceFn) -> GuidanceFn:
        if name in GUIDANCE_REGISTRY:
            raise ValueError(f"Guidance function '{name}' already registered")
        GUIDANCE_REGISTRY[name] = fn
        return fn

    return deco


# ---------------------------------------------------------------------------
# Built-in rules. All assume the state layout this repo already uses:
#   channel 0, 1 = x, y position
#   channel 2, 3 = vx, vy velocity   (present in nuscenes_dataset.py's D=4)
# Add your own with @register_guidance("name") elsewhere and they'll show
# up automatically in scripts/rollout.py's --guidance choices.
# ---------------------------------------------------------------------------


@register_guidance("collision_avoidance")
def collision_avoidance_cost(
    x: torch.Tensor,
    future_mask: Optional[torch.Tensor] = None,
    min_dist: float = 2.0,
    **_,
) -> torch.Tensor:
    """Penalize any pair of agents coming closer than `min_dist` meters
    at any future timestep. x: (B, F, A, D)."""
    pos = x[..., :2]
    diff = pos.unsqueeze(3) - pos.unsqueeze(2)          # (B, F, A, A, 2)
    dist = torch.linalg.norm(diff, dim=-1) + 1e-6        # (B, F, A, A)

    A = pos.shape[2]
    eye = torch.eye(A, device=x.device, dtype=torch.bool)
    dist = dist.masked_fill(eye, float("inf"))

    violation = torch.clamp(min_dist - dist, min=0.0)

    if future_mask is not None:
        pair_mask = (future_mask.unsqueeze(3) & future_mask.unsqueeze(2)).float()
        violation = violation * pair_mask

    return (violation ** 2).sum(dim=(1, 2, 3)).mean()


@register_guidance("target_speed")
def target_speed_cost(
    x: torch.Tensor,
    future_mask: Optional[torch.Tensor] = None,
    target_speed: float = 8.0,
    ego_index: Optional[int] = None,
    **_,
) -> torch.Tensor:
    """Penalize deviation from `target_speed` (m/s), computed from the
    (vx, vy) channels. If `ego_index` is given, only that agent is
    constrained; otherwise all agents are."""
    vel = x[..., 2:4]
    speed = torch.linalg.norm(vel, dim=-1)               # (B, F, A)

    mask = future_mask
    if ego_index is not None:
        speed = speed[:, :, ego_index:ego_index + 1]
        if mask is not None:
            mask = mask[:, :, ego_index:ego_index + 1]

    err = (speed - target_speed) ** 2
    if mask is not None:
        err = err * mask.float()
        return err.sum() / mask.float().sum().clamp_min(1.0)
    return err.mean()


@register_guidance("stay_behind_lead")
def stay_behind_lead_cost(
    x: torch.Tensor,
    future_mask: Optional[torch.Tensor] = None,
    ego_index: int = 0,
    lead_index: int = 1,
    min_gap: float = 5.0,
    **_,
) -> torch.Tensor:
    """Penalize the ego agent closing to less than `min_gap` (in the
    x-channel, i.e. treat x as the longitudinal axis) behind a lead
    agent."""
    ego_x = x[:, :, ego_index, 0]
    lead_x = x[:, :, lead_index, 0]
    gap = lead_x - ego_x
    violation = torch.clamp(min_gap - gap, min=0.0)

    if future_mask is not None:
        valid = (future_mask[:, :, ego_index] & future_mask[:, :, lead_index]).float()
        violation = violation * valid

    return (violation ** 2).mean()


@register_guidance("smoothness")
def smoothness_cost(x: torch.Tensor, future_mask: Optional[torch.Tensor] = None, **_) -> torch.Tensor:
    """Penalize large frame-to-frame acceleration (2nd derivative of
    position), useful as a mild regularizer alongside sharper rules."""
    pos = x[..., :2]
    vel = pos[:, 1:] - pos[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    cost = (acc ** 2).sum(dim=-1)                        # (B, F-2, A)
    if future_mask is not None:
        m = future_mask[:, 2:].float()
        return (cost * m).sum() / m.sum().clamp_min(1.0)
    return cost.mean()


def compose_guidance(
    names: List[str],
    guidance_kwargs: Optional[Dict[str, dict]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> GuidanceFn:
    """Build one cost function that is the weighted sum of several
    registered guidance functions, e.g.
        compose_guidance(["collision_avoidance", "target_speed"],
                          guidance_kwargs={"target_speed": {"target_speed": 10.0}},
                          weights={"collision_avoidance": 2.0})
    """
    guidance_kwargs = guidance_kwargs or {}
    weights = weights or {}

    missing = [n for n in names if n not in GUIDANCE_REGISTRY]
    if missing:
        raise KeyError(
            f"Unknown guidance function(s): {missing}. Available: {list(GUIDANCE_REGISTRY)}"
        )

    def combined(x: torch.Tensor, **kwargs) -> torch.Tensor:
        total = x.new_zeros(())
        for name in names:
            fn = GUIDANCE_REGISTRY[name]
            fn_kwargs = {**kwargs, **guidance_kwargs.get(name, {})}
            total = total + weights.get(name, 1.0) * fn(x, **fn_kwargs)
        return total

    return combined


def guidance_grad(
    cost_fn: GuidanceFn,
    x: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Return d(cost_fn(x)) / dx, without disturbing the caller's
    no_grad context. Used inside DiffusionModel.guided_sample, which
    otherwise runs under torch.no_grad() like a normal DDPM sampler."""
    with torch.enable_grad():
        x_ = x.detach().clone().requires_grad_(True)
        cost = cost_fn(x_, **kwargs)
        (grad,) = torch.autograd.grad(cost, x_)
    return grad