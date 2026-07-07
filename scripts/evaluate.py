from __future__ import annotations

"""
Quantitative evaluation of sampled/guided trajectories, analogous to what
CTG's `parse_scene_edit_results.py` does for its closed-loop rollouts:
aggregate accuracy-vs-ground-truth and rule-satisfaction metrics across a
batch of generated scenes.

Works on either:
  - outputs/samples/samples.npz      (from scripts/sample.py, key "pred_future")
  - outputs/rollouts/*/rollout.npz   (from scripts/rollout.py, key "guided_future")
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def pick_prediction_key(data: Dict[str, np.ndarray]) -> str:
    for key in ("guided_future", "pred_future"):
        if key in data:
            return key
    raise KeyError(f"No prediction array found. Keys present: {list(data.keys())}")


def masked_displacement(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """pred, gt: (S, T, A, D); mask: (S, T, A). Returns per-timestep L2
    displacement error on positions (first two channels), NaN where
    invalid."""
    err = np.linalg.norm(pred[..., :2] - gt[..., :2], axis=-1)  # (S, T, A)
    err = np.where(mask, err, np.nan)
    return err


def compute_ade_fde(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    disp = masked_displacement(pred, gt, mask)             # (S, T, A)
    ade = np.nanmean(disp)

    last_valid_idx = mask.shape[1] - 1 - np.argmax(mask[:, ::-1, :], axis=1)  # (S, A)
    has_any = mask.any(axis=1)                                                # (S, A)

    fde_vals = []
    S, T, A = mask.shape
    for s in range(S):
        for a in range(A):
            if not has_any[s, a]:
                continue
            t = last_valid_idx[s, a]
            fde_vals.append(disp[s, t, a])
    fde = float(np.mean(fde_vals)) if fde_vals else float("nan")

    return {"ADE": float(ade), "FDE": fde}


def compute_collision_rate(pred: np.ndarray, mask: np.ndarray, min_dist: float = 2.0) -> float:
    """Fraction of (scene, timestep) with at least one pair of agents
    closer than `min_dist` in the generated trajectories."""
    S, T, A, _ = pred.shape
    pos = pred[..., :2]
    violations = 0
    total = 0
    for s in range(S):
        for t in range(T):
            valid_agents = np.where(mask[s, t])[0]
            if len(valid_agents) < 2:
                continue
            total += 1
            p = pos[s, t, valid_agents]                    # (a_valid, 2)
            diff = p[:, None, :] - p[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dist, np.inf)
            if (dist < min_dist).any():
                violations += 1
    return violations / total if total > 0 else float("nan")


def compute_speed_stats(pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Mean/std speed (from vx, vy channels) over valid entries, useful to
    sanity-check e.g. a target_speed guidance run."""
    if pred.shape[-1] < 4:
        return {"mean_speed": float("nan"), "std_speed": float("nan")}
    speed = np.linalg.norm(pred[..., 2:4], axis=-1)
    valid = speed[mask]
    return {"mean_speed": float(valid.mean()), "std_speed": float(valid.std())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=str, required=True,
        help="Path to samples.npz (scripts/sample.py) or rollout.npz (scripts/rollout.py)",
    )
    parser.add_argument("--min_dist", type=float, default=2.0, help="Collision distance threshold (meters)")
    parser.add_argument("--out", type=str, default=None, help="Optional path to write metrics JSON")
    args = parser.parse_args()

    results_path = Path(args.results)
    data = load_npz(results_path)
    pred_key = pick_prediction_key(data)

    pred = data[pred_key]              # (S, T, A, D)
    gt = data["future"]                # (S, T, A, D)
    mask = data["future_mask"].astype(bool)

    metrics = {}
    metrics.update(compute_ade_fde(pred, gt, mask))
    metrics["collision_rate"] = compute_collision_rate(pred, mask, min_dist=args.min_dist)
    metrics.update(compute_speed_stats(pred, mask))
    metrics["num_scenes"] = int(pred.shape[0])
    metrics["prediction_key"] = pred_key

    print(f"Evaluated: {results_path}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    out_path = Path(args.out) if args.out else results_path.parent / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()