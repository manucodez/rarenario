from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import sys

# Ensure project root is importable when running:
# python scripts/visualize.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.nuscenes_dataset import NuScenesTrajectoryDataset


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_samples(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    return {
        "past": data["past"],
        "future": data["future"],
        "pred_future": data["pred_future"],
        "past_mask": data["past_mask"],
        "future_mask": data["future_mask"],
    }


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def denormalize_array(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    arr: (..., D)
    mean/std: (D,)
    """
    return arr * std.reshape(1, 1, 1, -1) + mean.reshape(1, 1, 1, -1)


def agent_validity_mask(past_mask: np.ndarray, future_mask: np.ndarray) -> np.ndarray:
    """
    Return agents that appear in at least one timestep.
    past_mask: (T_past, A)
    future_mask: (T_future, A)
    """
    past_present = past_mask.any(axis=0)
    future_present = future_mask.any(axis=0)
    return past_present | future_present


def plot_scene(
    past: np.ndarray,
    future: np.ndarray,
    pred_future: np.ndarray,
    past_mask: np.ndarray,
    future_mask: np.ndarray,
    out_path: Path,
    scene_idx: int = 0,
    max_agents: Optional[int] = 20,
):
    """
    past: (T_past, A, D)
    future: (T_future, A, D)
    pred_future: (T_future, A, D)
    past_mask: (T_past, A)
    future_mask: (T_future, A)
    """
    past_xy = past[..., :2]
    future_xy = future[..., :2]
    pred_xy = pred_future[..., :2]

    valid_agents = agent_validity_mask(past_mask, future_mask)
    agent_ids = np.where(valid_agents)[0]

    if max_agents is not None:
        agent_ids = agent_ids[:max_agents]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    panels = [
        (axes[0], "Past trajectory", past_xy, past_mask),
        (axes[1], "Ground truth future", future_xy, future_mask),
        (axes[2], "Predicted future", pred_xy, future_mask),
    ]

    for ax, title, xy, mask in panels:
        for agent_id in agent_ids:
            coords = xy[:, agent_id, :]
            agent_mask = mask[:, agent_id]

            if not agent_mask.any():
                continue

            coords = coords[agent_mask]

            if coords.shape[0] < 2:
                # Plot single point if only one timestep exists
                ax.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.7)
                continue

            ax.plot(coords[:, 0], coords[:, 1], linewidth=1.5, alpha=0.85)

            # Mark start and end
            ax.scatter(coords[0, 0], coords[0, 1], s=14, marker="o")
            ax.scatter(coords[-1, 0], coords[-1, 1], s=14, marker="x")

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(f"Scene {scene_idx}", fontsize=14)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize CTG-style trajectory samples.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="outputs/samples/samples.npz",
        help="Path to samples.npz produced by scripts/sample.py",
    )
    parser.add_argument(
        "--scene_idx",
        type=int,
        default=0,
        help="Which sampled scene to visualize",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/plots/scene_0.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--max_agents",
        type=int,
        default=20,
        help="Maximum number of agents to plot",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    samples = load_samples(Path(args.samples))

    dataset = NuScenesTrajectoryDataset(
        data_root=cfg["data_root"],
        past_steps=int(cfg["past_steps"]),
        future_steps=int(cfg["future_steps"]),
        normalize=True,
    )

    stats = dataset.get_statistics()
    if stats is None:
        raise RuntimeError("Dataset statistics are missing. Enable normalize=True in the dataset.")

    mean = stats["mean"].numpy()
    std = stats["std"].numpy()

    scene_idx = args.scene_idx
    if scene_idx < 0 or scene_idx >= samples["past"].shape[0]:
        raise IndexError(
            f"scene_idx={scene_idx} is out of range for {samples['past'].shape[0]} sampled scenes."
        )

    past = samples["past"][scene_idx]          # (1, T_past, A, D)
    future = samples["future"][scene_idx]      # (1, T_future, A, D)
    pred = samples["pred_future"][scene_idx]   # (1, T_future, A, D)
    past_mask = samples["past_mask"][scene_idx]
    future_mask = samples["future_mask"][scene_idx]

    # Remove batch dimension
    past = np.squeeze(past, axis=0)
    future = np.squeeze(future, axis=0)
    pred = np.squeeze(pred, axis=0)

    # Denormalize to original coordinate space for plotting
    past = denormalize_array(past[None, ...], mean, std)[0]
    future = denormalize_array(future[None, ...], mean, std)[0]
    pred = denormalize_array(pred[None, ...], mean, std)[0]

    out_path = Path(args.out)
    plot_scene(
        past=past,
        future=future,
        pred_future=pred,
        past_mask=past_mask,
        future_mask=future_mask,
        out_path=out_path,
        scene_idx=scene_idx,
        max_agents=args.max_agents,
    )

    print(f"Saved visualization to: {out_path}")


if __name__ == "__main__":
    main()