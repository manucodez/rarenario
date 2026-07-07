from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
import sys

# Ensure project root is importable when running:
# python scripts/sample.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.nuscenes_dataset import NuScenesTrajectoryDataset
from models.diffusion_model import DiffusionModel


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(model: DiffusionModel, checkpoint_path: str, device: torch.device) -> int:
    """
    Supports both:
      1) raw state_dict checkpoints
      2) dict checkpoints created by scripts/train.py
    Returns the saved epoch if available, else -1.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        return int(ckpt.get("epoch", -1))
    else:
        model.load_state_dict(ckpt)
        return -1


@torch.no_grad()
def sample_batch(
    model: DiffusionModel,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    past = batch["past"].to(device)
    future = batch["future"].to(device)
    past_mask = batch["past_mask"].to(device)
    future_mask = batch["future_mask"].to(device)

    pred_future = model.sample(
        past=past,
        past_mask=past_mask,
        future_mask=future_mask,
    )

    return {
        "past": past.squeeze(0).cpu(),
        "future": future.squeeze(0).cpu(),
        "pred_future": pred_future.squeeze(0).cpu(),
        "past_mask": past_mask.squeeze(0).cpu(),
        "future_mask": future_mask.squeeze(0).cpu(),
    }


def save_scene_outputs(
    outputs: List[Dict[str, torch.Tensor]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_past = []
    all_future = []
    all_pred = []
    all_past_mask = []
    all_future_mask = []

    for item in outputs:
        all_past.append(item["past"].numpy())
        all_future.append(item["future"].numpy())
        all_pred.append(item["pred_future"].numpy())
        all_past_mask.append(item["past_mask"].numpy())
        all_future_mask.append(item["future_mask"].numpy())

    np.savez_compressed(
        out_dir / "samples.npz",
        past=np.array(all_past, dtype=np.float32),
        future=np.array(all_future, dtype=np.float32),
        pred_future=np.array(all_pred, dtype=np.float32),
        past_mask=np.array(all_past_mask, dtype=bool),
        future_mask=np.array(all_future_mask, dtype=bool),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample trajectories from a trained diffusion model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint, e.g. checkpoints/model_epoch_50.pt",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/samples",
        help="Directory to store sampled outputs",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=3,
        help="How many scenes to sample from the dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu or cuda",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    past_steps = int(cfg["past_steps"])
    future_steps = int(cfg["future_steps"])
    state_dim = int(cfg["state_dim"])

    dataset = NuScenesTrajectoryDataset(
        data_root=cfg["data_root"],
        past_steps=past_steps,
        future_steps=future_steps,
        normalize=True,
    )

    model = DiffusionModel(
        state_dim=state_dim,
        past_steps=past_steps,
        future_steps=future_steps,
        context_dim=int(cfg["context_dim"]),
        agent_hidden_dim=int(cfg["agent_hidden_dim"]),
        time_emb_dim=int(cfg["time_emb_dim"]),
        num_diffusion_steps=int(cfg["num_diffusion_steps"]),
        beta_start=float(cfg["beta_start"]),
        beta_end=float(cfg["beta_end"]),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)

    saved_epoch = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    print(f"Loaded checkpoint: {args.checkpoint}")
    if saved_epoch >= 0:
        print(f"Saved epoch: {saved_epoch + 1}")

    outputs: List[Dict[str, torch.Tensor]] = []

    num_scenes = min(args.num_scenes, len(dataset))
    print(f"Sampling {num_scenes} scenes...")

    for idx in range(num_scenes):
        batch = dataset[idx]
        # Make a batch dimension of 1 so the model.sample() API is consistent.
        batch = {
            "past": batch["past"].unsqueeze(0),
            "future": batch["future"].unsqueeze(0),
            "past_mask": batch["past_mask"].unsqueeze(0),
            "future_mask": batch["future_mask"].unsqueeze(0),
        }

        sampled = sample_batch(model, batch, device)
        outputs.append(sampled)

        print(
            f"Scene {idx}: "
            f"past={tuple(sampled['past'].shape)}, "
            f"future={tuple(sampled['future'].shape)}, "
            f"pred={tuple(sampled['pred_future'].shape)}"
        )

    out_dir = Path(args.out_dir)
    save_scene_outputs(outputs, out_dir)

    print(f"Saved sampled outputs to: {out_dir / 'samples.npz'}")


if __name__ == "__main__":
    main()