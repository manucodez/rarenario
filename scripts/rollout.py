from __future__ import annotations

"""
Guided sampling / "scenario editing" — the CTG-defining script this repo
was missing. scripts/sample.py already does plain reverse diffusion;
this script is the same idea but routes generation through
models.guidance so you can ask for rare/controlled scenarios, e.g.:

    python scripts/rollout.py --checkpoint checkpoints/model_epoch_50.pt \
        --guidance collision_avoidance target_speed \
        --guidance_kwargs '{"target_speed": {"target_speed": 12.0}}' \
        --out_dir outputs/rollouts/near_miss_fast
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.nuscenes_dataset import NuScenesTrajectoryDataset
from models.diffusion_model import DiffusionModel
from models.guidance import GUIDANCE_REGISTRY, compose_guidance


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(model: DiffusionModel, checkpoint_path: str, device: torch.device) -> int:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        return int(ckpt.get("epoch", -1))
    model.load_state_dict(ckpt)
    return -1


def build_model_from_config(cfg: Dict[str, Any], device: torch.device) -> DiffusionModel:
    model = DiffusionModel(
        state_dim=int(cfg["state_dim"]),
        past_steps=int(cfg["past_steps"]),
        future_steps=int(cfg["future_steps"]),
        context_dim=int(cfg["context_dim"]),
        agent_hidden_dim=int(cfg["agent_hidden_dim"]),
        time_emb_dim=int(cfg["time_emb_dim"]),
        num_diffusion_steps=int(cfg["num_diffusion_steps"]),
        beta_start=float(cfg["beta_start"]),
        beta_end=float(cfg["beta_end"]),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    return model


def save_rollout(outputs: List[Dict[str, torch.Tensor]], out_dir: Path, guidance_names: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["past", "future", "guided_future", "past_mask", "future_mask"]
    arrays = {k: np.stack([o[k].numpy() for o in outputs], axis=0) for k in keys}
    np.savez_compressed(out_dir / "rollout.npz", **arrays)
    with open(out_dir / "rollout_meta.json", "w", encoding="utf-8") as f:
        json.dump({"guidance": guidance_names, "num_scenes": len(outputs)}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Guided (CTG-style) trajectory rollout / scenario editing.")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/rollouts/run1")
    parser.add_argument("--num_scenes", type=int, default=3)
    parser.add_argument(
        "--guidance", type=str, nargs="+", default=["collision_avoidance"],
        choices=list(GUIDANCE_REGISTRY.keys()),
        help="One or more registered guidance functions to combine (see models/guidance.py).",
    )
    parser.add_argument(
        "--guidance_kwargs", type=str, default="{}",
        help='JSON dict of per-guidance-function kwargs, e.g. \'{"target_speed": {"target_speed": 12.0}}\'',
    )
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--num_steps", type=int, default=None, help="Optional fewer reverse-diffusion steps for speed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    dataset = NuScenesTrajectoryDataset(
        data_root=cfg["data_root"],
        past_steps=int(cfg["past_steps"]),
        future_steps=int(cfg["future_steps"]),
        normalize=True,
    )

    model = build_model_from_config(cfg, device)
    saved_epoch = load_checkpoint(model, args.checkpoint, device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}" + (f" (epoch {saved_epoch + 1})" if saved_epoch >= 0 else ""))

    guidance_kwargs = json.loads(args.guidance_kwargs)
    guidance_fn = compose_guidance(args.guidance, guidance_kwargs=guidance_kwargs)
    print(f"Guidance: {args.guidance} (scale={args.guidance_scale})")

    num_scenes = min(args.num_scenes, len(dataset))
    outputs = []

    for idx in range(num_scenes):
        batch = dataset[idx]
        past = batch["past"].unsqueeze(0).to(device)
        future = batch["future"].unsqueeze(0).to(device)
        past_mask = batch["past_mask"].unsqueeze(0).to(device)
        future_mask = batch["future_mask"].unsqueeze(0).to(device)

        guided_future = model.guided_sample(
            past=past,
            guidance_fn=guidance_fn,
            past_mask=past_mask,
            future_mask=future_mask,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
        )

        outputs.append({
            "past": past.squeeze(0).cpu(),
            "future": future.squeeze(0).cpu(),
            "guided_future": guided_future.squeeze(0).cpu(),
            "past_mask": past_mask.squeeze(0).cpu(),
            "future_mask": future_mask.squeeze(0).cpu(),
        })
        print(f"Scene {idx}: guided_future={tuple(guided_future.shape)}")

    out_dir = Path(args.out_dir)
    save_rollout(outputs, out_dir, args.guidance)
    print(f"Saved guided rollout to: {out_dir / 'rollout.npz'}")
    print("Next: python scripts/evaluate.py --rollout " + str(out_dir / "rollout.npz"))


if __name__ == "__main__":
    main()