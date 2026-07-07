from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Make sure project root is on sys.path when running:
# python scripts/train.py
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.nuscenes_dataset import NuScenesTrajectoryDataset
from models.diffusion_model import DiffusionModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(
    ckpt_path: Path,
    model: DiffusionModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Dict[str, Any],
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        ckpt_path,
    )


def train_one_epoch(
    model: DiffusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        past = batch["past"].to(device)
        future = batch["future"].to(device)
        past_mask = batch["past_mask"].to(device)
        future_mask = batch["future_mask"].to(device)

        batch_size = past.shape[0]
        t = torch.randint(
            low=0,
            high=model.num_diffusion_steps,
            size=(batch_size,),
            device=device,
        )

        loss, metrics = model.p_losses(
            x_start=future,
            t=t,
            past=past,
            past_mask=past_mask,
            future_mask=future_mask,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += float(loss.item())
        num_batches += 1

    return running_loss / max(num_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CTG-style diffusion model on nuScenes mini.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Optional checkpoint path to resume from",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device_name = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    data_root = cfg["data_root"]
    past_steps = int(cfg["past_steps"])
    future_steps = int(cfg["future_steps"])
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_workers", 0))

    dataset = NuScenesTrajectoryDataset(
        data_root=data_root,
        past_steps=past_steps,
        future_steps=future_steps,
        normalize=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = DiffusionModel(
        state_dim=int(cfg["state_dim"]),
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(cfg["epochs"])
    save_every = int(cfg.get("save_every", 5))

    print("Starting training")
    print(f"Device: {device}")
    print(f"Scenes: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print("-----")

    history = []

    for epoch in range(start_epoch, epochs):
        avg_loss = train_one_epoch(model, loader, optimizer, device)
        history.append({"epoch": epoch, "loss": avg_loss})

        print(f"Epoch {epoch + 1:03d}/{epochs} | loss = {avg_loss:.6f}")

        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            ckpt_path = checkpoint_dir / f"model_epoch_{epoch + 1}.pt"
            save_checkpoint(ckpt_path, model, optimizer, epoch, cfg)
            print(f"Saved checkpoint: {ckpt_path}")

    with open(checkpoint_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    main()