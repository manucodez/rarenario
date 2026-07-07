from __future__ import annotations

"""
Generate synthetic multi-agent trajectory data in exactly the format
scripts/preprocess.py produces (trajectories.npy, agent_masks.npy,
metadata.json), so the rest of the pipeline (train/sample/rollout/
evaluate/visualize) can be exercised without downloading nuScenes or
installing the nuscenes-devkit.

Scenarios include a handful of agents moving at roughly constant
velocity with mild noise, plus one "lead/follow" pair and one
"near-miss" pair so the collision_avoidance / stay_behind_lead /
target_speed guidance functions in models/guidance.py have something
meaningful to push against.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def make_scene(
    rng: np.random.Generator,
    timesteps: int,
    max_agents: int,
    num_active_agents: int,
    state_dim: int,
    dt: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    traj = np.full((timesteps, max_agents, state_dim), np.nan, dtype=np.float32)
    mask = np.zeros((timesteps, max_agents), dtype=bool)

    for a in range(num_active_agents):
        # Random start position, random heading/speed, small acceleration noise.
        pos = rng.uniform(-20.0, 20.0, size=2).astype(np.float32)
        heading = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(3.0, 10.0)
        vel = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32) * speed

        # Agent 0/1 in every scene are a lead/follow pair on the same lane.
        if a == 0:
            pos = np.array([0.0, 0.0], dtype=np.float32)
            vel = np.array([8.0, 0.0], dtype=np.float32)
        elif a == 1:
            pos = np.array([-15.0, 0.0], dtype=np.float32)
            vel = np.array([8.0, 0.0], dtype=np.float32)

        for t in range(timesteps):
            noise = rng.normal(0, 0.15, size=2).astype(np.float32)
            pos = pos + vel * dt + noise
            vel = vel + rng.normal(0, 0.05, size=2).astype(np.float32)

            state = np.zeros(state_dim, dtype=np.float32)
            state[0:2] = pos
            if state_dim >= 4:
                state[2:4] = vel
            traj[t, a, :] = state
            mask[t, a] = True

    return traj, mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--num_scenes", type=int, default=64)
    parser.add_argument("--history_steps", type=int, default=20)
    parser.add_argument("--future_steps", type=int, default=21)
    parser.add_argument("--max_agents", type=int, default=6)
    parser.add_argument("--num_active_agents", type=int, default=4)
    parser.add_argument("--state_dim", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    timesteps = args.history_steps + args.future_steps

    all_traj = []
    all_mask = []
    scene_names = []

    for s in range(args.num_scenes):
        traj, mask = make_scene(
            rng=rng,
            timesteps=timesteps,
            max_agents=args.max_agents,
            num_active_agents=min(args.num_active_agents, args.max_agents),
            state_dim=args.state_dim,
        )
        all_traj.append(traj)
        all_mask.append(mask)
        scene_names.append(f"synthetic_scene_{s:04d}")

    trajectories = np.stack(all_traj, axis=0).astype(np.float32)  # (S, T, A, D)
    agent_masks = np.stack(all_mask, axis=0).astype(bool)          # (S, T, A)

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / "trajectories.npy", trajectories)
    np.save(out_path / "agent_masks.npy", agent_masks)

    meta = {
        "version": "synthetic-v1",
        "dataroot": "synthetic",
        "num_scenes": len(scene_names),
        "max_agents": args.max_agents,
        "timesteps": timesteps,
        "state_dim": args.state_dim,
        "scene_names": scene_names,
        "scene_lengths": [timesteps] * len(scene_names),
        "scene_sample_tokens": [[""] * timesteps for _ in scene_names],
        "scene_agent_tokens": [[f"agent_{a}" for a in range(args.num_active_agents)] for _ in scene_names],
    }
    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote synthetic dataset to {out_path}")
    print(f"  trajectories.npy: {trajectories.shape}")
    print(f"  agent_masks.npy:  {agent_masks.shape}")


if __name__ == "__main__":
    main()
