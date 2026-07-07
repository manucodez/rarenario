from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


DYNAMIC_PREFIXES = (
    "vehicle.",
    "human.pedestrian.",
    "bicycle.",
    "motorcycle.",
)


def is_dynamic_agent(category_name: str) -> bool:
    return category_name.startswith(DYNAMIC_PREFIXES)


def yaw_from_quaternion(rotation) -> float:
    """
    nuScenes stores quaternions as [w, x, y, z].
    """
    try:
        q = Quaternion(rotation)
        return float(q.yaw_pitch_roll[0])
    except Exception:
        return 0.0


def safe_box_velocity(nusc: NuScenes, ann_token: str) -> np.ndarray:
    """
    Returns a 2D velocity vector [vx, vy] in global coordinates.
    Falls back to zeros if velocity is unavailable.
    """
    try:
        vel = np.array(nusc.box_velocity(ann_token), dtype=np.float32).reshape(-1)
        if vel.size >= 2 and np.all(np.isfinite(vel[:2])):
            return vel[:2]
    except Exception:
        pass
    return np.zeros(2, dtype=np.float32)


def get_scene_sample_tokens(nusc: NuScenes, scene: dict) -> List[str]:
    """
    Traverse a scene from first sample to last sample.
    """
    tokens: List[str] = []
    sample_token = scene["first_sample_token"]
    while sample_token:
        tokens.append(sample_token)
        sample = nusc.get("sample", sample_token)
        sample_token = sample["next"]
    return tokens


def build_scene_tensor(
    nusc: NuScenes,
    scene: dict,
    max_agents: int,
    state_dim: int = 4,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Build one scene tensor of shape (T, A, D).

    State layout:
        D = 4 -> [x, y, vx, vy]

    Missing agents/frames are padded with NaN.
    """
    sample_tokens = get_scene_sample_tokens(nusc, scene)
    timesteps = len(sample_tokens)

    # First pass: collect all dynamic agents and their presence statistics.
    per_timestep_agents: List[Dict[str, dict]] = []
    first_seen: Dict[str, int] = {}
    counts: Dict[str, int] = defaultdict(int)
    categories: Dict[str, str] = {}

    for t, sample_token in enumerate(sample_tokens):
        sample = nusc.get("sample", sample_token)
        current_agents: Dict[str, dict] = {}

        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            if not is_dynamic_agent(ann["category_name"]):
                continue

            instance_token = ann["instance_token"]
            current_agents[instance_token] = ann

            counts[instance_token] += 1
            categories[instance_token] = ann["category_name"]
            if instance_token not in first_seen:
                first_seen[instance_token] = t

        per_timestep_agents.append(current_agents)

    if not first_seen:
        traj = np.full((timesteps, max_agents, state_dim), np.nan, dtype=np.float32)
        mask = np.zeros((timesteps, max_agents), dtype=bool)
        return traj, mask, sample_tokens, []

    # Rank agents by how often they appear, then by first appearance.
    ranked_instances = sorted(
        first_seen.keys(),
        key=lambda tok: (-counts[tok], first_seen[tok], tok),
    )
    selected_instances = ranked_instances[:max_agents]

    instance_to_index = {tok: i for i, tok in enumerate(selected_instances)}

    traj = np.full((timesteps, max_agents, state_dim), np.nan, dtype=np.float32)
    mask = np.zeros((timesteps, max_agents), dtype=bool)

    for t, agents_in_frame in enumerate(per_timestep_agents):
        for instance_token, ann in agents_in_frame.items():
            if instance_token not in instance_to_index:
                continue

            a = instance_to_index[instance_token]

            x, y = ann["translation"][:2]
            vx, vy = safe_box_velocity(nusc, ann["token"])

            state = np.array([x, y, vx, vy], dtype=np.float32)
            traj[t, a, :state_dim] = state[:state_dim]
            mask[t, a] = True

    agent_tokens = selected_instances
    return traj, mask, sample_tokens, agent_tokens


def preprocess_nuscenes(
    dataroot: str,
    out_dir: str,
    version: str = "v1.0-mini",
    max_agents: int = 64,
    max_timesteps: int | None = None,
    history_steps: int = 20,
    future_steps: int = 21,
    stride: int = 1,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    scenes = list(nusc.scene)
    if not scenes:
        raise RuntimeError("No scenes found in nuScenes dataset.")

    # Decide global temporal length.
    scene_lengths = []
    for scene in scenes:
        sample_tokens = get_scene_sample_tokens(nusc, scene)
        scene_lengths.append(len(sample_tokens))

    global_timesteps = max(scene_lengths) if max_timesteps is None else min(max(scene_lengths), max_timesteps)

    all_trajs = []
    all_masks = []
    scene_names = []
    scene_sample_tokens = []
    scene_agent_tokens = []
    scene_lengths_kept = []

    for scene in scenes:
        traj, mask, sample_tokens, agent_tokens = build_scene_tensor(
            nusc=nusc,
            scene=scene,
            max_agents=max_agents,
            state_dim=4,
        )

        # Truncate or pad to global_timesteps.
        if traj.shape[0] > global_timesteps:
            traj = traj[:global_timesteps]
            mask = mask[:global_timesteps]
            sample_tokens = sample_tokens[:global_timesteps]
        elif traj.shape[0] < global_timesteps:
            pad_t = global_timesteps - traj.shape[0]
            traj_pad = np.full((pad_t, traj.shape[1], traj.shape[2]), np.nan, dtype=np.float32)
            mask_pad = np.zeros((pad_t, mask.shape[1]), dtype=bool)
            traj = np.concatenate([traj, traj_pad], axis=0)
            mask = np.concatenate([mask, mask_pad], axis=0)
            sample_tokens = sample_tokens + [""] * pad_t

        window_size = history_steps + future_steps
        if traj.shape[0] < window_size:
            pad_t = window_size - traj.shape[0]
            traj_pad = np.full((pad_t, traj.shape[1], traj.shape[2]), np.nan, dtype=np.float32)
            mask_pad = np.zeros((pad_t, mask.shape[1]), dtype=bool)
            traj = np.concatenate([traj, traj_pad], axis=0)
            mask = np.concatenate([mask, mask_pad], axis=0)
            sample_tokens = sample_tokens + [""] * pad_t

        for start in range(0, traj.shape[0] - window_size + 1, stride):
            window_traj = traj[start : start + window_size]
            window_mask = mask[start : start + window_size]
            window_tokens = sample_tokens[start : start + window_size]

            all_trajs.append(window_traj)
            all_masks.append(window_mask)
            scene_names.append(scene["name"])
            scene_sample_tokens.append(window_tokens)
            scene_agent_tokens.append(agent_tokens)
            scene_lengths_kept.append(window_size)

        print(f"Processed {scene['name']}: {traj.shape[0]} timesteps, {len(agent_tokens)} agents")

    trajectories = np.stack(all_trajs, axis=0).astype(np.float32)   # (S, T, A, 4)
    agent_masks = np.stack(all_masks, axis=0).astype(bool)           # (S, T, A)

    np.save(out_path / "trajectories.npy", trajectories)
    np.save(out_path / "agent_masks.npy", agent_masks)

    meta = {
        "version": version,
        "dataroot": str(dataroot),
        "num_scenes": len(scene_names),
        "max_agents": max_agents,
        "timesteps": int(global_timesteps),
        "state_dim": 4,
        "scene_names": scene_names,
        "scene_lengths": scene_lengths_kept,
        "scene_sample_tokens": scene_sample_tokens,
        "scene_agent_tokens": scene_agent_tokens,
    }

    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved:")
    print(f"  {out_path / 'trajectories.npy'}")
    print(f"  {out_path / 'agent_masks.npy'}")
    print(f"  {out_path / 'metadata.json'}")
    print(f"Shape: {trajectories.shape}  -> (S, T, A, D)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess nuScenes mini into trajectory tensors.")
    parser.add_argument("--dataroot", type=str, required=True, help="Path to data/nuscenes")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="nuScenes version")
    parser.add_argument("--max_agents", type=int, default=64, help="Max agents per scene")
    parser.add_argument(
        "--max_timesteps",
        type=int,
        default=None,
        help="Optional cap on timesteps per scene. Default: max scene length",
    )
    parser.add_argument("--history_steps", type=int, default=20, help="Number of history timesteps per sample")
    parser.add_argument("--future_steps", type=int, default=21, help="Number of future timesteps per sample")
    parser.add_argument("--stride", type=int, default=1, help="Stride between sliding windows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_nuscenes(
        dataroot=args.dataroot,
        out_dir=args.out_dir,
        version=args.version,
        max_agents=args.max_agents,
        max_timesteps=args.max_timesteps,
        history_steps=args.history_steps,
        future_steps=args.future_steps,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()