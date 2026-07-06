import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader

from datasets.nuscenes_dataset import NuScenesTrajectoryDataset

dataset = NuScenesTrajectoryDataset(
    data_root="data/processed",
    past_steps=20,
    future_steps=21,
)

print("Dataset size:", len(dataset))

sample = dataset[0]

for k, v in sample.items():
    print(k, v.shape if hasattr(v, "shape") else v)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
)

batch = next(iter(loader))

print()

print("Batch shapes")

for k, v in batch.items():
    print(k, v.shape)