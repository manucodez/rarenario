import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.diffusion_model import DiffusionModel

model = DiffusionModel(past_steps=20, future_steps=21)
print(model)
