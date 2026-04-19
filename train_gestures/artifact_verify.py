import numpy as np
import json

data = np.load("artifacts/tap_idle_dataset.npz")
X = data["X"]
y = data["y"]

print("X shape:", X.shape)
print("y shape:", y.shape)

unique, counts = np.unique(y, return_counts=True)
print("label counts:", dict(zip(unique, counts)))

with open("artifacts/meta.json", "r") as f:
    meta = json.load(f)
print("label map:", meta["label_map"])