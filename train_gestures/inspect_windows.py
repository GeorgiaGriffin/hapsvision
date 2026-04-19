import numpy as np
import matplotlib.pyplot as plt

data = np.load("artifacts/tap_idle_dataset.npz")
X = data["X"]
y = data["y"]

idle_idxs = np.where(y == 0)[0][:3]
tap_idxs = np.where(y == 1)[0][:3]

def plot_sample(idx):
    fig, axes = plt.subplots(6, 1, figsize=(10, 8), sharex=True)
    names = ["ax", "ay", "az", "gx", "gy", "gz"]
    for i in range(6):
        axes[i].plot(X[idx, :, i])
        axes[i].set_ylabel(names[i])
    axes[0].set_title(f"sample {idx}, label={y[idx]}")
    axes[-1].set_xlabel("time step")
    plt.tight_layout()
    plt.show()

print("Idle indices:", idle_idxs)
print("Tap indices:", tap_idxs)

for idx in idle_idxs:
    plot_sample(idx)

for idx in tap_idxs:
    plot_sample(idx)