import os
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = "data"
CLASSES = ["idle", "tap"]

WINDOW = 100          # 100 samples ~= 1.0 s at 100 Hz
STRIDE = 25           # overlap for more data
TAP_CROP = 160        # crop around strongest motion
MIN_ROWS = 80

FEATURE_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]

BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

RUN_DIR = os.path.join(
    "artifacts",
    f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_csv(path):
    df = pd.read_csv(path)

    expected = ["time_ms", "label", "ax", "ay", "az", "gx", "gy", "gz"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df = df.dropna(subset=expected).copy()
    return df


def split_on_time_reset(df):
    sessions = []
    start = 0
    t = df["time_ms"].to_numpy()

    for i in range(1, len(df)):
        if t[i] < t[i - 1]:
            part = df.iloc[start:i].reset_index(drop=True)
            if len(part) >= MIN_ROWS:
                sessions.append(part)
            start = i

    part = df.iloc[start:].reset_index(drop=True)
    if len(part) >= MIN_ROWS:
        sessions.append(part)

    return sessions


def motion_energy(df):
    gx = df["gx"].to_numpy(dtype=np.float32)
    gy = df["gy"].to_numpy(dtype=np.float32)
    gz = df["gz"].to_numpy(dtype=np.float32)

    ax = df["ax"].to_numpy(dtype=np.float32)
    ay = df["ay"].to_numpy(dtype=np.float32)
    az = df["az"].to_numpy(dtype=np.float32)

    gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

    dax = np.diff(ax, prepend=ax[0])
    day = np.diff(ay, prepend=ay[0])
    daz = np.diff(az, prepend=az[0])
    accel_change = np.sqrt(dax * dax + day * day + daz * daz)

    return gyro_mag + 0.25 * accel_change


def crop_tap_session(df, crop_len=TAP_CROP):
    score = motion_energy(df)
    peak_idx = int(np.argmax(score))

    half = crop_len // 2
    start = max(0, peak_idx - half)
    end = min(len(df), start + crop_len)
    start = max(0, end - crop_len)

    return df.iloc[start:end].reset_index(drop=True)


def sliding_windows(arr, window=WINDOW, stride=STRIDE):
    xs = []
    for start in range(0, len(arr) - window + 1, stride):
        xs.append(arr[start:start + window])
    return xs


def collect_windows():
    X = []
    y = []
    meta = []

    label_map = {name: idx for idx, name in enumerate(CLASSES)}

    for class_name in CLASSES:
        paths = sorted(glob.glob(os.path.join(BASE_DIR, class_name, "*.csv")))
        print(f"[INFO] {class_name}: found {len(paths)} files")

        for path in paths:
            try:
                df = load_csv(path)
            except Exception as e:
                print(f"[WARN] skipping {path}: {e}")
                continue

            sessions = split_on_time_reset(df)

            for sess_idx, sess in enumerate(sessions):
                if class_name == "tap":
                    sess = crop_tap_session(sess, crop_len=TAP_CROP)

                arr = sess[FEATURE_COLS].to_numpy(dtype=np.float32)
                windows = sliding_windows(arr, WINDOW, STRIDE)

                for w in windows:
                    X.append(w)
                    y.append(label_map[class_name])
                    meta.append({
                        "file": path,
                        "class": class_name,
                        "session": sess_idx
                    })

    X = np.array(X, dtype=np.float32)   # [N, T, C]
    y = np.array(y, dtype=np.int64)
    return X, y, meta, label_map


def fit_scaler(X_train):
    # X_train: [N, T, C]
    flat = X_train.reshape(-1, X_train.shape[-1])  # [N*T, C]
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_scaler(X, mean, std):
    # X is [N, T, C]
    Xn = (X - mean[None, None, :]) / std[None, None, :]
    return Xn.reshape(Xn.shape[0], -1).astype(np.float32)   # [N, 600]


class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TinyTapMLP(nn.Module):
    def __init__(self, in_dim=WINDOW * len(FEATURE_COLS), num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

            y_true.extend(yb.numpy().tolist())
            y_pred.extend(pred.tolist())

    return np.array(y_true), np.array(y_pred)


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    mean, std = fit_scaler(X_train)
    X_train = apply_scaler(X_train, mean, std)
    X_test = apply_scaler(X_test, mean, std)

    train_ds = WindowDataset(X_train, y_train)
    test_ds = WindowDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTapMLP().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        avg_loss = running_loss / len(train_ds)
        print(f"[INFO] epoch {epoch+1:02d}/{EPOCHS} loss={avg_loss:.4f}")

    y_true, y_pred = evaluate(model, test_loader, device)

    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    print("=== Confusion matrix ===")
    print(confusion_matrix(y_true, y_pred))

    return model, mean, std, (X_test, y_test)


def save_outputs(X, y, meta, label_map, model, mean, std):
    os.makedirs(RUN_DIR, exist_ok=True)

    np.savez_compressed(
        os.path.join(RUN_DIR, "tap_idle_dataset.npz"),
        X=X,
        y=y
    )

    with open(os.path.join(RUN_DIR, "meta.json"), "w") as f:
        json.dump(
            {
                "label_map": label_map,
                "window": WINDOW,
                "stride": STRIDE,
                "tap_crop": TAP_CROP,
                "num_samples": int(len(y)),
                "feature_cols": FEATURE_COLS,
                "meta_preview": meta[:20]
            },
            f,
            indent=2
        )

    np.savez(
        os.path.join(RUN_DIR, "scaler_stats.npz"),
        mean=mean,
        std=std
    )

    torch.save(
        model.state_dict(),
        os.path.join(RUN_DIR, "tap_idle_torch_state_dict.pt")
    )

    # Optional: export ONNX for ESP-PPQ flow
    model.eval()
    dummy = torch.randn(1, WINDOW * len(FEATURE_COLS), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        os.path.join(RUN_DIR, "tap_idle.onnx"),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
        opset_version=13
    )

    print("\n[INFO] Saved:")
    print(f"  {RUN_DIR}/tap_idle_dataset.npz")
    print(f"  {RUN_DIR}/meta.json")
    print(f"  {RUN_DIR}/scaler_stats.npz")
    print(f"  {RUN_DIR}/tap_idle_torch_state_dict.pt")
    print(f"  {RUN_DIR}/tap_idle.onnx")


def main():
    X, y, meta, label_map = collect_windows()

    if len(X) == 0:
        raise RuntimeError("No usable windows found. Check your CSV folders.")

    print(f"\n[INFO] Dataset shape: X={X.shape}, y={y.shape}")
    counts = {CLASSES[i]: int((y == i).sum()) for i in range(len(CLASSES))}
    print(f"[INFO] Class counts: {counts}")

    model, mean, std, _ = train_model(X, y)
    save_outputs(X, y, meta, label_map, model, mean, std)


if __name__ == "__main__":
    main()