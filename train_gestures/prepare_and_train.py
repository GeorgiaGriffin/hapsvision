import os
import glob
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = "data"
CLASSES = ["idle", "tap"]

WINDOW = 100          # 100 samples ~= 1.0 s at 100 Hz
STRIDE = 25           # overlap for more data
TAP_CROP = 160        # crop around strongest motion
MIN_ROWS = 80

FEATURE_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]


def load_csv(path):
    df = pd.read_csv(path)

    expected = ["time_ms", "label", "ax", "ay", "az", "gx", "gy", "gz"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df = df.dropna(subset=expected).copy()
    return df


def split_on_time_reset(df):
    """
    If time_ms goes backwards because of reset / reconnect,
    split into separate sessions.
    """
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
    """
    Simple motion score for peak-finding.
    For taps, gyro is usually most informative.
    """
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

    score = gyro_mag + 0.25 * accel_change
    return score


def crop_tap_session(df, crop_len=TAP_CROP):
    """
    Automatically keep the highest-motion region in a tap file.
    """
    score = motion_energy(df)
    peak_idx = int(np.argmax(score))

    half = crop_len // 2
    start = max(0, peak_idx - half)
    end = min(len(df), start + crop_len)

    # re-adjust start if near end
    start = max(0, end - crop_len)

    cropped = df.iloc[start:end].reset_index(drop=True)
    return cropped


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


def train_baseline(X, y):
    """
    Start with a simple MLP on flattened windows.
    This is easy and fast.
    """
    n, t, c = X.shape
    X_flat = X.reshape(n, t * c)

    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=300,
        random_state=42
    )
    clf.fit(X_train_sc, y_train)

    y_pred = clf.predict(X_test_sc)

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, target_names=CLASSES))
    print("=== Confusion matrix ===")
    print(confusion_matrix(y_test, y_pred))

    return clf, scaler


def save_outputs(X, y, meta, label_map, clf, scaler):
    os.makedirs("artifacts", exist_ok=True)

    np.savez_compressed(
        "artifacts/tap_idle_dataset.npz",
        X=X,
        y=y
    )

    with open("artifacts/meta.json", "w") as f:
        json.dump(
            {
                "label_map": label_map,
                "window": WINDOW,
                "stride": STRIDE,
                "tap_crop": TAP_CROP,
                "num_samples": int(len(y)),
                "meta_preview": meta[:20]
            },
            f,
            indent=2
        )

    joblib.dump(clf, "artifacts/tap_idle_mlp.joblib")
    joblib.dump(scaler, "artifacts/tap_idle_scaler.joblib")

    print("\n[INFO] Saved:")
    print("  artifacts/tap_idle_dataset.npz")
    print("  artifacts/meta.json")
    print("  artifacts/tap_idle_mlp.joblib")
    print("  artifacts/tap_idle_scaler.joblib")


def main():
    X, y, meta, label_map = collect_windows()

    if len(X) == 0:
        raise RuntimeError("No usable windows found. Check your CSV folders.")

    print(f"\n[INFO] Dataset shape: X={X.shape}, y={y.shape}")
    counts = {CLASSES[i]: int((y == i).sum()) for i in range(len(CLASSES))}
    print(f"[INFO] Class counts: {counts}")

    clf, scaler = train_baseline(X, y)
    save_outputs(X, y, meta, label_map, clf, scaler)


if __name__ == "__main__":
    main()