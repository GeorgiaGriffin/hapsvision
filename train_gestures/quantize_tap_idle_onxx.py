import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from esp_ppq.api import espdl_quantize_onnx

# CHANGE THIS to the NEW run folder produced by retraining/export
RUN_DIR = r"artifacts\run_20260419_014037"

ONNX_MODEL_PATH = os.path.join(RUN_DIR, "tap_idle.onnx")
ESPDL_MODEL_PATH = os.path.join(RUN_DIR, "tap_idle.espdl")
DATASET_PATH = os.path.join(RUN_DIR, "tap_idle_dataset.npz")
SCALER_PATH = os.path.join(RUN_DIR, "scaler_stats.npz")

# NEW flat input shape
INPUT_SHAPE = [1, 600]

TARGET = "esp32s3"
NUM_OF_BITS = 8
DEVICE = "cpu"

def collate_fn(batch):
    batch_x = batch[0].to(DEVICE)
    return batch_x

def main():
    data = np.load(DATASET_PATH)
    X = data["X"].astype(np.float32)   # raw saved dataset is still [N, 100, 6]

    scaler = np.load(SCALER_PATH)
    mean = scaler["mean"].astype(np.float32)   # [6]
    std = scaler["std"].astype(np.float32)     # [6]

    # same normalization as training
    X = (X - mean[None, None, :]) / std[None, None, :]

    # flatten to [N, 600] to match the new model
    X = X.reshape(X.shape[0], -1).astype(np.float32)

    x_tensor = torch.tensor(X, dtype=torch.float32)
    dummy_y = torch.zeros((len(X),), dtype=torch.long)
    dataset = TensorDataset(x_tensor, dummy_y)

    # batch size must stay 1 for esp-dl quantization/deployment
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Quantizing: {ONNX_MODEL_PATH}")
    print(f"Exporting:  {ESPDL_MODEL_PATH}")

    _ = espdl_quantize_onnx(
        onnx_import_file=ONNX_MODEL_PATH,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=min(32, len(dataloader)),
        input_shape=INPUT_SHAPE,
        inputs=None,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        collate_fn=collate_fn,
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
    )

    print("Done.")
    print("Expected outputs:")
    print(" -", ESPDL_MODEL_PATH)
    print(" -", ESPDL_MODEL_PATH.replace(".espdl", ".info"))
    print(" -", ESPDL_MODEL_PATH.replace(".espdl", ".json"))

if __name__ == "__main__":
    main()