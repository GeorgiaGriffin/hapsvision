import numpy as np
stats = np.load("artifacts/run_20260419_014037/scaler_stats.npz")  # use your actual run dir
print("static const float NORM_MEAN[6] = {", ", ".join(f"{v:.6f}f" for v in stats["mean"]), "};")
print("static const float NORM_STD[6]  = {", ", ".join(f"{v:.6f}f" for v in stats["std"]),  "};")