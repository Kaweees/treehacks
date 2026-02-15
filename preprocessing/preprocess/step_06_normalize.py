"""
Step 06: Z-score normalize each feature column.

Uses the mask from step_05 to compute mean and std only from real
(non-masked) values. Masked positions stay 0 after normalization.

Run directly:
    python -m preprocess.step_06_normalize
"""

import os
import numpy as np

PREPROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preprocessed")
INPUT_PATH = os.path.join(PREPROCESSED_DIR, "dataset_3class_masked.npz")
OUTPUT_PATH = os.path.join(PREPROCESSED_DIR, "dataset_3class_norm.npz")


def main():
    data = np.load(INPUT_PATH, allow_pickle=True)
    features = data["features"]  # (226800, 726) float32, NaN replaced with 0
    mask = data["mask"]           # (226800, 726) int8, 1=real 0=missing
    labels = data["labels"]
    subjects = data["subjects"]
    trials = data["trials"]

    print(f"Input: {features.shape}")
    real_pct = 100 * mask.sum() / mask.size
    print(f"Mask: {real_pct:.1f}% real values")

    # Compute mean/std only from real (mask==1) values per column
    mask_f = mask.astype(np.float32)
    count = mask_f.sum(axis=0)                                   # (726,)
    mean = (features * mask_f).sum(axis=0) / count               # (726,)
    var = ((features - mean) ** 2 * mask_f).sum(axis=0) / count  # (726,)
    std = np.sqrt(var)

    # Avoid division by zero for constant features
    zero_std = std == 0
    if zero_std.any():
        print(f"Warning: {zero_std.sum()} features have zero std, left as 0")
        std[zero_std] = 1.0

    print(f"Before — mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"Before — std range:  [{std.min():.4f}, {std.max():.4f}]")

    # Normalize, then re-zero masked positions
    features_norm = ((features - mean) / std * mask_f).astype(np.float32)

    print(f"After  — real values mean range: "
          f"[{(features_norm * mask_f).sum(axis=0).min() / count.min():.6f}, "
          f"{(features_norm * mask_f).sum(axis=0).max() / count.max():.6f}]")

    np.savez_compressed(OUTPUT_PATH, features=features_norm, mask=mask,
                        labels=labels, subjects=subjects, trials=trials,
                        norm_mean=mean, norm_std=std)
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
