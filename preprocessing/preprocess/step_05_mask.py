"""
Step 05: Create attention mask from NaN positions.

Produces a binary mask with the same shape as features:
  1 = real measurement (attend to it)
  0 = missing/NaN (ignore in attention)

NaN values in features are replaced with 0. The actual value doesn't
matter since the mask tells the transformer to ignore those positions.

Run directly:
    python -m preprocess.step_05_mask
"""

import os
import numpy as np

PREPROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preprocessed")
INPUT_PATH = os.path.join(PREPROCESSED_DIR, "dataset_3class.npz")
OUTPUT_PATH = os.path.join(PREPROCESSED_DIR, "dataset_3class_masked.npz")


def main():
    data = np.load(INPUT_PATH, allow_pickle=True)
    features = data["features"]  # (226800, 726)
    labels = data["labels"]
    subjects = data["subjects"]
    trials = data["trials"]

    print(f"Input: {features.shape}")

    # Create mask: 1 where real, 0 where NaN
    mask = (~np.isnan(features)).astype(np.int8)

    nan_count = np.isnan(features).sum()
    nan_cols = (mask.sum(axis=0) < features.shape[0]).sum()
    real_cols = (mask.sum(axis=0) == features.shape[0]).sum()
    print(f"NaN values: {nan_count} / {features.size} ({100*nan_count/features.size:.1f}%)")
    print(f"Columns: {real_cols} fully real, {nan_cols} partially missing")

    # Replace NaN with 0 (masked out, value irrelevant)
    features_filled = np.nan_to_num(features, nan=0.0).astype(np.float32)

    np.savez_compressed(OUTPUT_PATH, features=features_filled, mask=mask,
                        labels=labels, subjects=subjects, trials=trials)
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
