"""
Step 04: Filter dataset to only Left Fist, Right Fist, and Relax.

Loads dataset.npz from step_03, keeps only rows with labels 0, 1, 4,
then re-encodes to 0, 1, 2:

  0 = Left Fist
  1 = Right Fist
  2 = Relax

Run directly:
    python -m preprocess.step_04_filter_labels
"""

import os
import numpy as np

PREPROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preprocessed")
INPUT_PATH = os.path.join(PREPROCESSED_DIR, "dataset.npz")
OUTPUT_PATH = os.path.join(PREPROCESSED_DIR, "dataset_3class.npz")

# Old label encoding from step_03
# 0 = Left Fist, 1 = Right Fist, 2 = Both Fists, 3 = Tongue Tapping, 4 = Relax
KEEP_LABELS = {0, 1, 4}
REMAP = {0: 0, 1: 1, 4: 2}  # Relax: 4 → 2
LABEL_NAMES = {0: "Left Fist", 1: "Right Fist", 2: "Relax"}


def main():
    data = np.load(INPUT_PATH, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    subjects = data["subjects"]
    trials = data["trials"]
    print(f"Input: {features.shape[0]} rows, {len(np.unique(trials))} trials")

    # Filter
    mask = np.isin(labels, list(KEEP_LABELS))
    features = features[mask]
    labels = labels[mask]
    subjects = subjects[mask]
    trials = trials[mask]

    # Remap labels
    labels_new = np.vectorize(REMAP.get)(labels).astype(np.int32)

    print(f"Output: {features.shape[0]} rows, {len(np.unique(trials))} trials")
    for idx, name in LABEL_NAMES.items():
        count = np.sum(labels_new == idx)
        print(f"  {name}: {count} timesteps ({count // 300} trials)")

    np.savez_compressed(OUTPUT_PATH, features=features, labels=labels_new,
                        subjects=subjects, trials=trials)
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
