"""
Step 03: Flatten all trials into a single tensor.

Loads all preprocessed files from steps 00–02, treats each timestep
as an independent sample, and saves one combined .npz:

  features: (300 * n_trials, 726)  float32
  labels:   (300 * n_trials,)      int32
  subjects: (300 * n_trials,)      str — for train/test splitting by subject
  trials:   (300 * n_trials,)      int32 — trial index for grouping

Label encoding:
  0 = Left Fist
  1 = Right Fist
  2 = Both Fists
  3 = Tongue Tapping
  4 = Relax

This step is NOT run by pipeline.py (it operates on pipeline output,
not on individual samples). Run it directly:
    python -m preprocess.step_03_flatten
"""

import os
import glob
import numpy as np

PREPROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preprocessed")
OUTPUT_PATH = os.path.join(PREPROCESSED_DIR, "dataset.npz")

LABEL_MAP = {
    "Left Fist": 0,
    "Right Fist": 1,
    "Both Fists": 2,
    "Tongue Tapping": 3,
    "Relax": 4,
}


def main():
    files = sorted(glob.glob(os.path.join(PREPROCESSED_DIR, "*.npz")))
    # Exclude dataset.npz itself if it exists from a previous run
    files = [f for f in files if os.path.basename(f) != "dataset.npz"]
    print(f"Loading {len(files)} trial files...")

    all_features = []
    all_labels = []
    all_subjects = []
    all_trials = []

    EXPECTED_TIMESTEPS = 300
    skipped = 0

    for trial_idx, f in enumerate(files):
        data = np.load(f, allow_pickle=True)
        features = data["features"]  # (300, 726)
        meta = data["label"].item()

        if features.shape[0] != EXPECTED_TIMESTEPS:
            skipped += 1
            continue

        label_int = LABEL_MAP[meta["label"]]

        all_features.append(features)
        all_labels.append(np.full(EXPECTED_TIMESTEPS, label_int, dtype=np.int32))
        all_subjects.append(np.full(EXPECTED_TIMESTEPS, meta["subject_id"], dtype=object))
        all_trials.append(np.full(EXPECTED_TIMESTEPS, trial_idx, dtype=np.int32))

    if skipped:
        print(f"Skipped {skipped} trials with != {EXPECTED_TIMESTEPS} timesteps")

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0)
    subjects = np.concatenate(all_subjects, axis=0)
    trials = np.concatenate(all_trials, axis=0)

    print(f"Features: {features.shape}")
    print(f"Labels:   {labels.shape}")
    print(f"Subjects: {len(np.unique(subjects))} unique")
    print(f"Trials:   {len(np.unique(trials))} unique")

    # Verify alignment
    assert features.shape[0] == labels.shape[0] == subjects.shape[0] == trials.shape[0]

    # Label distribution
    for name, idx in LABEL_MAP.items():
        count = np.sum(labels == idx)
        print(f"  {name}: {count} timesteps ({count // 300} trials)")

    np.savez_compressed(OUTPUT_PATH, features=features, labels=labels,
                        subjects=subjects, trials=trials)
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
