"""
Step 02: Combine EEG and NIRS into a single feature matrix.

After resampling (step_00) and clipping (step_01), both are at 50 Hz
with 300 timesteps. This step flattens NIRS from (300, 40, 3, 2, 3)
to (300, 720) and concatenates with EEG (300, 6) to produce a single
array of shape (300, 726).

Feature layout per timestep:
  [0:6]    EEG channels
  [6:726]  NIRS: 40 modules × 3 SDS × 2 wavelengths × 3 moments
"""

import numpy as np


def process(sample):
    eeg = sample["feature_eeg"]       # (300, 6)
    nirs = sample["feature_moments"]  # (300, 40, 3, 2, 3)

    n_time = eeg.shape[0]
    nirs_flat = nirs.reshape(n_time, -1)  # (300, 720)

    sample["features"] = np.concatenate([eeg, nirs_flat], axis=1)  # (300, 726)

    del sample["feature_eeg"]
    del sample["feature_moments"]

    return sample
