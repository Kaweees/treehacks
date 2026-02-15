"""
Step 00: Resample EEG and NIRS to a common 50 Hz grid.

EEG  (500 Hz → 50 Hz):  Average every 10 consecutive samples.
NIRS (4.76 Hz → 50 Hz): Forward-fill — each 50 Hz target gets the most
                         recent NIRS sample at or before that time.

Both modalities end up with the same number of time steps (duration × 50).
Must run BEFORE clipping (step_01).
"""

import numpy as np

EEG_RATE = 500
NIRS_RATE = 4.76
TARGET_RATE = 50


def process(sample):
    eeg = sample["feature_eeg"]      # (7499, 6)
    nirs = sample["feature_moments"]  # (72, 40, 3, 2, 3)

    n_eeg = eeg.shape[0]
    n_nirs = nirs.shape[0]

    # Use EEG duration as the reference (it's the shorter, cleaner boundary)
    duration = n_eeg / EEG_RATE  # 15.0s (approx)
    n_out = int(duration * TARGET_RATE)  # 750

    # ── EEG: 500 → 50 Hz by averaging blocks of 10 ───────────────────────
    block = int(EEG_RATE / TARGET_RATE)  # 10
    # Trim to exact multiple of block size
    n_trim = n_out * block
    eeg_trimmed = eeg[:n_trim]  # (7500 or 7490, 6) → use n_out*10
    eeg_50 = eeg_trimmed.reshape(n_out, block, -1).mean(axis=1)  # (750, 6)

    # ── NIRS: 4.76 → 50 Hz by forward fill ───────────────────────────────
    # NIRS sample timestamps
    nirs_times = np.arange(n_nirs) / NIRS_RATE  # (72,)
    # Target 50 Hz timestamps
    target_times = np.arange(n_out) / TARGET_RATE  # (750,)

    # For each target time, find the index of the most recent NIRS sample
    # searchsorted('right') gives the insertion point; subtract 1 for "at or before"
    indices = np.searchsorted(nirs_times, target_times, side="right") - 1
    indices = np.clip(indices, 0, n_nirs - 1)

    # Index into NIRS along time axis
    nirs_50 = nirs[indices]  # (750, 40, 3, 2, 3)

    sample["feature_eeg"] = eeg_50
    sample["feature_moments"] = nirs_50
    return sample
