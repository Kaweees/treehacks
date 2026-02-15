"""
Step 01: Clip EEG and TD-NIRS data to t=5s–11s.

Removes the rest period (t=0–3s), early stimulus ramp-up (t=3–5s),
and post-stimulus silence (t>11s). Keeps a clean 6-second window
where all trials have active signal.

After step_00 resampling, both EEG and NIRS are at 50 Hz:
  50 Hz → samples 250:550  (6s × 50 = 300 samples)
"""

RATE = 50  # both modalities at 50 Hz after step_00

T_START = 5   # seconds
T_END = 11    # seconds

START = int(T_START * RATE)  # 250
END = int(T_END * RATE)      # 550


def process(sample):
    """Clip a single sample's EEG and NIRS to the t=5–11s window.

    Args:
        sample: dict with keys 'feature_eeg', 'feature_moments', 'label'

    Returns:
        sample with clipped arrays
    """
    sample["feature_eeg"] = sample["feature_eeg"][START:END]
    sample["feature_moments"] = sample["feature_moments"][START:END]
    return sample
