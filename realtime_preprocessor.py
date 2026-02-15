"""
Real-time preprocessing module for NIRSTransformer.

Accepts streaming EEG (6,) and NIRS (40,3,2,3) samples at different rates
and outputs the same (1, 726) normalized feature vector as the offline pipeline.

Feature layout: [EEG(6) | NIRS(720)] = 726
"""

import collections
import threading

import numpy as np


class RealtimePreprocessor:
    """Thread-safe real-time feature aggregator.

    Parameters
    ----------
    norm_mean : ndarray (726,)
        Per-feature mean from training set (step_06_normalize).
    norm_std : ndarray (726,)
        Per-feature std from training set. Zero-std features should
        already be set to 1.0 by the offline pipeline.
    """

    N_EEG = 6
    N_NIRS = 720
    N_FEATURES = N_EEG + N_NIRS
    EEG_WINDOW_MS = 20.0

    def __init__(self, norm_mean: np.ndarray, norm_std: np.ndarray):
        self.norm_mean = norm_mean.astype(np.float32).ravel()
        self.norm_std = norm_std.astype(np.float32).ravel()
        assert self.norm_mean.shape == (self.N_FEATURES,)
        assert self.norm_std.shape == (self.N_FEATURES,)

        self._eeg_buf: collections.deque = collections.deque()
        self._latest_nirs: np.ndarray | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def push_eeg(self, timestamp_ms: float, channels: np.ndarray) -> None:
        """Buffer a single EEG sample."""
        channels = np.asarray(channels, dtype=np.float32).ravel()
        assert channels.shape == (self.N_EEG,)
        with self._lock:
            self._eeg_buf.append((timestamp_ms, channels))

    def push_nirs(self, data: np.ndarray) -> None:
        """Store the latest NIRS frame (replaces previous)."""
        data = np.asarray(data, dtype=np.float32)
        assert data.shape == (40, 3, 2, 3)
        with self._lock:
            self._latest_nirs = data

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def get_features(self, current_time_ms: float):
        """Aggregate buffered data into a single (1, 726) feature vector.

        Returns
        -------
        features : ndarray (1, 726), float32
        mask : ndarray (1, 726), int8
        """
        with self._lock:
            eeg_avg, eeg_mask = self._aggregate_eeg(current_time_ms)
            nirs_flat, nirs_mask = self._aggregate_nirs()

        features = np.concatenate([eeg_avg, nirs_flat])       # (726,)
        mask = np.concatenate([eeg_mask, nirs_mask])           # (726,)

        # Replace any remaining NaN/inf with 0 and clear mask
        bad = ~np.isfinite(features)
        features[bad] = 0.0
        mask[bad] = 0

        # Normalize, then re-zero masked positions
        mask_f = mask.astype(np.float32)
        features = (features - self.norm_mean) / self.norm_std * mask_f

        return (
            features.reshape(1, self.N_FEATURES).astype(np.float32),
            mask.reshape(1, self.N_FEATURES).astype(np.int8),
        )

    # ------------------------------------------------------------------
    # Internal helpers (called under lock)
    # ------------------------------------------------------------------

    def _aggregate_eeg(self, current_time_ms: float):
        cutoff = current_time_ms - self.EEG_WINDOW_MS

        # Prune old samples
        while self._eeg_buf and self._eeg_buf[0][0] < cutoff:
            self._eeg_buf.popleft()

        if self._eeg_buf:
            samples = np.stack([s for _, s in self._eeg_buf])  # (N, 6)
            avg = samples.mean(axis=0)                         # (6,)
            mask = np.ones(self.N_EEG, dtype=np.int8)
        else:
            avg = np.zeros(self.N_EEG, dtype=np.float32)
            mask = np.zeros(self.N_EEG, dtype=np.int8)

        return avg, mask

    def _aggregate_nirs(self):
        if self._latest_nirs is not None:
            flat = self._latest_nirs.reshape(self.N_NIRS)      # (720,)
            mask = np.ones(self.N_NIRS, dtype=np.int8)
            bad = ~np.isfinite(flat)
            flat[bad] = 0.0
            mask[bad] = 0
        else:
            flat = np.zeros(self.N_NIRS, dtype=np.float32)
            mask = np.zeros(self.N_NIRS, dtype=np.int8)

        return flat, mask
