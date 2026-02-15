"""Tests for RealtimePreprocessor."""

import os
import tempfile

import numpy as np
import pytest

from realtime_preprocessor import RealtimePreprocessor

N_EEG = 6
N_NIRS = 720
N_FEAT = N_EEG + N_NIRS


def _make_stats_file(mean, std):
    """Save norm stats to a temp .npz and return the path."""
    f = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    np.savez(f.name, norm_mean=mean, norm_std=std)
    return f.name


@pytest.fixture
def preprocessor(tmp_path):
    """Preprocessor with mean=0, std=1 (identity normalization)."""
    path = str(tmp_path / "stats.npz")
    np.savez(path,
             norm_mean=np.zeros(N_FEAT, dtype=np.float32),
             norm_std=np.ones(N_FEAT, dtype=np.float32))
    return RealtimePreprocessor(path)


@pytest.fixture
def norm_preprocessor(tmp_path):
    """Preprocessor with non-trivial normalization stats."""
    rng = np.random.default_rng(42)
    path = str(tmp_path / "stats.npz")
    np.savez(path,
             norm_mean=rng.standard_normal(N_FEAT).astype(np.float32),
             norm_std=(np.abs(rng.standard_normal(N_FEAT)).astype(np.float32) + 0.1))
    return RealtimePreprocessor(path)


def _make_nirs(value=1.0):
    return np.full((40, 3, 2, 3), value, dtype=np.float32)


# ------------------------------------------------------------------
# Shape
# ------------------------------------------------------------------


def test_output_shape(preprocessor):
    features, mask = preprocessor.get_features(0.0)
    assert features.shape == (1, N_FEAT)
    assert mask.shape == (1, N_FEAT)
    assert features.dtype == np.float32
    assert mask.dtype == np.int8


# ------------------------------------------------------------------
# Empty state
# ------------------------------------------------------------------


def test_no_data_returns_zeros(preprocessor):
    features, mask = preprocessor.get_features(0.0)
    np.testing.assert_array_equal(features, 0.0)
    np.testing.assert_array_equal(mask, 0)


# ------------------------------------------------------------------
# EEG only
# ------------------------------------------------------------------


def test_eeg_only(preprocessor):
    eeg = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    preprocessor.push_eeg(10.0, eeg)
    features, mask = preprocessor.get_features(15.0)

    # EEG columns populated
    np.testing.assert_array_almost_equal(features[0, :N_EEG], eeg)
    np.testing.assert_array_equal(mask[0, :N_EEG], 1)
    # NIRS columns masked
    np.testing.assert_array_equal(features[0, N_EEG:], 0.0)
    np.testing.assert_array_equal(mask[0, N_EEG:], 0)


# ------------------------------------------------------------------
# NIRS only
# ------------------------------------------------------------------


def test_nirs_only(preprocessor):
    nirs = _make_nirs(2.0)
    preprocessor.push_nirs(nirs)
    features, mask = preprocessor.get_features(0.0)

    # NIRS populated
    np.testing.assert_array_almost_equal(features[0, N_EEG:], 2.0)
    np.testing.assert_array_equal(mask[0, N_EEG:], 1)
    # EEG masked
    np.testing.assert_array_equal(features[0, :N_EEG], 0.0)
    np.testing.assert_array_equal(mask[0, :N_EEG], 0)


# ------------------------------------------------------------------
# Combined
# ------------------------------------------------------------------


def test_combined(preprocessor):
    eeg = np.ones(N_EEG, dtype=np.float32)
    nirs = _make_nirs(1.0)
    preprocessor.push_eeg(10.0, eeg)
    preprocessor.push_nirs(nirs)
    features, mask = preprocessor.get_features(15.0)

    np.testing.assert_array_equal(mask, 1)
    np.testing.assert_array_almost_equal(features[0, :N_EEG], 1.0)
    np.testing.assert_array_almost_equal(features[0, N_EEG:], 1.0)


# ------------------------------------------------------------------
# EEG windowing
# ------------------------------------------------------------------


def test_eeg_window_pruning(preprocessor):
    """Samples older than 20ms before current_time should be excluded."""
    preprocessor.push_eeg(0.0, np.array([100] * N_EEG, dtype=np.float32))   # old
    preprocessor.push_eeg(25.0, np.array([10] * N_EEG, dtype=np.float32))   # in window

    features, mask = preprocessor.get_features(30.0)  # window = [10, 30]

    # Only the second sample (value=10) should be included
    np.testing.assert_array_almost_equal(features[0, :N_EEG], 10.0)
    np.testing.assert_array_equal(mask[0, :N_EEG], 1)


def test_eeg_averaging(preprocessor):
    """Mean of multiple samples in window."""
    preprocessor.push_eeg(10.0, np.array([2] * N_EEG, dtype=np.float32))
    preprocessor.push_eeg(12.0, np.array([4] * N_EEG, dtype=np.float32))
    preprocessor.push_eeg(14.0, np.array([6] * N_EEG, dtype=np.float32))

    features, mask = preprocessor.get_features(20.0)  # window = [0, 20]

    expected_avg = 4.0  # mean of 2, 4, 6
    np.testing.assert_array_almost_equal(features[0, :N_EEG], expected_avg)


# ------------------------------------------------------------------
# NIRS NaN / inf handling
# ------------------------------------------------------------------


def test_nirs_nan_inf_handling(preprocessor):
    nirs = _make_nirs(5.0)
    nirs[0, 0, 0, 0] = np.nan
    nirs[1, 0, 0, 0] = np.inf
    nirs[2, 0, 0, 0] = -np.inf
    preprocessor.push_nirs(nirs)

    features, mask = preprocessor.get_features(0.0)

    # Positions corresponding to nan/inf should be masked and zeroed
    # The flat indices for (0,0,0,0), (1,0,0,0), (2,0,0,0) in (40,3,2,3):
    # stride: (3*2*3=18, 2*3=6, 3, 1)
    bad_indices = [N_EEG + 0 * 18, N_EEG + 1 * 18, N_EEG + 2 * 18]
    for idx in bad_indices:
        assert mask[0, idx] == 0
        assert features[0, idx] == 0.0

    # Other NIRS values should be valid
    good_nirs_mask = mask[0, N_EEG:]
    assert good_nirs_mask.sum() == N_NIRS - 3


# ------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------


def test_normalization_correctness(norm_preprocessor):
    rng = np.random.default_rng(99)
    eeg = rng.standard_normal(N_EEG).astype(np.float32)
    nirs = rng.standard_normal((40, 3, 2, 3)).astype(np.float32)

    norm_preprocessor.push_eeg(10.0, eeg)
    norm_preprocessor.push_nirs(nirs)
    features, mask = norm_preprocessor.get_features(15.0)

    raw = np.concatenate([eeg, nirs.reshape(N_NIRS)])
    expected = (raw - norm_preprocessor.norm_mean) / norm_preprocessor.norm_std
    np.testing.assert_array_almost_equal(features[0], expected, decimal=5)
    np.testing.assert_array_equal(mask, 1)


# ------------------------------------------------------------------
# NIRS overwrite
# ------------------------------------------------------------------


def test_nirs_overwrite(preprocessor):
    preprocessor.push_nirs(_make_nirs(1.0))
    preprocessor.push_nirs(_make_nirs(9.0))
    features, _ = preprocessor.get_features(0.0)

    np.testing.assert_array_almost_equal(features[0, N_EEG:], 9.0)
