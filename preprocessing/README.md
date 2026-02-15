# Preprocessing Pipeline

Transforms raw EEG + TD-NIRS recordings from the [KernelCo/robot_control](https://huggingface.co/datasets/KernelCo/robot_control) dataset into a single normalized tensor ready for model training.

## Data Layout

Place the raw `.npz` trial files at:

```
<repo_root>/data/data/*.npz
```

Each file contains:
- `feature_eeg`: EEG time series at 500 Hz, 6 channels
- `feature_moments`: TD-NIRS moments at 4.76 Hz, shape `(time, 40 modules, 3 SDS, 2 wavelengths, 3 moments)`
- `label`: dict with `label`, `subject_id`, `session_id`, `duration`

## Running the Pipeline

From the `preprocessing/` directory:

```bash
# Step 0–2: Resample, clip, and combine (per-trial)
python pipeline.py

# Step 3: Flatten all trials into one tensor
python -m preprocess.step_03_flatten

# Step 4: Keep only Left Fist, Right Fist, Relax
python -m preprocess.step_04_filter_labels

# Step 5: Create attention mask for missing NIRS values
python -m preprocess.step_05_mask

# Step 6: Z-score normalize each feature
python -m preprocess.step_06_normalize
```

Output is written to `preprocessing/preprocessed/`.

## What Each Step Does

| Step | Script | Input | Output | Description |
|------|--------|-------|--------|-------------|
| 00 | `step_00_resample.py` | Raw 500Hz EEG + 4.76Hz NIRS | Both at 50Hz | EEG: average every 10 samples. NIRS: forward-fill (repeat last known value) |
| 01 | `step_01_clip.py` | 50Hz, full trial | 50Hz, t=5–11s | Clips to 300 samples (6s stimulus window) |
| 02 | `step_02_combine.py` | Separate EEG (300,6) + NIRS (300,40,3,2,3) | Combined (300, 726) | Flattens NIRS and concatenates with EEG |
| 03 | `step_03_flatten.py` | 1395 trial files | Single tensor | Stacks all trials, each timestep becomes a row. Drops trials shorter than 300 timesteps |
| 04 | `step_04_filter_labels.py` | 5-class dataset | 3-class dataset | Keeps Left Fist (0), Right Fist (1), Relax (2) |
| 05 | `step_05_mask.py` | Features with NaN | Features + binary mask | Creates attention mask (1=real, 0=missing). NaN replaced with 0 |
| 06 | `step_06_normalize.py` | Unnormalized features + mask | Z-scored features + mask | Per-feature z-score using only real (masked-in) values |

## Final Output

`dataset_3class_norm.npz` contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `features` | (226800, 726) | Z-scored feature matrix (float32) |
| `mask` | (226800, 726) | Attention mask: 1=real, 0=missing (int8) |
| `labels` | (226800,) | 0=Left Fist, 1=Right Fist, 2=Relax (int32) |
| `subjects` | (226800,) | Subject ID per row (for train/test splitting) |
| `trials` | (226800,) | Trial index per row (300 consecutive rows = 1 trial) |
| `norm_mean` | (726,) | Per-feature mean (for applying to new data) |
| `norm_std` | (726,) | Per-feature std (for applying to new data) |

### Feature vector layout (726 columns)

| Columns | Count | Description |
|---------|-------|-------------|
| 0–5 | 6 | EEG RMS, channels 0–5 |
| 6–725 | 720 | NIRS: 40 modules × 3 SDS (short/medium/long) × 2 wavelengths (690nm/905nm) × 3 moments (intensity/mean ToF/variance) |

NIRS column index formula: `6 + module*18 + sds*6 + wavelength*3 + moment`

## Feature Filter

Use `filter_features.py` to drop NIRS dimensions that may be noise. This is useful when you want to train on a subset of features — for example, only medium SDS + infrared + mean ToF.

```bash
python filter_features.py
```

The script interactively asks which dimensions to include:

```
SDS range:
  Include short? [y/n]
  Include medium? [y/n]
  Include long? [y/n]

Wavelength:
  Include 690nm (red)? [y/n]
  Include 905nm (infrared)? [y/n]

Moment:
  Include intensity? [y/n]
  Include mean ToF? [y/n]
  Include variance? [y/n]
```

EEG channels are always kept. The script logs every kept column index and name, then saves `dataset_3class_norm_filtered.npz` with the same format as the unfiltered dataset (plus a `feature_names` array).
