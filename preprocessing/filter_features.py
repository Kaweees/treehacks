"""
Interactive feature filter for the normalized dataset.

Asks the user which NIRS dimensions to keep (SDS, wavelength, moment),
then drops unselected columns and saves a filtered copy.

Usage:
    python filter_features.py
"""

import numpy as np
import os

INPUT_PATH = "dataset_3class_norm.npz"
OUTPUT_PATH = "dataset_3class_norm_filtered.npz"

SDS_NAMES = {0: "short", 1: "medium", 2: "long"}
WL_NAMES = {0: "690nm (red)", 1: "905nm (infrared)"}
MOMENT_NAMES = {0: "intensity", 1: "mean ToF", 2: "variance"}

N_EEG = 6
N_MODULES = 40
N_SDS = 3
N_WL = 2
N_MOM = 3


def ask_group(group_name, options):
    """Ask user which options to keep. Returns set of selected indices."""
    print(f"\n{group_name}:")
    selected = set()
    for idx, name in options.items():
        answer = input(f"  Include {name}? [y/n] ").strip().lower()
        if answer in ("y", "yes", ""):
            selected.add(idx)
    if not selected:
        print(f"  Warning: nothing selected for {group_name}, keeping all")
        selected = set(options.keys())
    return selected


SDS_SHORT = {0: "short", 1: "medium", 2: "long"}
WL_SHORT = {0: "690nm", 1: "905nm"}
MOM_SHORT = {0: "intensity", 1: "mtof", 2: "variance"}


def compute_keep_columns(sds_keep, wl_keep, mom_keep):
    """Return sorted list of (column_index, name) tuples to keep."""
    keep = [(i, f"EEG_ch{i}") for i in range(N_EEG)]

    for m in range(N_MODULES):
        for s in range(N_SDS):
            for w in range(N_WL):
                for k in range(N_MOM):
                    if s in sds_keep and w in wl_keep and k in mom_keep:
                        col = N_EEG + m * (N_SDS * N_WL * N_MOM) + s * (N_WL * N_MOM) + w * N_MOM + k
                        name = f"M{m}_{SDS_SHORT[s]}_{WL_SHORT[w]}_{MOM_SHORT[k]}"
                        keep.append((col, name))
    return keep


def main():
    print("=== Feature Filter ===")
    print(f"Input: {INPUT_PATH}")

    sds_keep = ask_group("SDS range", SDS_NAMES)
    wl_keep = ask_group("Wavelength", WL_NAMES)
    mom_keep = ask_group("Moment", MOMENT_NAMES)

    keep = compute_keep_columns(sds_keep, wl_keep, mom_keep)
    keep_indices = [col for col, _ in keep]
    keep_names = [name for _, name in keep]
    n_nirs = len(keep) - N_EEG

    print(f"\nKeeping {N_EEG} EEG + {n_nirs} NIRS = {len(keep)} features (from 726)")
    print(f"\nKept columns:")
    for col, name in keep:
        print(f"  col {col:3d} -> {name}")

    # Load and filter
    data = np.load(INPUT_PATH, allow_pickle=True)
    features = data["features"][:, keep_indices]
    mask = data["mask"][:, keep_indices]
    norm_mean = data["norm_mean"][keep_indices]
    norm_std = data["norm_std"][keep_indices]

    np.savez_compressed(OUTPUT_PATH,
                        features=features,
                        mask=mask,
                        labels=data["labels"],
                        subjects=data["subjects"],
                        trials=data["trials"],
                        norm_mean=norm_mean,
                        norm_std=norm_std,
                        feature_names=np.array(keep_names))

    print(f"\nSaved {OUTPUT_PATH}")
    print(f"  features: {features.shape}")
    print(f"  mask:     {mask.shape}")


if __name__ == "__main__":
    main()
