"""
Preprocessing pipeline for the KernelCo/robot_control dataset.

Discovers all step_*.py modules in preprocess/, sorts them by name,
and applies each step's process() function to every sample.

Usage:
    python pipeline.py                  # run all steps
    python pipeline.py --steps 01       # run only step 01
    python pipeline.py --steps 01,02    # run steps 01 and 02

Each step module must define:
    process(sample: dict) -> dict

Input:  data/data/*.npz
Output: preprocessed/*.npz
"""

import argparse
import glob
import importlib
import os
import sys
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessed")


def discover_steps(filter_ids=None):
    """Find and import all preprocess/step_*.py modules in order."""
    step_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocess")
    pattern = os.path.join(step_dir, "step_*.py")
    step_files = sorted(glob.glob(pattern))

    steps = []
    for path in step_files:
        name = os.path.basename(path).replace(".py", "")
        # Extract step ID (e.g., "01" from "step_01_clip")
        step_id = name.split("_")[1]

        if filter_ids and step_id not in filter_ids:
            continue

        module = importlib.import_module(f"preprocess.{name}")
        if not hasattr(module, "process"):
            print(f"  SKIP {name}: no process() function")
            continue

        steps.append((name, module.process))

    return steps


def load_sample(path):
    """Load a .npz file into a mutable dict."""
    arr = np.load(path, allow_pickle=True)
    return {key: arr[key] for key in arr.files}


def save_sample(sample, path):
    """Save a sample dict as .npz."""
    np.savez_compressed(path, **sample)


def run(filter_ids=None):
    steps = discover_steps(filter_ids)
    if not steps:
        print("No steps found. Add step_*.py modules to preprocess/")
        sys.exit(1)

    print(f"Pipeline steps ({len(steps)}):")
    for name, _ in steps:
        print(f"  - {name}")

    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.npz")))
    print(f"\nProcessing {len(files)} files...")

    for i, fpath in enumerate(files):
        sample = load_sample(fpath)

        for step_name, process_fn in steps:
            sample = process_fn(sample)

        out_path = os.path.join(OUT_DIR, os.path.basename(fpath))
        save_sample(sample, out_path)

        if (i + 1) % 200 == 0 or (i + 1) == len(files):
            print(f"  {i + 1}/{len(files)}")

    print(f"\nDone. Output: {OUT_DIR}/")

    # Print a quick sanity check on the first file
    first = np.load(os.path.join(OUT_DIR, os.path.basename(files[0])), allow_pickle=True)
    print(f"\nSanity check ({os.path.basename(files[0])}):")
    for key in first.files:
        if key == "label":
            print(f"  {key}: {first[key].item()}")
        else:
            print(f"  {key}: {first[key].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument("--steps", type=str, default=None,
                        help="Comma-separated step IDs to run (e.g., '01,02')")
    args = parser.parse_args()

    filter_ids = args.steps.split(",") if args.steps else None
    run(filter_ids)
