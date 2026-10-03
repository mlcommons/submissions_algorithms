#!/usr/bin/env python3
"""Pass 1 of the curve-plotting pipeline: raw logs -> tidy ``curve_data.csv``.

Walks ``logs/self_tuning/<submission>/study_*/<workload>*/trial_*/`` once,
reading each trial's ``eval_measurements.csv`` (the curve) and
``meta_data_0.json`` (the target metric/value), and emits a single tidy
long-format CSV with one row per validation measurement:

    submission, workload, workload_variant, study, trial,
    metric_name, target_value, higher_is_better,
    global_step, accumulated_submission_time, metric_value

``plot_curves.py`` consumes this file, so the expensive glob + CSV parse only
happens when the logs actually change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from plot_config import (
    DEFAULT_DATA_PATH,
    DEFAULT_LOG_DIR,
    is_higher_better,
    registered_submissions,
)

TIME_COL = "accumulated_submission_time"
STEP_COL = "global_step"

OUTPUT_COLUMNS = [
    "submission",
    "workload",
    "workload_variant",
    "study",
    "trial",
    "metric_name",
    "target_value",
    "higher_is_better",
    STEP_COL,
    TIME_COL,
    "metric_value",
]


def _trailing_int(name: str, default: int = 0) -> int:
    """``study_0`` -> 0, ``trial_12`` -> 12; ``default`` if not parseable."""
    try:
        return int(name.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return default


def _base_workload(variant: str) -> str:
    """Strip the framework suffix: ``criteo1tb_pytorch`` -> ``criteo1tb``."""
    return variant.replace("_pytorch", "").replace("_jax", "")


def parse_trial(trial_dir: Path, submission: str) -> pd.DataFrame | None:
    """Parse one trial directory into tidy rows, or None if unusable."""
    meta_path = trial_dir / "meta_data_0.json"
    csv_path = trial_dir / "eval_measurements.csv"
    if not meta_path.is_file() or not csv_path.is_file():
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    metric_name = meta.get("workload.target_metric_name")
    if not metric_name:
        return None
    target_value = meta.get("workload.validation_target_value")

    metric_col = f"validation/{metric_name}"
    df = pd.read_csv(csv_path)
    if metric_col not in df.columns:
        return None
    df = df.dropna(subset=[metric_col, TIME_COL, STEP_COL])
    if df.empty:
        return None

    variant = trial_dir.parent.name
    return pd.DataFrame(
        {
            "submission": submission,
            "workload": _base_workload(variant),
            "workload_variant": variant,
            "study": _trailing_int(trial_dir.parent.parent.name),
            "trial": _trailing_int(trial_dir.name),
            "metric_name": metric_name,
            "target_value": target_value,
            "higher_is_better": is_higher_better(metric_name),
            STEP_COL: df[STEP_COL].to_numpy(),
            TIME_COL: df[TIME_COL].to_numpy(),
            "metric_value": df[metric_col].to_numpy(),
        }
    )


def parse_logs(log_dir: Path, submissions: list[str] | None) -> pd.DataFrame:
    """Parse every trial under ``log_dir`` (optionally restricted to ``submissions``)."""
    if submissions is None:
        sub_dirs = sorted(p for p in log_dir.iterdir() if p.is_dir())
    else:
        sub_dirs = [log_dir / s for s in submissions]

    frames: list[pd.DataFrame] = []
    for sub_dir in sub_dirs:
        if not sub_dir.is_dir():
            print(f"  skip (missing): {sub_dir.name}")
            continue
        trial_dirs = sorted(sub_dir.glob("study_*/*/trial_*"))
        n_ok = 0
        for trial_dir in trial_dirs:
            try:
                rows = parse_trial(trial_dir, sub_dir.name)
            except Exception as e:  # noqa: BLE001 - keep going, report the file
                print(f"  error parsing {trial_dir}: {e}")
                continue
            if rows is not None:
                frames.append(rows)
                n_ok += 1
        print(f"  {sub_dir.name}: {n_ok} trial(s) parsed")

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(frames, ignore_index=True)[OUTPUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Root of the self-tuning run logs (default: repo logs/self_tuning).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to write the tidy curve_data.csv (default: logs/curve_plotting/curve_data.csv).",
    )
    parser.add_argument(
        "--registered-only",
        action="store_true",
        help="Only parse submissions referenced in the ALGO_CONFIGS registry.",
    )
    args = parser.parse_args()

    submissions = registered_submissions() if args.registered_only else None
    print(f"Parsing logs under: {args.log_dir}")
    data = parse_logs(args.log_dir, submissions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    print(
        f"\nWrote {len(data)} rows "
        f"({data['submission'].nunique()} submissions, "
        f"{data['workload'].nunique()} workloads) to {args.out}"
    )


if __name__ == "__main__":
    main()
