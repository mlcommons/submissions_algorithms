#!/usr/bin/env python3
"""Compute pass for the step-time and step-vs-wall-clock analyses.

Loads the raw self-tuning run logs once (via the scoring package, so results
match official scoring) and emits tidy CSVs consumed by the plotting scripts:

  analysis/data/
    step_time.csv                 submission x workload, median seconds/step
    eval_share.csv                submission x workload, eval / (eval + train) wall-clock
    time_to_target_steps.csv      submission x workload, global_step at target
    time_to_target_wallclock.csv  submission x workload, wall-clock (s) at target
    leaderboard.csv               per submission: score_by_steps, score_by_wallclock
    raw_measurements.csv          tidy long-form cache of every eval row, for reuse

Wall-clock uses the official scoring clock (`score` column, identical to
`accumulated_submission_time`). Time-to-target is the per-study median, exactly
as `performance_profile.get_workloads_time_to_target` computes it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The vendored scoring code targets numpy>=2.0 (np.trapezoid). Shim for
# numpy<2 (np.trapz) so we can reuse it unmodified.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

# Make the `scoring` package importable when run as a plain script.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scoring import performance_profile, scoring_utils  # noqa: E402
from scoring.config import WorkloadConfig  # noqa: E402

DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "self_tuning"
DEFAULT_TARGETS = REPO_ROOT / "scoring" / "workload_targets.json"
DEFAULT_OUT_DIR = REPO_ROOT / "analysis" / "data"

STEP_COL = "global_step"
WALLCLOCK_COL = "score"  # official scoring clock == accumulated_submission_time
EVAL_COL = "accumulated_eval_time"  # wall-clock spent evaluating (clock paused)
SUBMIT_COL = "accumulated_submission_time"  # training-only clock


def load_results(log_dir: Path, config: WorkloadConfig) -> dict[str, pd.DataFrame]:
    """Load every submission under ``log_dir`` into {submission: experiment df}."""
    results = {}
    for sub_dir in sorted(p for p in log_dir.iterdir() if p.is_dir()):
        df = scoring_utils.get_experiment_df(str(sub_dir))
        if df.empty:
            print(f"  {sub_dir.name}: no data, skipping")
            continue
        results[sub_dir.name] = df
        print(f"  {sub_dir.name}: {len(df)} trial-rows, "
              f"{df['workload'].nunique()} workloads")
    return results


# --- Cache (tidy long-form CSV) ---------------------------------------------
# get_experiment_df returns {submission: df} where each cell is a per-trial time
# series (a ragged numpy array). Pickling that is brittle across library/version
# bumps and opaque to inspect. Instead we round-trip through a tidy long table --
# one row per (submission, study, trial, eval) with scalar columns -- which is
# just every trial's measurements.csv unioned. CSV keeps the cache human-readable
# and needs no engine beyond pandas (unlike parquet's pyarrow). ``results_to_tidy``
# / ``tidy_to_results`` are exact inverses for what the scoring package consumes.
ID_COLS = ["submission", "workload", "study", "trial", "experiment_dir", "eval_idx"]


def results_to_tidy(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flatten {submission: wide df} into one tidy long-form DataFrame."""
    frames = []
    for submission, df in results.items():
        for _, row in df.iterrows():
            trial_name, experiment_dir = row["trial"]
            # Metric columns hold arrays; scalar-NaN columns (metric absent for
            # this workload) are skipped, so they stay sparse after concat.
            metrics = {
                c: np.asarray(row[c])
                for c in df.columns
                if c not in ("workload", "trial", "study")
                and isinstance(row[c], (list, np.ndarray))
            }
            rec = pd.DataFrame(metrics)
            rec.insert(0, "submission", submission)
            rec.insert(1, "workload", row["workload"])
            rec.insert(2, "study", row["study"])
            rec.insert(3, "trial", trial_name)
            rec.insert(4, "experiment_dir", experiment_dir)
            rec.insert(5, "eval_idx", np.arange(len(rec)))
            frames.append(rec)
    return pd.concat(frames, ignore_index=True)


def tidy_to_results(tidy: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Rebuild {submission: wide df} matching get_experiment_df's structure."""
    metric_cols = [c for c in tidy.columns if c not in ID_COLS]
    results = {}
    for submission, sdf in tidy.groupby("submission", sort=False):
        rows = []
        keys = ["workload", "study", "trial", "experiment_dir"]
        for (workload, study, trial, exp_dir), g in sdf.groupby(keys, sort=False):
            g = g.sort_values("eval_idx")
            rec = {"workload": workload, "trial": (trial, exp_dir), "study": study}
            for c in metric_cols:
                col = g[c].to_numpy()
                # Restore scalar NaN for metrics that never applied (as concat
                # produced originally), else keep the per-trial array.
                rec[c] = np.nan if pd.isna(col).all() else col
            rows.append(rec)
        results[submission] = pd.DataFrame(rows)
    return results


def step_time_matrix(results, config) -> pd.DataFrame:
    """Median seconds/step per (submission, workload).

    Mirrors score_submissions.get_summary_df: for each trial take
    diff(wallclock)/diff(step), pool across trials, take the median.
    """
    rows = {}
    for submission, df in results.items():
        per_workload = {}
        for workload, group in df.groupby("workload"):
            base = config.base_workload_name(workload)
            ratios = []
            for _, trial in group.iterrows():
                wall = np.diff(trial[WALLCLOCK_COL], prepend=0)
                step = np.diff(trial[STEP_COL], prepend=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratios.append(wall / step)
            per_workload[base] = np.nanmedian(np.concatenate(ratios))
        rows[submission] = per_workload
    mat = pd.DataFrame(rows).T
    mat.index.name = "submission"
    return mat.reindex(sorted(mat.columns), axis=1)


def eval_share_matrix(results, config) -> pd.DataFrame:
    """Fraction of run wall-clock spent evaluating, per (submission, workload).

    ``eval / (eval + train)`` where train is the submission clock and eval is
    ``accumulated_eval_time``. This is the wall-clock the official scoring clock
    *excludes* -- invisible in the step-time matrix, but often a large share of
    what a run actually costs. Per trial we take the last logged (finite) row;
    we pool the ratio across trials and take the median.
    """
    rows = {}
    for submission, df in results.items():
        per_workload = {}
        for workload, group in df.groupby("workload"):
            base = config.base_workload_name(workload)
            shares = []
            for _, trial in group.iterrows():
                evalt = np.asarray(trial[EVAL_COL], dtype=float)
                train = np.asarray(trial[SUBMIT_COL], dtype=float)
                finite = np.isfinite(evalt) & np.isfinite(train)
                if finite.any():
                    e, s = evalt[finite][-1], train[finite][-1]
                    if e + s > 0:
                        shares.append(e / (e + s))
            if shares:
                per_workload[base] = float(np.median(shares))
        rows[submission] = per_workload
    mat = pd.DataFrame(rows).T
    mat.index.name = "submission"
    return mat.reindex(sorted(mat.columns), axis=1)


def time_to_target_matrix(results, config, time_col) -> pd.DataFrame:
    """submission x workload table of time-to-target in ``time_col`` units."""
    dfs = []
    for submission, df in results.items():
        dfs.append(
            performance_profile.get_workloads_time_to_target(
                df, submission, config, time_col=time_col,
                self_tuning_ruleset=True, verbosity=0,
            )
        )
    mat = pd.concat(dfs)
    return mat.reindex(sorted(mat.columns), axis=1)


def leaderboard(results, config, out_dir: Path) -> pd.DataFrame:
    """Leaderboard scores computed under steps vs wall-clock time-to-target."""
    scores = {}
    for label, time_col in [("score_by_steps", STEP_COL),
                            ("score_by_wallclock", WALLCLOCK_COL)]:
        perf_df = performance_profile.compute_performance_profiles(
            results, config, time_col=time_col,
            min_tau=1.0, max_tau=4.0, num_points=100, scale="linear",
            self_tuning_ruleset=True, strict=False, output_dir=str(out_dir),
        )
        s = performance_profile.compute_leaderboard_score(perf_df, normalize=True)
        scores[label] = s["score"]
    board = pd.DataFrame(scores)
    board["rank_by_steps"] = board["score_by_steps"].rank(ascending=False).astype(int)
    board["rank_by_wallclock"] = (
        board["score_by_wallclock"].rank(ascending=False).astype(int)
    )
    board["rank_shift"] = board["rank_by_wallclock"] - board["rank_by_steps"]
    board.index.name = "submission"
    return board.sort_values("score_by_wallclock", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse analysis/data/raw_measurements.csv instead of re-reading logs.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = WorkloadConfig.from_json(args.targets)
    cache = args.out_dir / "raw_measurements.csv"
    if args.use_cache and cache.is_file():
        print(f"Loading cached results from {cache}")
        results = tidy_to_results(pd.read_csv(cache))
    else:
        print(f"Loading logs from {args.log_dir} (benchmark {config.benchmark_version})")
        results = load_results(args.log_dir, config)
        results_to_tidy(results).to_csv(cache, index=False)

    print("\nComputing step-time matrix...")
    step_time_matrix(results, config).to_csv(args.out_dir / "step_time.csv")

    print("Computing eval-share matrix...")
    eval_share_matrix(results, config).to_csv(args.out_dir / "eval_share.csv")

    print("Computing time-to-target (steps)...")
    time_to_target_matrix(results, config, STEP_COL).to_csv(
        args.out_dir / "time_to_target_steps.csv"
    )
    print("Computing time-to-target (wall-clock)...")
    time_to_target_matrix(results, config, WALLCLOCK_COL).to_csv(
        args.out_dir / "time_to_target_wallclock.csv"
    )

    print("Computing leaderboard under steps vs wall-clock...")
    leaderboard(results, config, args.out_dir).to_csv(args.out_dir / "leaderboard.csv")

    print(f"\nDone. Wrote analysis CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
