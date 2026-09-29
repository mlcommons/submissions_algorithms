#!/usr/bin/env python3
"""Generate normalized step-execution-time table (tables/step_times.tex).

Forked from ``generate_step_size_table_sfadamw.py`` to cover all leaderboard
and Muon study submissions in ``logs/self_tuning/``, grouped into JAX and
PyTorch submissions and normalized to PyTorch Schedule-Free AdamW v2
(``schedule_free_adamw_v2``).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_LOG_DIR = REPO_ROOT / "logs" / "self_tuning"
OUT_DIR = Path(__file__).resolve().parent / "out" / "results"

WORKLOADS = [
    ("criteo1tb", r"\criteo"),
    ("fastmri", r"\fastmri"),
    ("finewebedu_lm", r"\finewebedu"),
    ("imagenet_resnet", r"\resnet"),
    ("imagenet_vit", r"\vit"),
    ("librispeech_conformer", r"\conformer"),
    ("librispeech_deepspeech", r"\deepspeech"),
    ("ogbg", r"\ogbg"),
    ("wmt", r"\wmt"),
]

JAX_SUBMISSIONS = [
    ("schedule_free_adamw_jax", "Schedule-Free v1"),
    ("schedule_free_adamw_jax_v2", "Schedule-Free v2"),
    ("muon", "Muon v1"),
    ("nadamw", "NAdamW v1"),
    ("nadamw_baselinev05", "NAdamW Baseline v0.5"),
    ("nadamw_resnet", "NAdamW ResNet"),
    ("cautious_nadamw", "Cautious NAdamW"),
    ("single_worker_diloco", "Single-Worker DiLoCo"),
]

PYTORCH_SUBMISSIONS = [
    ("schedule_free_adamw", "Schedule-Free v1"),
    ("schedule_free_adamw_v2", "Schedule-Free v2 (reference)"),
    ("muon_torch", "Muon v1"),
    ("muon_torch_jax_hps_lr_fix", "Muon v2 (JAX HPs)"),
    ("muon_torch_replicated_jax_hps", "Muon Replicated (JAX HPs)"),
    ("muon_torch_replicated_torch_hps", "Muon Replicated (Torch HPs)"),
    ("ademamix", "AdEMAMix (AdamW equiv.)"),
    ("ademamix_golden", "AdEMAMix"),
    ("lion", "Lion"),
]

REFERENCE_SUBMISSION = "schedule_free_adamw_v2"


def compute_trial_step_times_ms(sub_key: str, wl: str) -> list[float]:
    """Compute average step time (ms) per trial for (sub_key, workload)."""
    pattern = str(BASE_LOG_DIR / sub_key / "study_*" / f"{wl}*" / "trial_*")
    trial_dirs = sorted(glob.glob(pattern))
    trial_times: list[float] = []
    for td in trial_dirs:
        for fname in ("measurements.csv", "eval_measurements.csv"):
            fpath = os.path.join(td, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                df = pd.read_csv(fpath)
            except Exception:
                continue
            if (
                "accumulated_submission_time" in df.columns
                and "global_step" in df.columns
            ):
                df_valid = df.dropna(
                    subset=["accumulated_submission_time", "global_step"]
                )
                if len(df_valid) >= 2:
                    first_row = df_valid.iloc[0]
                    last_row = df_valid.iloc[-1]
                    delta_t = (
                        last_row["accumulated_submission_time"]
                        - first_row["accumulated_submission_time"]
                    )
                    delta_s = last_row["global_step"] - first_row["global_step"]
                    if delta_s > 0:
                        trial_times.append((delta_t / delta_s) * 1000.0)
                        break
    return trial_times


def format_ratio(times: list[float], ref_mean: float) -> str:
    """Format mean ± std normalized by ref_mean."""
    if not times or not np.isfinite(ref_mean) or ref_mean <= 0:
        return "---"
    mean_val = float(np.mean(times)) / ref_mean
    std_val = float(np.std(times)) / ref_mean
    if len(times) > 1 and std_val > 0.01:
        return f"${mean_val:.2f} \\pm {std_val:.2f}$"
    return f"${mean_val:.2f}$"


def main() -> None:
    all_subs = JAX_SUBMISSIONS + PYTORCH_SUBMISSIONS
    raw_times: dict[str, dict[str, list[float]]] = {
        sub_key: {} for sub_key, _ in all_subs
    }
    for sub_key, _ in all_subs:
        for wl, _ in WORKLOADS:
            raw_times[sub_key][wl] = compute_trial_step_times_ms(sub_key, wl)

    ref_means = {
        wl: float(np.mean(raw_times[REFERENCE_SUBMISSION][wl]))
        for wl, _ in WORKLOADS
    }

    wl_headers = " & ".join(macro for _, macro in WORKLOADS)
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Step execution times, normalized to PyTorch Schedule-Free AdamW v2.",
        r"  Each value is the ratio of a submission's average time per step to the",
        r"  reference's on that workload; submissions run at their own batch sizes.}",
        r"\label{tab:step_time_comparison}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        f"Optimizer & {wl_headers} \\\\",
        r"\midrule",
        r"\multicolumn{10}{@{}l}{\textit{JAX submissions}} \\",
        r"\midrule",
    ]

    for sub_key, label in JAX_SUBMISSIONS:
        cells = [
            format_ratio(raw_times[sub_key][wl], ref_means[wl])
            for wl, _ in WORKLOADS
        ]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{10}{@{}l}{\textit{PyTorch submissions}} \\",
            r"\midrule",
        ]
    )

    for sub_key, label in PYTORCH_SUBMISSIONS:
        cells = [
            format_ratio(raw_times[sub_key][wl], ref_means[wl])
            for wl, _ in WORKLOADS
        ]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )

    latex_output = "\n".join(lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "step_times.tex"
    out_path.write_text(latex_output)
    print(latex_output)
    print(f"Saved LaTeX table to {out_path}")


if __name__ == "__main__":
    main()
