#!/usr/bin/env python3
"""Analysis 1: per-step cost of submissions across workloads.

Three figures, from the matrices in compute_metrics.py:

  step_time_heatmap.png           raw median seconds/step (rows grouped by
                                  framework, since step time is dominated by the
                                  framework + workload, not the optimizer).
  step_time_overhead_heatmap.png  the headline view: per-step cost normalized to
                                  the cheapest optimizer *within the same
                                  framework and workload*, so a cell isolates the
                                  optimizer's own overhead rather than JAX-vs-
                                  PyTorch or model-size differences. Faceted by
                                  framework; rows sorted by the geomean "Overall"
                                  column, which is the relative ranking of
                                  optimizers by per-step cost.
  eval_share_heatmap.png          fraction of run wall-clock spent evaluating.
                                  The scoring clock (and hence the step-time
                                  numbers) excludes this, so it is otherwise
                                  invisible -- yet it is ~half the wall-clock on
                                  Criteo and WMT.

Step time itself already excludes eval and logging: it is
diff(accumulated_submission_time)/diff(step), and that clock is paused during
evaluation. The eval-share figure surfaces what that exclusion hides.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm  # noqa: E402

from style import DIVERGING_CMAP, SEQUENTIAL_CMAP, apply_rc  # noqa: E402
from submissions import SUBMISSIONS, meta_for  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "analysis" / "data"
DEFAULT_OUT = REPO_ROOT / "analysis" / "figures"

WORKLOAD_LABELS = {
    "criteo1tb": "Criteo 1TB",
    "fastmri": "fastMRI",
    "finewebedu_lm": "FineWeb-Edu LM",
    "imagenet_resnet": "ImageNet ResNet",
    "imagenet_vit": "ImageNet ViT",
    "librispeech_conformer": "LS Conformer",
    "librispeech_deepspeech": "LS DeepSpeech",
    "ogbg": "OGBG",
    "wmt": "WMT",
}

FRAMEWORK_LABELS = {"jax": "JAX", "pytorch": "PyTorch"}


def order_rows(mat: pd.DataFrame) -> pd.DataFrame:
    """Group rows by framework (JAX then PyTorch), then registry order within."""
    def key(sub):
        fw = meta_for(sub).framework
        reg = list(SUBMISSIONS).index(sub) if sub in SUBMISSIONS else len(SUBMISSIONS)
        return (0 if fw == "jax" else 1, reg)

    return mat.loc[sorted(mat.index, key=key)]


def geomean(row: np.ndarray) -> float:
    """Geometric mean ignoring NaNs (the natural average of a ratio)."""
    vals = row[np.isfinite(row) & (row > 0)]
    return float(np.exp(np.mean(np.log(vals)))) if vals.size else np.nan


def _render(ax, mat, *, norm, cmap, fmt, na_text="—", pct=False, diverging=False):
    """Draw an annotated heatmap on ``ax`` (no colorbar) and return the image.

    ``diverging`` flips the text-contrast rule: diverging ramps are dark at both
    ends and light in the middle, so text goes white only at the extremes.
    """
    data = np.ma.masked_invalid(mat.to_numpy(dtype=float))
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    im.cmap.set_bad("#f0f0f0")  # NaN cells

    labels = [meta_for(s).label for s in mat.index]
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(np.arange(mat.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    vals = mat.to_numpy(dtype=float)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                txt, color = na_text, "#999999"
            else:
                txt = format(v * 100 if pct else v, fmt)
                frac = float(np.clip(norm(v), 0, 1))
                dark = (frac < 0.18 or frac > 0.82) if diverging else frac > 0.55
                color = "white" if dark else "#222222"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    return im


def raw_step_time(mat: pd.DataFrame, out_dir: Path) -> None:
    """Figure 1: raw median seconds/step, rows grouped by framework."""
    mat = order_rows(mat)
    n_jax = sum(meta_for(s).framework == "jax" for s in mat.index)
    fig, ax = plt.subplots(figsize=(10, 8))
    _render(
        ax, mat,
        norm=LogNorm(vmin=np.nanmin(mat.values), vmax=np.nanmax(mat.values)),
        cmap=SEQUENTIAL_CMAP, fmt=".2f",
    )
    cols = [WORKLOAD_LABELS.get(c, c) for c in mat.columns]
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.axhline(n_jax - 0.5, color="#222222", linewidth=2)  # JAX | PyTorch divider
    cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("median seconds / step (log scale)")
    ax.set_title(
        "Step time by submission and workload\n"
        "(raw wall-clock/step — dominated by framework & model size, "
        "not the optimizer)",
        fontweight="bold", pad=12, fontsize=13,
    )
    fig.savefig(out_dir / "step_time_heatmap.png")
    plt.close(fig)


def overhead(mat: pd.DataFrame, out_dir: Path) -> None:
    """Figure 2: per-step cost vs. the typical optimizer, faceted by framework.

    Within each framework, every column is divided by its *median*, so a cell is
    the optimizer's per-step cost relative to the typical optimizer on the same
    workload and framework -- the compute floor (fwd/bwd, framework, model size)
    cancels out, leaving the optimizer's own contribution. The median reference
    is robust to a single anomalously fast/slow run, which a min/max reference is
    not. Rows are sorted by the geomean "Overall" column: the overall relative
    ranking of optimizers by per-step cost.
    """
    facets = {}
    lo, hi = 1.0, 1.0
    for fw in ("jax", "pytorch"):
        subs = [s for s in mat.index if meta_for(s).framework == fw]
        sub = mat.loc[subs]
        rel = sub.div(sub.median(axis=0), axis=1)  # normalize per workload to median
        rel["Overall"] = rel.apply(lambda r: geomean(r.to_numpy()), axis=1)
        rel = rel.sort_values("Overall")  # cheapest optimizer first
        facets[fw] = rel
        lo, hi = min(lo, np.nanmin(rel.values)), max(hi, np.nanmax(rel.values))
        rel.to_csv(out_dir.parent / "data" / f"step_time_overhead_{fw}.csv")

    # Diverging around 1.0: blue = cheaper than typical, red = costlier.
    norm = TwoSlopeNorm(vmin=lo, vcenter=1.0, vmax=hi)
    heights = [facets["jax"].shape[0], facets["pytorch"].shape[0]]
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 13),
        gridspec_kw={"height_ratios": heights, "hspace": 0.42},
    )
    cols = [WORKLOAD_LABELS.get(c, c) for c in mat.columns] + ["Overall"]
    im = None
    for ax, fw in zip(axes, ("jax", "pytorch")):
        rel = facets[fw]
        im = _render(ax, rel, norm=norm, cmap=DIVERGING_CMAP, fmt=".1f",
                     diverging=True)
        ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
        ax.axvline(len(cols) - 1.5, color="#222222", linewidth=2)  # | Overall
        ax.set_title(FRAMEWORK_LABELS[fw], fontweight="bold", loc="left", pad=6)

    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("per-step cost / typical optimizer, same framework (×)")
    fig.suptitle(
        "Per-step optimizer overhead (framework-normalized)\n"
        "red = costlier per step than the typical optimizer, blue = cheaper",
        fontweight="bold", fontsize=14, y=0.98,
    )
    fig.savefig(out_dir / "step_time_overhead_heatmap.png")
    plt.close(fig)


def eval_share(mat: pd.DataFrame, out_dir: Path) -> None:
    """Figure 3: fraction of wall-clock spent evaluating (excluded from scoring)."""
    mat = order_rows(mat)
    n_jax = sum(meta_for(s).framework == "jax" for s in mat.index)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = _render(
        ax, mat, norm=Normalize(vmin=0, vmax=np.nanmax(mat.values)),
        cmap=SEQUENTIAL_CMAP, fmt=".0f", pct=True,
    )
    cols = [WORKLOAD_LABELS.get(c, c) for c in mat.columns]
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.axhline(n_jax - 0.5, color="#222222", linewidth=2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("eval / (eval + train) wall-clock")
    ax.set_title(
        "Eval-time share of wall-clock\n"
        "(paused by the scoring clock, so invisible in step time — "
        "yet ~half the run on Criteo & WMT)",
        fontweight="bold", pad=12, fontsize=13,
    )
    fig.savefig(out_dir / "eval_share_heatmap.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    apply_rc()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    step_time = pd.read_csv(args.data_dir / "step_time.csv", index_col=0)

    raw_step_time(step_time, args.out_dir)
    overhead(step_time, args.out_dir)

    eval_csv = args.data_dir / "eval_share.csv"
    if eval_csv.is_file():
        eval_share(pd.read_csv(eval_csv, index_col=0), args.out_dir)
    else:
        print(f"  (skipping eval-share figure; {eval_csv} not found — "
              f"rerun compute_metrics.py)")

    print(f"Wrote step-time figures to {args.out_dir}")


if __name__ == "__main__":
    main()
