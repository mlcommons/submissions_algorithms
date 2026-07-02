# Submission analysis: step time & step-vs-wall-clock

Two analyses built on top of the self-tuning scoring logs, reusing the
`scoring/` package so numbers match official scoring.

```
logs/self_tuning/**   ─(compute_metrics.py)─▶   analysis/data/*.csv   ─(plot_*.py)─▶  analysis/figures/*.png
```

## Run

```bash
python3 analysis/compute_metrics.py            # parse logs -> tidy CSVs (--use-cache to skip re-parse)
python3 analysis/plot_step_time.py             # Analysis 1 figures
python3 analysis/plot_step_vs_wallclock.py     # Analysis 2 figures
```

`compute_metrics.py` needs `absl-py` (the scoring package dependency). It loads
every submission under `logs/self_tuning/`, then writes to `analysis/data/`:
`step_time.csv`, `eval_share.csv`, `time_to_target_steps.csv`,
`time_to_target_wallclock.csv`, `leaderboard.csv` (scores under both clocks), and
`raw_measurements.csv`.

`raw_measurements.csv` is a `--use-cache` speedup: a tidy long-form table (one row
per eval, just every trial's `measurements.csv` unioned with submission/study/
trial columns) that round-trips back into the exact structure the scoring package
consumes. Plain CSV over pickle so the cache stays human-readable and needs no
extra dependency (pandas reads it directly); it is regenerable from the logs, so
it need not be committed.

Wall-clock is the official scoring clock (`score`, identical to
`accumulated_submission_time`). Step time is `median(Δ score / Δ global_step)`
pooled over trials — the same quantity `score_submissions.py` reports as
`step_time (s)`. Note this clock is **paused during evaluation and logging**, so
step time already excludes eval; `eval_share.csv` reports how much wall-clock
that exclusion hides (`accumulated_eval_time / (eval + train)`).

---

## Analysis 1 — Step time (submissions × workloads)   *[Yifan, Andy]*

**Tables:** `analysis/data/step_time.csv` (median seconds/step per
submission×workload), `step_time_overhead_{jax,pytorch}.csv` (÷ the *median*
optimizer per workload, within framework), and `eval_share.csv`.
**Figures:** `step_time_heatmap.png` (raw), `step_time_overhead_heatmap.png`
(framework-normalized, the headline view), `eval_share_heatmap.png`.

A raw step-time cell mixes three things — workload compute (fwd/bwd), framework,
and the optimizer's own update — and the first two dominate. So the raw heatmap
mostly measures the workload and framework; rows are grouped JAX-then-PyTorch
with a divider because comparing across it is apples-to-oranges. The **overhead**
figure isolates the optimizer: within each framework it divides every column by
that column's *median*, cancelling the compute floor, and facets JAX/PyTorch so
the reference is comparable. The median reference (not min/max) is robust to a
single anomalous run — e.g. SF-AdamW JAX v1's 1.26 s/step on LS DeepSpeech, an
outlier among ~2.5–2.9 JAX peers, would otherwise become the column's floor and
inflate everyone else. Rows are sorted by the geomean **Overall** column: the
overall relative ranking of optimizers by per-step cost.

Findings:

- **Step time spans 0.09–2.86 s/step** across the grid — ~32× end to end — but
  most of that is workload + framework, not the optimizer.
- **Framework-normalized, most optimizers sit within ±10% of typical.** The real
  spread is on the short-step workloads where measurement is noisiest: fastMRI
  (0.3–3.0×) and Criteo. Muon variants and DiLoCo are the consistently costlier
  steps (Overall ~1.2×); schedule-free / Lion / cautious are typical-or-cheaper.
- **JAX vs PyTorch splits the speech workloads cleanly.** On LS
  Conformer/DeepSpeech the JAX submissions sit at ~2.5–2.9 s/step while the
  PyTorch ones are ~0.6–0.8 — a framework effect, so it lives across the divider
  in the raw figure, not in the overhead figure. *(Descriptive — runs may differ
  in hardware; not a controlled framework benchmark.)*
- **Eval time is excluded from step time and scoring, but it is large.**
  `eval_share.csv`/figure: eval is ~2–6% of wall-clock on ImageNet/Conformer but
  **~50% on Criteo and WMT**. It is also relatively larger under JAX (WMT ~50%
  JAX vs ~26% PyTorch), since JAX training is faster there so eval is a bigger
  slice. None of this touches the leaderboard — the scoring clock pauses for it.

## Analysis 2 — Effect of step vs wall-clock   *[Ahmed, Andy]*

Re-runs the exact scoring pipeline with time-to-target measured in `global_step`
vs wall-clock, then compares the resulting leaderboards.
**Table:** `analysis/data/leaderboard.csv`.
**Figures:** `step_vs_wallclock_scatter.png`, `leaderboard_rank_shift.png`.

Findings:

- **The two clocks agree in the aggregate but disagree in the middle.** Spearman
  between the step-based and wall-clock scores is **0.97**, yet **6 of 16**
  submissions change rank.
- **Muon (PyTorch) is the headline: rank 3 by steps → rank 6 by wall-clock
  (−3).** It reaches targets in competitive step counts, but its expensive steps
  (Analysis 1) erase the advantage on the clock that scoring actually uses.
- **Cheap-step methods rise under wall-clock:** SF-AdamW PyTorch v1 (5→3) and
  SF-AdamW JAX v2 (7→5) each gain two places; SF-AdamW PyTorch v2 takes #1.
- **Takeaway.** `wall_clock_to_target ≈ steps_to_target × step_time`, so
  Analysis 1 is the mechanism behind Analysis 2: a method's leaderboard standing
  depends on *how expensive its steps are*, not just *how few it needs*. In the
  scatter, points above the diagonal (Muon, DiLoCo) are step-efficient but
  wall-clock-slow; points below (schedule-free AdamW) are the opposite.

## Files

| File | Role |
|------|------|
| `submissions.py` | Submission → family/label/canonical metadata. |
| `style.py` | Shared palette (CVD-safe Okabe-Ito families, sequential/diverging ramps) + rcParams. |
| `compute_metrics.py` | Load logs via `scoring/`, emit tidy CSVs. |
| `plot_step_time.py` | Analysis 1 heatmaps. |
| `plot_step_vs_wallclock.py` | Analysis 2 scatter + rank-shift bump chart. |

`analysis/data/` holds generated CSVs + `raw_measurements.csv`;
`analysis/figures/` holds the PNGs.
