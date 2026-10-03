# Curve plotting

Two-pass pipeline that turns raw self-tuning run logs into per-workload
training-curve comparison plots.

```
logs/self_tuning/**            parse_logs.py          plot_curves.py
(raw eval CSVs + meta)   ──▶   curve_data.csv   ──▶   <algo>/<workload>_curves.png
```

Splitting parse from plot means the expensive log walk happens once; iterating
on the plots (styling, adding an algorithm) only re-runs the cheap second pass.

## Figure layout

Each `<algo>/<workload>_curves.png` is a 2×2, one distinctly-colored curve per
submission (mean over trials, ±1σ band):

|            | vs. wall-clock time | vs. training steps |
|------------|---------------------|--------------------|
| **top**    | validation metric   | validation metric  |
| **bottom** | distance to target (log) | distance to target (log) |

The metric rows read like normal training curves. The **distance-to-target**
rows plot `|target − metric|` on a log axis, which blows up near the target so
runs that otherwise pile up on the target line stay distinguishable — each curve
simply ends when it reaches the target.

## Usage

```bash
# Pass 1: parse logs -> tidy curve_data.csv (re-run only when logs change)
python parse_logs.py

# Pass 2: render plots from curve_data.csv (cheap, re-runnable)
python plot_curves.py --algo all
```

Both scripts resolve the repo root from their own location, so they work from
any checkout and any working directory — no hardcoded paths.

Useful flags:

- `parse_logs.py --log-dir DIR --out FILE --registered-only`
- `plot_curves.py --algo {sfadamw,muon,ademamix,cautious_nadamw,lion,nadamw,all} --data FILE --out-dir DIR --jobs N --dpi N`

## Speed

Pass 2 renders every (algo, workload) figure in parallel (`--jobs`, default =
CPU count) since each figure is independent. Rendering cost scales with `--dpi`
(default 200); drop to `--dpi 150` for quicker on-screen iteration. The logs are
never re-parsed between plot runs.

## Files

| File | Role |
|------|------|
| `plot_config.py` | Shared registry (`ALGO_CONFIGS`), styling, path defaults, metric-direction helper. |
| `parse_logs.py`  | Pass 1 — logs ➜ tidy `curve_data.csv` (one row per validation measurement). |
| `plot_curves.py` | Pass 2 — `curve_data.csv` ➜ the 2×2 PNGs (distinct colors, ±1σ bands, distance-to-target). |

`curve_data.csv` is a generated artifact.
