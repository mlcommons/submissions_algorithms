# Regenerating the tech report

Run from the repository root:

```bash
uv sync --locked --extra dev --extra analysis
uv run --locked python -m artifacts.tech_report_v1.regenerate \
  --submission-directory logs/self_tuning \
  --output-root scoring_results_tech_report
```

Supply the complete submission pool and a new, empty output directory. The
runner executes the leaderboard, Muon, and framework notebooks in order,
including the 0–20% target-relaxation sweep. Outputs include figures, tables,
CSVs, executed notebooks, and a `manifest.json` with input/source hashes and
completion status. Each section records its actual settings in
`out/run_metadata.json`. Failed runs must not be used as results.

The default is non-strict self-tuning scoring over nine workloads. Pool and
workload changes affect scores. Configure filters and scoring settings in the
paired scripts; add submission labels in `report_utils.py`. The Muon, selected
sweep, and framework views require their configured submissions. Recheck the
framework view's batch-size assumptions when submission recipes change.

Sections share loading/scoring in `report_data.py`, table rendering in
`report_tables.py`, and labels, styles, paths, and PDF/PNG export in
`report_utils.py`. Notebook and command-line plots use the same renderers.

For interactive use, select the repository's `.venv` kernel. Run Section 4
before Muon; the framework notebook runs independently. After editing either
member of a notebook/script pair, synchronize it before running:

```bash
uv run jupytext --sync artifacts/tech_report_v1/section_4_leaderboards/score_submissions.py
uv run jupytext --sync artifacts/tech_report_v1/section_6_algorithms/muon_workloads.py
uv run jupytext --sync artifacts/tech_report_v1/section_7_frameworks/framework_comparison.py
```

Source notebooks have no saved outputs. Review executed notebooks and artifacts
in the fresh output directory; synchronization alone does not refresh results.
Section-local `out/` artifacts can be checked in; refresh them from a completed
run before committing generated results.
Paper export is separate, and paper tables remain hand-maintained. Appendix
curves use their [own workflow](appendix_training_curves/curve_plotting/README.md).
