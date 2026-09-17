"""Shared log loading, summaries, and canonical scoring for report sections."""

from pathlib import Path
import pickle

import pandas as pd

from artifacts.tech_report_v1.report_utils import DISPLAY_TO_RAW, pretty


DEFAULT_MIN_TAU = 1.0
DEFAULT_MAX_TAU = 4.0
DEFAULT_PROFILE_POINTS = 100


def load_runs(
  directory, *, include=(), exclude=(), cache=None, save_cache=None
):
  """Load a named comparison pool from logs or an explicitly supplied cache."""
  from scoring import scoring_utils

  def names(value):
    return (
      {s.strip() for s in value.split(',') if s.strip()}
      if isinstance(value, str)
      else set(value)
    )

  include, exclude = names(include), names(exclude)
  directory = Path(directory)
  if cache:
    with Path(cache).open('rb') as handle:
      available = {
        DISPLAY_TO_RAW.get(name, name): frame
        for name, frame in pickle.load(handle).items()
      }
  else:
    available = {p.name: None for p in directory.iterdir() if p.is_dir()}
  missing = include - available.keys()
  if missing:
    raise ValueError(f'Requested submissions are missing: {sorted(missing)}')
  selected = sorted((include or available.keys()) - exclude)
  if not selected:
    raise ValueError('No submissions selected for scoring.')
  runs = {}
  for raw in selected:
    name = pretty(raw)
    if name in runs:
      raise ValueError(f'Duplicate submission display name: {name}')
    print(f'Loading {name} ({raw})', flush=True)
    frame = (
      available[raw]
      if cache
      else scoring_utils.get_experiment_df(str(directory / raw))
    )
    if frame.empty:
      raise ValueError(f'No evaluation measurements found for {raw}.')
    runs[name] = frame
  if save_cache:
    path = Path(save_cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
      pickle.dump(runs, handle)
  return runs


def write_summaries(runs, config, directory):
  """Refresh study summaries even when the parsed runs came from a cache."""
  from scoring.score_submissions import get_submission_summary

  directory = Path(directory)
  directory.mkdir(parents=True, exist_ok=True)
  summaries = {}
  for name, frame in runs.items():
    summary = get_submission_summary(frame, config)
    raw = DISPLAY_TO_RAW.get(name, name)
    summary.to_csv(directory / f'{raw}_summary.csv')
    summaries[name] = summary
  return summaries


def score_runs(
  runs,
  config,
  output_dir,
  *,
  time_col='score',
  artifact_suffix='',
  min_tau=DEFAULT_MIN_TAU,
  max_tau=DEFAULT_MAX_TAU,
  num_points=DEFAULT_PROFILE_POINTS,
  scale='linear',
  strict=False,
  self_tuning_ruleset=True,
):
  """Return profiles, normalized scores, and target times for the same pool."""
  from scoring import performance_profile

  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  profiles = performance_profile.compute_performance_profiles(
    runs,
    config,
    time_col=time_col,
    min_tau=min_tau,
    max_tau=max_tau,
    num_points=num_points,
    scale=scale,
    strict=strict,
    self_tuning_ruleset=self_tuning_ruleset,
    verbosity=0,
    output_dir=str(output_dir),
    artifact_suffix=artifact_suffix,
  )
  scores = performance_profile.compute_leaderboard_score(
    profiles, normalize=True
  )
  times = pd.read_csv(
    output_dir / f'time_to_targets{artifact_suffix}.csv', index_col=0
  )
  return profiles, scores, times.loc[scores.index]
