"""This script can
1. Summarize the raw submission times for each workload run in a set of studies and trials.
2. Produce the performance profiles and scores of a group of submissions. 
Note that for performance profiles and final scores are computed w.r.t. a group of submissions.
If you only have logs for one submission you may group it with some reference submission
to compare the performance.

Example usage:
python3 score_submissions.py \
  --submission_directory $HOME/algoperf-runs/submissions/rolling_leaderboard/self_tuning \
  --compute_performance_profiles \
  --output_dir scoring_results_self_tuning \
  --self_tuning_ruleset

"""

import operator
import os
import pickle

import numpy as np
import pandas as pd
from absl import app, flags, logging
from tabulate import tabulate

from scoring import performance_profile, scoring_utils
from scoring.config import DEFAULT_TARGETS_PATH, WorkloadConfig

flags.DEFINE_string(
  'submission_directory',
  None,
  'Path to submission directory containing experiment directories.',
)
flags.DEFINE_string(
  'workload_targets',
  DEFAULT_TARGETS_PATH,
  'Path to the version-specific workload_targets*.json scoring config '
  '(selects the benchmark version: base/held-out workloads and targets). '
  'Defaults to the latest (v0.6); use workload_targets_v05.json for v0.5.',
)
flags.DEFINE_string(
  'output_dir',
  'scoring_results',
  'Path to save performance profile artifacts, submission_summaries and results files.',
)
flags.DEFINE_boolean(
  'compute_performance_profiles',
  False,
  'Whether or not to compute the performance profiles.',
)
flags.DEFINE_boolean(
  'strict',
  False,
  'Whether to enforce scoring criteria on variant performance and on'
  '5-trial median performance. Note that during official scoring this '
  'flag will be set to True.',
)
flags.DEFINE_boolean(
  'self_tuning_ruleset',
  False,
  'Whether to score on self-tuning ruleset or externally tuned ruleset',
)
flags.DEFINE_string(
  'save_results_to_filename',
  None,
  'Filename to save the processed results that are fed into the performance profile functions.',
)
flags.DEFINE_string(
  'load_results_from_filename',
  None,
  'Filename to load processed results from that are fed into performance profile functions',
)
flags.DEFINE_string(
  'exclude_submissions',
  '',
  'Optional comma seperated list of names of submissions to exclude from scoring.',
)
flags.DEFINE_string(
  'include_submissions',
  '',
  'Optional comma seperated list of names of submissions to include from scoring.',
)
flags.DEFINE_string(
  'target_relaxations',
  '',
  'Optional comma-separated target relaxations in SELECTOR=FRACTION form. '
  'For example, "all=0.05,imagenet_resnet=0.10" relaxes every target by '
  '5%, then the ImageNet ResNet workload family by 10%. A base workload '
  'selector includes its held-out variants. Fractions must be in [0, 1).',
)
FLAGS = flags.FLAGS


def parse_target_relaxations(spec):
  """Parse a target-relaxation flag into an ordered selector mapping."""
  relaxations = {}
  if not spec.strip():
    return relaxations

  for assignment in spec.split(','):
    assignment = assignment.strip()
    if assignment.count('=') != 1:
      raise ValueError(
        'Target relaxations must use SELECTOR=FRACTION syntax; '
        f'got {assignment!r}.'
      )
    selector, fraction_text = (part.strip() for part in assignment.split('='))
    if not selector:
      raise ValueError(f'Missing workload selector in {assignment!r}.')
    if selector in relaxations:
      raise ValueError(f'Duplicate target-relaxation selector {selector!r}.')
    try:
      fraction = float(fraction_text)
    except ValueError:
      raise ValueError(
        f'Target relaxation for {selector!r} must be a number; '
        f'got {fraction_text!r}.'
      ) from None
    if not 0 <= fraction < 1:
      raise ValueError(
        f'Target relaxation for {selector!r} must be in [0, 1); got {fraction}.'
      )
    relaxations[selector] = fraction
  return relaxations


def prepare_scoring_runs(official_config, relaxation_spec):
  """Build the official and optional relaxed scoring configurations."""
  relaxations = parse_target_relaxations(relaxation_spec)
  scoring_runs = [('official', '', official_config)]
  if not relaxations:
    return scoring_runs

  relaxed_config = official_config.with_target_relaxations(relaxations)
  scoring_runs.append(('relaxed', '_relaxed', relaxed_config))
  return scoring_runs


def get_summary_df(workload, workload_df, config):
  print(f' WORKLOAD: {workload}')
  validation_metric, validation_target = config.metric_and_target(workload)

  is_minimized = config.target_is_minimized(workload)
  target_op = operator.le if is_minimized else operator.ge
  best_op = min if is_minimized else max
  idx_op = np.argmin if is_minimized else np.argmax

  summary_df = pd.DataFrame()
  summary_df['workload'] = workload_df['workload']
  summary_df['trial'] = workload_df['trial'].apply(lambda x: x[0])
  summary_df['val target metric name'] = validation_metric
  summary_df['val target metric value'] = validation_target

  summary_df['val target reached'] = (
    workload_df[validation_metric]
    .apply(lambda x: target_op(x, validation_target))
    .apply(np.any)
  )
  summary_df['best metric value on val'] = workload_df[validation_metric].apply(
    lambda x: best_op(x)
  )
  workload_df['index best eval on val'] = workload_df[validation_metric].apply(
    lambda x: idx_op(x)
  )
  summary_df['time to best eval on val (s)'] = workload_df.apply(
    lambda x: x['accumulated_submission_time'][x['index best eval on val']],
    axis=1,
  )
  workload_df['val target reached'] = (
    workload_df[validation_metric]
    .apply(lambda x: target_op(x, validation_target))
    .apply(np.any)
  )
  workload_df['index to target on val'] = workload_df.apply(
    lambda x: np.argmax(target_op(x[validation_metric], validation_target))
    if x['val target reached']
    else np.nan,
    axis=1,
  )
  summary_df['time to target on val (s)'] = workload_df.apply(
    lambda x: x['accumulated_submission_time'][int(x['index to target on val'])]
    if x['val target reached']
    else np.inf,
    axis=1,
  )

  # compute the step times
  def delta(series):
    return series.apply(lambda x: np.diff(x, prepend=0))

  accumulated_time_intervals = delta(workload_df['accumulated_submission_time'])
  step_intervals = delta(workload_df['global_step'])
  if len(accumulated_time_intervals) < 2:
    print(
      f'WARNING: The number of evals may be too low to calculate reliable step time for {workload}'
    )

  # Flatten all intervals from all trials and take the global median
  with np.errstate(divide='ignore', invalid='ignore'):
    all_ratios = np.concatenate(
      (accumulated_time_intervals / step_intervals).values
    )
  summary_df['step_time (s)'] = np.nanmedian(all_ratios)

  summary_df['step_hint'] = config.step_hint(workload)

  return summary_df


def get_submission_summary(df, config):
  """Summarizes the submission results into metric and time tables
  organized by workload.
  """

  dfs = []
  print(df)
  for workload, group in df.groupby('workload'):
    summary_df = get_summary_df(workload, group, config)
    dfs.append(summary_df)

  df = pd.concat(dfs)
  logging.info('\n' + tabulate(df, headers='keys', tablefmt='psql'))
  return df


def compute_leaderboard_score(df, normalize=True):
  """Compute leaderboard score by taking integral of performance profile.

  Args:
    df: pd.DataFrame returned from `compute_performance_profiles`.
    normalize: divide by the range of the performance profile's tau.

  Returns:
    pd.DataFrame with one column of scores indexed by submission.
  """
  scores = np.trapezoid(df, x=df.columns)
  if normalize:
    scores /= df.columns.max() - df.columns.min()
  return pd.DataFrame(scores, columns=['score'], index=df.index)


def score_results(
  results,
  config,
  run_name,
  artifact_suffix,
  output_dir,
  self_tuning_ruleset=False,
  strict=False,
):
  """Score parsed results once with one workload configuration."""
  logging.info('Computing %s scores.', run_name)
  profile_df = performance_profile.compute_performance_profiles(
    results,
    config,
    time_col='score',
    min_tau=1.0,
    max_tau=4.0,
    reference_submission_tag=None,
    num_points=100,
    scale='linear',
    verbosity=0,
    self_tuning_ruleset=self_tuning_ruleset,
    strict=strict,
    output_dir=output_dir,
    artifact_suffix=artifact_suffix,
  )
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)
  performance_profile.plot_performance_profiles(
    profile_df,
    'score',
    save_dir=output_dir,
    artifact_suffix=artifact_suffix,
  )
  logging.info(
    '%s performance profile:\n%s',
    run_name.capitalize(),
    tabulate(profile_df.T, headers='keys', tablefmt='psql'),
  )

  scores = compute_leaderboard_score(profile_df)
  scores.to_csv(os.path.join(output_dir, f'scores{artifact_suffix}.csv'))
  logging.info(
    '%s scores:\n%s',
    run_name.capitalize(),
    tabulate(scores, headers='keys', tablefmt='psql'),
  )


def main(_):
  results = {}
  os.makedirs(FLAGS.output_dir, exist_ok=True)
  official_config = WorkloadConfig.from_json(FLAGS.workload_targets)
  try:
    scoring_runs = prepare_scoring_runs(
      official_config, FLAGS.target_relaxations
    )
  except ValueError as e:
    raise app.UsageError(str(e)) from e
  logging.info(
    f'Scoring submissions in {FLAGS.submission_directory} '
    f'with benchmark version {official_config.benchmark_version} targets '
    f'({FLAGS.workload_targets})'
  )

  # Optionally read results to filename
  if FLAGS.load_results_from_filename:
    with open(
      os.path.join(FLAGS.output_dir, FLAGS.load_results_from_filename), 'rb'
    ) as f:
      results = pickle.load(f)
  else:
    all_submission_dirs = list(os.listdir(FLAGS.submission_directory))
    if not FLAGS.include_submissions:
      include_submissions = all_submission_dirs
    else:
      include_submissions = FLAGS.include_submissions.split(',')

    for submission in all_submission_dirs:
      print(submission)
      if submission not in FLAGS.exclude_submissions.split(',') and (
        submission in include_submissions
      ):
        experiment_path = os.path.join(FLAGS.submission_directory, submission)
        df = scoring_utils.get_experiment_df(experiment_path)
        results[submission] = df
        for _, artifact_suffix, config in scoring_runs:
          summary_df = get_submission_summary(df, config)
          summary_path = os.path.join(
            FLAGS.output_dir, f'{submission}_summary{artifact_suffix}.csv'
          )
          with open(summary_path, 'w') as fout:
            summary_df.to_csv(fout)

    # Optionally save results to filename
    if FLAGS.save_results_to_filename:
      with open(
        os.path.join(FLAGS.output_dir, FLAGS.save_results_to_filename), 'wb'
      ) as f:
        pickle.dump(results, f)

  if not FLAGS.strict:
    logging.warning(
      'You are running with strict=False. This will relax '
      'scoring criteria on the held-out workloads, number of trials and number '
      'of studies. Your score may not be an accurate representation '
      'under competition scoring rules. To enforce the criteria set strict=True.'
    )
  if FLAGS.compute_performance_profiles:
    for run_name, artifact_suffix, config in scoring_runs:
      score_results(
        results,
        config,
        run_name,
        artifact_suffix,
        FLAGS.output_dir,
        self_tuning_ruleset=FLAGS.self_tuning_ruleset,
        strict=FLAGS.strict,
      )


if __name__ == '__main__':
  # flags.mark_flag_as_required('submission_directory')
  app.run(main)
