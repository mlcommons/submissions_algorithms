"""End-to-end smoke test for the scoring pipeline"""

import os
import subprocess
import sys
import tempfile

import pandas as pd
from absl.testing import absltest

from scoring.config import WorkloadConfig, WorkloadTarget
from scoring.score_submissions import (
  parse_target_relaxations,
  prepare_scoring_runs,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TARGETS = os.path.join(_REPO_ROOT, 'scoring', 'workload_targets_v05.json')
_EXTERNAL_LOGS = os.path.join(
  _REPO_ROOT, 'previous_leaderboards', 'algoperf_v05', 'logs', 'external_tuning'
)

# Published v0.5 external-tuning leaderboard scores (previous_leaderboards/
# algoperf_v05/README.md). Scoring is relative to the full group of submissions,
# so all of them must be scored together to reproduce these numbers.
_EXPECTED_EXTERNAL_SCORES = {
  'shampoo_submission': 0.7784,
  'schedule_free_adamw': 0.7077,
  'generalized_adam': 0.6383,
  'cyclic_lr': 0.6301,
  'nadamp': 0.5909,
  'baseline': 0.5707,
  'amos': 0.4918,
  'caspr_adaptive': 0.4722,
  'lawa_queue': 0.3699,
  'lawa_ema': 0.3384,
  'schedule_free_prodigy': 0.0,
}


class TargetRelaxationTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.config = WorkloadConfig(
      benchmark_version='test',
      base_workloads=('accuracy_workload', 'loss_workload'),
      held_out_workloads=('accuracy_workload_variant',),
      workloads={
        'accuracy_workload': WorkloadTarget('accuracy', 'maximize', 0.8, 100),
        'accuracy_workload_variant': WorkloadTarget(
          'accuracy', 'maximize', 0.75, 100
        ),
        'loss_workload': WorkloadTarget('loss', 'minimize', 0.2, 100),
      },
    )

  def test_parse_target_relaxations(self):
    self.assertEqual(
      parse_target_relaxations('all=0.05, accuracy_workload=0.1'),
      {'all': 0.05, 'accuracy_workload': 0.1},
    )
    self.assertEqual(parse_target_relaxations(''), {})

  def test_parse_target_relaxations_rejects_invalid_values(self):
    for spec in ('accuracy_workload', 'all=one', 'all=-0.1', 'all=1'):
      with self.subTest(spec=spec):
        with self.assertRaises(ValueError):
          parse_target_relaxations(spec)

  def test_relaxes_minimized_and_maximized_targets(self):
    relaxed = self.config.with_target_relaxations(
      {'all': 0.05, 'accuracy_workload': 0.1}
    )

    # The explicit base-workload selector overrides `all` for the base and its
    # held-out variants. Accuracy is maximized, so its target decreases.
    self.assertAlmostEqual(
      relaxed.workloads['accuracy_workload'].validation_target_value, 0.72
    )
    self.assertAlmostEqual(
      relaxed.workloads['accuracy_workload_variant'].validation_target_value,
      0.675,
    )
    # Loss is minimized, so its target increases.
    self.assertAlmostEqual(
      relaxed.workloads['loss_workload'].validation_target_value, 0.21
    )

  def test_relaxation_direction_comes_from_config(self):
    target = WorkloadTarget('loss', 'maximize', 0.2, 100)
    self.assertAlmostEqual(target.relaxed(0.1).validation_target_value, 0.18)

  def test_rejects_unknown_metric_goal(self):
    with self.assertRaisesRegex(ValueError, 'minimize.*maximize'):
      WorkloadTarget('accuracy', 'sideways', 0.8, 100)

  def test_prepare_scoring_runs_adds_relaxed_config_with_suffix(self):
    scoring_runs = prepare_scoring_runs(self.config, 'accuracy_workload=0.1')

    self.assertEqual(
      [(name, suffix) for name, suffix, _ in scoring_runs],
      [('official', ''), ('relaxed', '_relaxed')],
    )
    self.assertIs(scoring_runs[0][2], self.config)
    self.assertAlmostEqual(
      scoring_runs[1][2].workloads['accuracy_workload'].validation_target_value,
      0.72,
    )

  def test_prepare_scoring_runs_without_relaxation_scores_once(self):
    scoring_runs = prepare_scoring_runs(self.config, '')
    self.assertEqual(scoring_runs, [('official', '', self.config)])

  def test_exact_variant_selector_does_not_change_base(self):
    relaxed = self.config.with_target_relaxations(
      {'accuracy_workload_variant': 0.1}
    )
    self.assertEqual(
      relaxed.workloads['accuracy_workload'].validation_target_value, 0.8
    )
    self.assertAlmostEqual(
      relaxed.workloads['accuracy_workload_variant'].validation_target_value,
      0.675,
    )

  def test_rejects_unknown_workload(self):
    with self.assertRaisesRegex(ValueError, 'Unknown.*missing'):
      self.config.with_target_relaxations({'missing': 0.1})


class ScoreSubmissionsEndToEndTest(absltest.TestCase):
  def test_reproduces_v05_external_tuning_leaderboard(self):
    with tempfile.TemporaryDirectory() as output_dir:
      subprocess.run(
        [
          sys.executable,
          '-m',
          'scoring.score_submissions',
          '--workload_targets',
          _TARGETS,
          '--submission_directory',
          _EXTERNAL_LOGS,
          '--compute_performance_profiles',
          '--output_dir',
          output_dir,
        ],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
      )
      scores = pd.read_csv(os.path.join(output_dir, 'scores.csv'), index_col=0)[
        'score'
      ]
    self.assertCountEqual(
      scores.index.tolist(), _EXPECTED_EXTERNAL_SCORES.keys()
    )
    for submission, expected in _EXPECTED_EXTERNAL_SCORES.items():
      # Tolerance is set by the published values' 4-decimal rounding
      self.assertAlmostEqual(
        scores[submission],
        expected,
        delta=1e-4,
        msg=f'score for {submission} drifted from the published v0.5 value',
      )


if __name__ == '__main__':
  absltest.main()
