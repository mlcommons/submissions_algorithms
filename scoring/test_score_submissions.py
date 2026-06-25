"""End-to-end smoke test for the scoring pipeline"""

import os
import subprocess
import sys
import tempfile

import pandas as pd
from absl.testing import absltest

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
