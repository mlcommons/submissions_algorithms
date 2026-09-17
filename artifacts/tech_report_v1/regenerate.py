"""Execute the current report notebooks in dependency order from fresh logs."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import jupytext
import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

from artifacts.tech_report_v1.report_utils import REPO_ROOT, REPORT_ROOT

NOTEBOOKS = (
  ('section_4_leaderboards', 'score_submissions'),
  ('section_6_algorithms', 'muon_workloads'),
  ('section_7_frameworks', 'framework_comparison'),
)


def fingerprints(paths, root):
  """Hash contents, including uncommitted changes and newly supplied logs."""
  return {
    str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(paths)
  }


def read_paired_notebook(script):
  """Fail on unsynchronized sources instead of trusting file timestamps."""
  notebook = jupytext.read(script)
  paired = jupytext.read(script.with_suffix('.ipynb'))

  def cells(nb):
    return [(c.cell_type, c.source) for c in nb.cells]

  if cells(notebook) != cells(paired):
    raise ValueError(
      f'Notebook pair differs: {script}. Reconcile edits, then synchronize '
      'with jupytext before regenerating.'
    )
  return notebook


def execute_notebook(notebook, destination, env, timeout):
  # Capture actual settings from the executed namespace, not duplicated defaults.
  notebook.cells.append(
    nbformat.v4.new_code_cell(r"""
import dataclasses, json
from pathlib import Path
_keys = (
    'SUBMISSION_DIRECTORY', 'INCLUDE_SUBMISSIONS', 'EXCLUDE_SUBMISSIONS',
    'STRICT', 'SELF_TUNING_RULESET', 'MIN_TAU', 'MAX_TAU', 'NUM_POINTS',
    'SCALE', 'TARGET_RELAXATION_FRACTION', 'SELECTED_SUBMISSIONS',
    'LOAD_RESULTS_FROM', 'SAVE_RESULTS_TO',
)
_metadata = {key: globals()[key] for key in _keys if key in globals()}
if _metadata.get('LOAD_RESULTS_FROM'):
    raise ValueError('Regeneration requires fresh logs; disable LOAD_RESULTS_FROM.')
if 'WORKLOAD_CONFIG' in globals():
    _metadata['workload_config'] = dataclasses.asdict(WORKLOAD_CONFIG)
if 'results' in globals():
    _metadata['submission_pool'] = sorted(results)
if 'SWEEP_PERCENTAGES' in globals():
    _metadata['sweep_percentages'] = list(SWEEP_PERCENTAGES)
Path(OUTPUT_DIR, 'run_metadata.json').write_text(
    json.dumps(_metadata, indent=2) + '\n'
)
""")
  )
  manager = KernelManager(kernel_name='python3')
  # Use the invoking uv environment even if another python3 kernel is installed.
  manager.kernel_spec.argv[0] = sys.executable
  client = NotebookClient(notebook, km=manager, timeout=timeout)
  try:
    client.execute(cwd=str(REPO_ROOT), env=env)
  finally:
    notebook.cells.pop()
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)


def regenerate(submission_directory, output_root, *, timeout=1800):
  submission_directory = submission_directory.resolve()
  output_root = output_root.resolve()
  if not submission_directory.is_dir():
    raise ValueError(f'Input directory does not exist: {submission_directory}')
  if output_root.exists() and any(output_root.iterdir()):
    raise ValueError(
      f'Output directory must be empty: {output_root}. '
      'Choose a new directory to avoid mixing results from different inputs.'
    )
  notebooks = [
    (section, name, read_paired_notebook(REPORT_ROOT / section / f'{name}.py'))
    for section, name in NOTEBOOKS
  ]

  def inputs():
    return fingerprints(
      submission_directory.rglob('eval_measurements.csv'), submission_directory
    )

  def sources():
    return fingerprints(
      [
        *REPORT_ROOT.rglob('*.py'),
        *REPO_ROOT.joinpath('scoring').glob('*.py'),
        *REPO_ROOT.joinpath('scoring').glob('workload_targets*.json'),
        REPO_ROOT / 'pyproject.toml',
        REPO_ROOT / 'uv.lock',
      ],
      REPO_ROOT,
    )

  manifest = {
    'status': 'running',
    'git_revision': subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True
    ).strip(),
    'python': platform.python_version(),
    'packages': {
      name: importlib.metadata.version(name)
      for name in ('numpy', 'pandas', 'matplotlib', 'jupytext', 'nbclient')
    },
    'submission_directory': str(submission_directory),
    'input_sha256': inputs(),
    'source_sha256': sources(),
  }
  if not manifest['input_sha256']:
    raise ValueError(
      'No eval_measurements.csv files found in the input directory.'
    )
  output_root.mkdir(parents=True, exist_ok=True)
  manifest_path = output_root / 'manifest.json'

  def save_manifest():
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

  save_manifest()
  env = {
    **os.environ,
    'ALGOPERF_REPORT_INPUT': str(submission_directory),
    'ALGOPERF_REPORT_OUTPUT': str(output_root),
    'MPLBACKEND': 'module://matplotlib_inline.backend_inline',
  }
  try:
    for section, name, notebook in notebooks:
      print(f'Executing {section}/{name} ...', flush=True)
      execute_notebook(
        notebook, output_root / section / f'{name}.ipynb', env, timeout
      )
    if (
      inputs() != manifest['input_sha256']
      or sources() != manifest['source_sha256']
    ):
      raise RuntimeError('Inputs or source changed during regeneration; rerun.')
    manifest['outputs_sha256'] = fingerprints(
      (p for p in output_root.rglob('*') if p.is_file() and p != manifest_path),
      output_root,
    )
    manifest['status'] = 'complete'
  except Exception as error:
    manifest.update(status='failed', error=str(error))
    raise
  finally:
    save_manifest()
  print(f'Completed. Results and executed notebooks: {output_root}', flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--submission-directory', type=Path, default=REPO_ROOT / 'logs/self_tuning'
  )
  parser.add_argument(
    '--output-root',
    type=Path,
    default=REPO_ROOT / 'scoring_results_tech_report',
  )
  parser.add_argument(
    '--timeout', type=int, default=1800, help='Seconds per cell.'
  )
  args = parser.parse_args()
  regenerate(args.submission_directory, args.output_root, timeout=args.timeout)


if __name__ == '__main__':
  main()
