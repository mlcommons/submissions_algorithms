"""Scoring configuration"""

import json
import os
import re
from dataclasses import dataclass

# Strips the framework suffix from a logged workload name, e.g.
# 'imagenet_resnet_jax' -> 'imagenet_resnet'.
_FRAMEWORK_SUFFIX = re.compile(r'(.*)(_jax|_pytorch)$')

# The latest version's targets, vendored next to this module; used as the
# default when no --workload_targets is supplied.
DEFAULT_TARGETS_PATH = os.path.join(
  os.path.dirname(os.path.abspath(__file__)), 'workload_targets.json'
)


@dataclass(frozen=True)
class WorkloadTarget:
  """Scoring constants for a single workload."""

  target_metric_name: str
  validation_target_value: float
  step_hint: int


@dataclass(frozen=True)
class WorkloadConfig:
  """One benchmark version's scoring configuration.

  A `WorkloadConfig` describes one benchmark version's scoring inputs: which
  workloads count toward the score (`base_workloads`), which held-out variants
  were sampled (`held_out_workloads`), and each workload's target metric, target
  value, and step hint.
  """

  benchmark_version: str
  base_workloads: tuple[str, ...]
  held_out_workloads: tuple[str, ...]
  workloads: dict[str, WorkloadTarget]  # workload name (no framework suffix)

  @classmethod
  def from_json(cls, path: str | os.PathLike) -> 'WorkloadConfig':
    """Loads and validates a `workload_targets*.json` file."""
    with open(path, 'r') as f:
      raw = json.load(f)
    try:
      workloads = {
        name: WorkloadTarget(**spec) for name, spec in raw['workloads'].items()
      }
      config = cls(
        benchmark_version=raw['benchmark_version'],
        base_workloads=tuple(raw['base_workloads']),
        held_out_workloads=tuple(raw['held_out_workloads']),
        workloads=workloads,
      )
    except (KeyError, TypeError) as e:
      raise ValueError(f'Malformed workload targets file {path!r}: {e}') from e

    missing = [
      w
      for w in config.base_workloads + config.held_out_workloads
      if w not in config.workloads
    ]
    if missing:
      raise ValueError(
        f'{path!r}: base/held-out workloads missing from `workloads`: {missing}'
      )
    return config

  @property
  def num_base_workloads(self) -> int:
    return len(self.base_workloads)

  @property
  def num_variant_workloads(self) -> int:
    return len(self.held_out_workloads)

  def base_workload_name(self, workload_name: str) -> str:
    """Maps a (possibly variant) workload name to its base workload name."""
    for base_workload_name in self.base_workloads:
      if base_workload_name in workload_name:
        return base_workload_name
    return workload_name

  def _target(self, workload: str) -> WorkloadTarget:
    match = _FRAMEWORK_SUFFIX.match(workload)
    name = match.group(1) if match else workload
    try:
      return self.workloads[name]
    except KeyError:
      raise KeyError(
        f'No scoring target for workload {name!r} (from {workload!r}) in the '
        f'{self.benchmark_version} targets. Known workloads: '
        f'{sorted(self.workloads)}.'
      ) from None

  def metric_and_target(self, workload: str) -> tuple[str, float]:
    """Returns the (validation metric column, target value) for a workload."""
    target = self._target(workload)
    return (
      f'validation/{target.target_metric_name}',
      target.validation_target_value,
    )

  def step_hint(self, workload: str) -> int:
    """Returns the step hint for a workload."""
    return self._target(workload).step_hint
