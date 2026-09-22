"""Scoring configuration"""

import json
import os
import re
from dataclasses import dataclass, replace

# Strips the framework suffix from a logged workload name, e.g.
# 'imagenet_resnet_jax' -> 'imagenet_resnet'.
_FRAMEWORK_SUFFIX = re.compile(r'(.*)(_jax|_pytorch)$')

# The self-tuning ruleset gets this multiple of the external-tuning ruleset's
# per-trial runtime budget, since self-tuning submissions don't get external
# hyperparameter-tuning trials in exchange. See algorithmic-efficiency's
# submission_runner.py: "use 1.5x the runtime budget for the self-tuning
# ruleset".
SELF_TUNING_RUNTIME_FACTOR = 1.5

# The latest version's targets, vendored next to this module; used as the
# default when no --workload_targets is supplied.
DEFAULT_TARGETS_PATH = os.path.join(
  os.path.dirname(os.path.abspath(__file__)), 'workload_targets.json'
)


@dataclass(frozen=True)
class WorkloadTarget:
  """Scoring constants for a single workload."""

  target_metric_name: str
  target_metric_goal: str
  validation_target_value: float
  step_hint: int
  # External-tuning ruleset runtime budget, in seconds; None for targets
  # files that predate this field (e.g. workload_targets_v05.json).
  max_allowed_runtime_sec: int | None = None

  def __post_init__(self):
    if self.target_metric_goal not in ('minimize', 'maximize'):
      raise ValueError(
        'Target metric goal must be "minimize" or "maximize"; '
        f'got {self.target_metric_goal!r}.'
      )

  @property
  def is_minimized(self) -> bool:
    return self.target_metric_goal == 'minimize'

  def relaxed(self, fraction: float) -> 'WorkloadTarget':
    """Return this target relaxed by a relative fraction."""
    if not 0 <= fraction < 1:
      raise ValueError(f'Target relaxation must be in [0, 1); got {fraction}.')
    direction = 1 if self.is_minimized else -1
    return replace(
      self,
      validation_target_value=self.validation_target_value
      * (1 + direction * fraction),
    )


@dataclass(frozen=True)
class WorkloadConfig:
  """One benchmark version's scoring configuration.

  A `WorkloadConfig` describes one benchmark version's scoring inputs: which
  workloads count toward the score (`base_workloads`), which held-out variants
  were sampled (`held_out_workloads`), and each workload's target metric, target
  goal, target value, and step hint.
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

  def with_target_relaxations(
    self, relaxations: dict[str, float]
  ) -> 'WorkloadConfig':
    """Return a copy with selected convergence targets relaxed.

    The special selector ``all`` matches every configured workload. A base
    workload selector also matches its held-out variants. Later selectors
    override earlier selectors.
    """
    workload_relaxations = {}
    for selector, fraction in relaxations.items():
      if selector == 'all':
        matches = self.workloads
      elif selector in self.base_workloads:
        matches = (
          workload
          for workload in self.workloads
          if self.base_workload_name(workload) == selector
        )
      elif selector in self.workloads:
        matches = (selector,)
      else:
        raise ValueError(
          f'Unknown target-relaxation selector {selector!r}. Use "all" or '
          f'one of: {", ".join(sorted(self.workloads))}.'
        )
      for workload in matches:
        workload_relaxations[workload] = fraction

    workloads = {
      name: target.relaxed(workload_relaxations[name])
      if name in workload_relaxations
      else target
      for name, target in self.workloads.items()
    }
    return replace(self, workloads=workloads)

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

  def target_is_minimized(self, workload: str) -> bool:
    """Return whether the workload's target metric is minimized."""
    return self._target(workload).is_minimized

  def step_hint(self, workload: str) -> int:
    """Returns the step hint for a workload."""
    return self._target(workload).step_hint

  def max_runtime_sec(
    self, workload: str, self_tuning_ruleset: bool = False
  ) -> float:
    """Returns the runtime budget (seconds) for a workload.

    Returns the external-tuning ruleset's budget, or
    `SELF_TUNING_RUNTIME_FACTOR` times that when `self_tuning_ruleset` is
    True. Raises if this targets file predates `max_allowed_runtime_sec`.
    """
    target = self._target(workload)
    if target.max_allowed_runtime_sec is None:
      raise ValueError(
        f'No max_allowed_runtime_sec for workload {workload!r} in the '
        f'{self.benchmark_version} targets; this targets file predates that '
        'field.'
      )
    factor = SELF_TUNING_RUNTIME_FACTOR if self_tuning_ruleset else 1
    return target.max_allowed_runtime_sec * factor
