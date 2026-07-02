"""Submission metadata: algorithm family, display label, and canonical flag.

Used by the analysis scripts to group/label the 17 self-tuning submissions and
to flag dev/duplicate variants without dropping them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubmissionMeta:
    family: str  # algorithm family for grouping
    label: str  # short display label
    canonical: bool  # True for the representative submission of its family
    framework: str  # "jax" or "pytorch" (the runtime the submission ran under)


# Keyed by submission directory name (as under logs/self_tuning/ and results/).
SUBMISSIONS: dict[str, SubmissionMeta] = {
    "nadamw": SubmissionMeta("nadamw", "NAdamW (JAX)", True, "jax"),
    "nadamw_baselinev05": SubmissionMeta("nadamw", "NAdamW baseline v0.5", False, "jax"),
    "nadamw_resnet": SubmissionMeta("nadamw", "NAdamW resnet", False, "jax"),
    "schedule_free_adamw": SubmissionMeta("schedule_free_adamw", "SF-AdamW PyTorch v1", True, "pytorch"),
    "schedule_free_adamw_v2": SubmissionMeta("schedule_free_adamw", "SF-AdamW PyTorch v2", False, "pytorch"),
    "schedule_free_adamw_jax": SubmissionMeta("schedule_free_adamw", "SF-AdamW JAX v1", False, "jax"),
    "schedule_free_adamw_jax_v2": SubmissionMeta("schedule_free_adamw", "SF-AdamW JAX v2", False, "jax"),
    "muon": SubmissionMeta("muon", "Muon (JAX)", True, "jax"),
    "muon_torch": SubmissionMeta("muon", "Muon PyTorch", False, "pytorch"),
    "muon_torch_jax_hps": SubmissionMeta("muon", "Muon PyTorch (JAX HPs)", False, "pytorch"),
    "muon_torch_jax_hps_achandr": SubmissionMeta("muon", "Muon PyTorch (achandr)", False, "pytorch"),
    "muon_torch_jax_hps_lr_fix": SubmissionMeta("muon", "Muon PyTorch (lr fix)", False, "pytorch"),
    "ademamix": SubmissionMeta("ademamix", "AdEMAMix (PyTorch)", True, "pytorch"),
    "cautious_nadamw": SubmissionMeta("cautious_nadamw", "Cautious NAdamW (JAX)", True, "jax"),
    "lion": SubmissionMeta("lion", "Lion (PyTorch)", True, "pytorch"),
    "single_worker_diloco": SubmissionMeta("diloco", "Single-worker DiLoCo", True, "jax"),
}


def meta_for(submission: str) -> SubmissionMeta:
    """Return metadata for a submission, falling back to a sensible default."""
    return SUBMISSIONS.get(
        submission, SubmissionMeta(submission, submission, False, "jax")
    )
