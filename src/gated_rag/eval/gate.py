"""The gate: compare measured metrics to configured thresholds and FAIL the run if unmet.

This is the whole point of the project — retrieval quality is enforced like a data-quality check,
not just reported.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import GateConfig


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: dict[str, tuple[float, float]]   # metric -> (measured, threshold) for each breach


def evaluate_gate(measured: dict[str, float], cfg: GateConfig) -> GateResult:
    """Compare measured metrics against cfg.thresholds.

    Hint: for each (name, threshold) in cfg.thresholds, look up measured[name]; record a failure
    if measured < threshold (or missing). passed = no failures.
    """
    # TODO: build GateResult from threshold comparisons.
    raise NotImplementedError


def enforce(result: GateResult, cfg: GateConfig) -> None:
    """Act on the gate: block when on_failure == 'fail', else warn.

    Hint: if not result.passed and cfg.on_failure == 'fail' -> raise SystemExit(1) with the breaches;
    if 'warn' -> log and continue. This is what makes the pipeline gate.
    """
    # TODO: raise SystemExit on hard failure, otherwise log warnings.
    raise NotImplementedError
