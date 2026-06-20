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

    A metric breaches if it is missing from `measured` or below its threshold. A missing metric is
    recorded with measured = float('nan') so the breach is visible rather than silently skipped.
    """
    failures: dict[str, tuple[float, float]] = {}
    for name, threshold in cfg.thresholds.items():
        if name not in measured:
            failures[name] = (float("nan"), threshold)
        elif measured[name] < threshold:
            failures[name] = (measured[name], threshold)
    return GateResult(passed=not failures, failures=failures)


def enforce(result: GateResult, cfg: GateConfig) -> None:
    """Act on the gate: block when on_failure == 'fail', else warn.

    On a hard failure this raises SystemExit(1) — that non-zero exit is what makes the pipeline
    gate (a CI step or job run goes red). On 'warn' it prints the breaches and continues.
    """
    if result.passed:
        print("[gate] PASS — all thresholds met")
        return

    breaches = ", ".join(
        f"{name}={measured:.3f} < {threshold:.3f}"
        for name, (measured, threshold) in result.failures.items()
    )
    if cfg.on_failure == "fail":
        raise SystemExit(f"[gate] FAIL — retrieval quality below threshold: {breaches}")
    print(f"[gate] WARN — thresholds unmet (on_failure=warn): {breaches}")
