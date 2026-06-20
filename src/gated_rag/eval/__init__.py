"""Eval package: metrics + the threshold gate.

Typical flow (wired later): load golden set -> run retriever over each question -> compute metrics
at eval.k_values -> evaluate_gate(measured, cfg) -> enforce(result, cfg).
"""
from .gate import GateResult, enforce, evaluate_gate

__all__ = ["GateResult", "evaluate_gate", "enforce"]
