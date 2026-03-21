"""Benchmark helpers for Phase 9 performance suites."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkResult:
    """Simple benchmark result payload."""

    name: str
    runs: int
    total_seconds: float
    avg_seconds: float


def benchmark_callable(
    *,
    name: str,
    func: Callable[[], T],
    runs: int = 3,
) -> tuple[T, BenchmarkResult]:
    """Run callable repeatedly and return last output with timing stats."""
    if runs <= 0:
        raise ValueError("runs must be greater than 0")

    output: T | None = None
    start = perf_counter()
    for _ in range(runs):
        output = func()
    total = perf_counter() - start

    if output is None:
        raise RuntimeError("benchmark_callable failed to execute target function")

    return output, BenchmarkResult(
        name=name,
        runs=runs,
        total_seconds=float(total),
        avg_seconds=float(total / runs),
    )
