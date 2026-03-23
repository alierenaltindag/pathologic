"""Progress tracking helpers for search workflows."""

from __future__ import annotations

import logging
import sys
from threading import Event, Lock, Thread
from time import monotonic

from tqdm import tqdm

from pathologic.search.logging import colorize, emit


class CandidateProgressTracker:
    """Track and emit stage-level progress for one candidate."""

    def __init__(
        self,
        *,
        index: int,
        total_candidates: int,
        candidate_name: str,
        show_candidate_progress: bool,
        step_start: float,
        stage_order: tuple[str, ...],
        stage_bar: tqdm,
        candidate_bar: tqdm,
        run_logger: logging.Logger,
    ) -> None:
        self._index = int(index)
        self._total_candidates = int(total_candidates)
        self._candidate_name = str(candidate_name)
        self._show_candidate_progress = bool(show_candidate_progress)
        self._step_start = float(step_start)
        self._stage_order = tuple(stage_order)
        self._stage_done: set[str] = set()
        self._stage_started: dict[str, float] = {}
        self._stage_bar = stage_bar
        self._candidate_bar = candidate_bar
        self._run_logger = run_logger
        self._lock = Lock()
        self._stop_refresh = Event()
        self._refresh_thread = Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()

    @staticmethod
    def _format_mm_ss(seconds: float) -> str:
        safe = max(int(round(seconds)), 0)
        minutes = safe // 60
        remainder = safe % 60
        return f"{minutes:02d}:{remainder:02d}"

    def _estimate_remaining_seconds(self, *, elapsed_seconds: float) -> float | None:
        total = float(self._total_candidates)
        if total <= 0:
            return None

        completed_candidates = float(self._candidate_bar.n)
        completed_stages = float(len(self._stage_done))
        total_stages = float(len(self._stage_order))
        stage_fraction = (completed_stages / total_stages) if total_stages > 0 else 0.0
        progress = completed_candidates + stage_fraction
        if progress <= 0.0:
            return None

        rate = progress / max(elapsed_seconds, 1e-9)
        if rate <= 0.0:
            return None
        return max((total - progress) / rate, 0.0)

    def _refresh_loop(self) -> None:
        while not self._stop_refresh.wait(1.0):
            if not self._show_candidate_progress:
                continue
            with self._lock:
                self._apply_live_postfix()

    def _apply_live_postfix(self) -> None:
        elapsed_stage = float(monotonic() - self._step_start)
        remaining = self._estimate_remaining_seconds(elapsed_seconds=elapsed_stage)
        remaining_text = self._format_mm_ss(remaining) if remaining is not None else "--:--"

        current_stage = "pending"
        if self._stage_started:
            # Preserve insertion order and report the latest started stage.
            current_stage = next(reversed(self._stage_started.keys()))

        self._stage_bar.set_postfix(
            stage=current_stage,
            elapsed=self._format_mm_ss(elapsed_stage),
            remaining=remaining_text,
        )
        self._stage_bar.refresh()
        self._candidate_bar.set_postfix(
            model=self._candidate_name,
            stage=current_stage,
            elapsed=self._format_mm_ss(elapsed_stage),
            remaining=remaining_text,
        )
        self._candidate_bar.refresh()

    def close(self) -> None:
        self._stop_refresh.set()
        if self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=1.5)

    def update(self, stage: str, *, state: str, detail: str | None = None) -> None:
        stage_label = f"{stage}:{state}"
        with self._lock:
            if state == "start":
                self._stage_started[stage] = monotonic()

            if (
                stage in self._stage_order
                and state in {"done", "failed"}
                and stage not in self._stage_done
            ):
                self._stage_done.add(stage)
                self._stage_bar.update(1)

            stage_duration: float | None = None
            if state in {"done", "failed"}:
                start_ts = self._stage_started.get(stage)
                if start_ts is not None:
                    stage_duration = max(monotonic() - start_ts, 0.0)

            if self._show_candidate_progress:
                self._apply_live_postfix()

        message = (
            f"[candidate {self._index}/{self._total_candidates}] "
            f"{self._candidate_name} {stage_label}"
        )
        if state in {"done", "failed"} and stage_duration is not None:
            message += f" [{self._format_mm_ss(stage_duration)}]"
        if detail:
            message += f" ({detail})"

        if self._show_candidate_progress:
            tqdm.write(colorize(message, "cyan"), file=sys.stderr)
            self._run_logger.info(message)
        else:
            emit(message, color="cyan", run_logger=self._run_logger)
