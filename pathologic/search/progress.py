"""Progress tracking helpers for search workflows."""

from __future__ import annotations

import logging
import sys
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
        self._stage_bar = stage_bar
        self._candidate_bar = candidate_bar
        self._run_logger = run_logger

    def update(self, stage: str, *, state: str, detail: str | None = None) -> None:
        elapsed_stage = float(monotonic() - self._step_start)
        stage_label = f"{stage}:{state}"

        if self._show_candidate_progress:
            self._stage_bar.set_postfix(
                stage=stage_label,
                elapsed=f"{elapsed_stage:.1f}s",
            )
            self._stage_bar.refresh()
            self._candidate_bar.set_postfix(
                model=self._candidate_name,
                stage=stage_label,
                elapsed=f"{elapsed_stage:.1f}s",
            )
            self._candidate_bar.refresh()

        if (
            stage in self._stage_order
            and state in {"done", "failed"}
            and stage not in self._stage_done
        ):
            self._stage_done.add(stage)
            self._stage_bar.update(1)

        message = (
            f"[candidate {self._index}/{self._total_candidates}] "
            f"{self._candidate_name} {stage_label}"
        )
        if detail:
            message += f" ({detail})"

        if self._show_candidate_progress:
            tqdm.write(colorize(message, "cyan"), file=sys.stderr)
            self._run_logger.info(message)
        else:
            emit(message, color="cyan", run_logger=self._run_logger)
