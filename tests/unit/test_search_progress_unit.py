from __future__ import annotations

import logging

from pathologic.search.progress import CandidateProgressTracker


class _FakeBar:
    def __init__(self) -> None:
        self.n = 0
        self.postfix: dict[str, str] = {}
        self.refresh_count = 0

    def set_postfix(self, **kwargs) -> None:
        self.postfix = {str(k): str(v) for k, v in kwargs.items()}

    def refresh(self) -> None:
        self.refresh_count += 1

    def update(self, value: int) -> None:
        self.n += int(value)


def test_format_mm_ss_outputs_expected_values() -> None:
    assert CandidateProgressTracker._format_mm_ss(0.0) == "00:00"
    assert CandidateProgressTracker._format_mm_ss(5.4) == "00:05"
    assert CandidateProgressTracker._format_mm_ss(65.0) == "01:05"


def test_update_done_adds_stage_duration_to_message(monkeypatch) -> None:
    emitted_messages: list[str] = []

    def _fake_emit(message: str, **kwargs) -> None:
        emitted_messages.append(message)

    monkeypatch.setattr("pathologic.search.progress.emit", _fake_emit)

    stage_bar = _FakeBar()
    candidate_bar = _FakeBar()
    tracker = CandidateProgressTracker(
        index=1,
        total_candidates=3,
        candidate_name="xgboost",
        show_candidate_progress=False,
        step_start=0.0,
        stage_order=("hpo", "train"),
        stage_bar=stage_bar,
        candidate_bar=candidate_bar,
        run_logger=logging.getLogger("test.search.progress"),
    )

    # Ensure deterministic elapsed computation for this test.
    monotonic_values = iter([10.0, 12.0])
    monkeypatch.setattr(
        "pathologic.search.progress.monotonic",
        lambda: next(monotonic_values),
    )

    try:
        tracker.update("hpo", state="start")
        tracker.update("hpo", state="done", detail="ok")
    finally:
        tracker.close()

    assert stage_bar.n == 1
    assert emitted_messages
    assert "hpo:done" in emitted_messages[-1]
    assert "[00:02]" in emitted_messages[-1]
