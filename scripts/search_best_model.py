"""Compatibility entrypoint for exhaustive model search.

This script is intentionally thin and delegates implementation to
`pathologic.search` modules. It preserves legacy import surface used by
existing tests and external tooling while keeping orchestration logic in
package modules.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathologic.search import candidate as _search_candidate
from pathologic.search import artifacts as _search_artifacts
from pathologic.search import cli as _search_cli
from pathologic.search import core as _search_core
from pathologic.search import data as _search_data
from pathologic.search import explainability as _search_explainability
from pathologic.search.utils import parse_model_pool as _parse_model_pool


# Legacy compatibility exports used by tests and downstream tooling.
prepare_dataset_for_pathologic = _search_data.prepare_dataset_for_pathologic
_build_candidate_specs = _search_candidate.build_candidate_specs
_build_pair_tuning_search_space = _search_candidate.build_pair_tuning_search_space
_build_hybrid_strategy_tuning_search_space = _search_candidate.build_hybrid_strategy_tuning_search_space
_parse_hybrid_weights = _search_candidate.parse_hybrid_weights
_resolve_hybrid_strategy_config = _search_candidate.resolve_hybrid_strategy_config
_resolve_hybrid_config_for_report = _search_candidate.resolve_hybrid_config_for_report
_extract_scores_from_model = _search_artifacts.extract_scores_from_model
_build_error_analysis_run_summary = _search_artifacts.build_error_analysis_run_summary
_build_global_importance_label = _search_explainability.build_global_importance_label
_build_hotspot_label = _search_explainability.build_hotspot_label


# CLI entrypoint exports.
build_arg_parser = _search_cli.build_arg_parser
run_exhaustive_search = _search_core.run_exhaustive_search


def _configure_windows_joblib_cpu_detection() -> None:
    """Avoid loky WMIC subprocess noise on Windows images without wmic."""
    if os.name != "nt":
        return
    if os.environ.get("LOKY_MAX_CPU_COUNT"):
        return
    cpu_count = os.cpu_count()
    if not isinstance(cpu_count, int) or cpu_count < 1:
        return
    cpu_limit = max(1, cpu_count - 1)
    os.environ["LOKY_MAX_CPU_COUNT"] = str(cpu_limit)


def _suppress_known_parallel_warnings() -> None:
    """Keep known loky core-detection warnings from polluting CLI output."""
    warnings.filterwarnings(
        "ignore",
        message=r"Could not find the number of physical cores",
    )


def main(argv: list[str] | None = None) -> int:
    return _search_cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
