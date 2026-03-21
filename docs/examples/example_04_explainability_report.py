"""Phase 8 example: explainability workflow with optional HTML export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pathologic import PathoLogic


def run_explainability_workflow(data_path: str, output_html: str | None = None) -> dict[str, Any]:
    model = PathoLogic("logreg")
    model.train(data_path)
    report = model.explain(data_path)

    html_path: str | None = None
    html_content = report.get("visual_report_html")
    if output_html is not None and isinstance(html_content, str):
        out = Path(output_html)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html_content, encoding="utf-8")
        html_path = str(out)

    return {
        "model_name": model.model_name,
        "backend": report.get("backend"),
        "global_feature_count": len(report.get("global_feature_importance", [])),
        "html_path": html_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run explainability report example.")
    parser.add_argument("data_path", help="Path to CSV/Parquet dataset")
    parser.add_argument("--output-html", default=None, help="Optional output HTML path")
    args = parser.parse_args()

    result = run_explainability_workflow(args.data_path, output_html=args.output_html)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
