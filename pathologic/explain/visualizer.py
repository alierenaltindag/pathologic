"""Minimal HTML visual report builder for explainability outputs."""

from __future__ import annotations

from html import escape

from pathologic.explain.schemas import ExplainabilityReport, FeatureAttribution, SampleExplanation


class ExplainabilityVisualizer:
    """Render explainability reports into lightweight HTML for auditing."""

    def render_html(self, report: ExplainabilityReport) -> str:
        max_global_abs = max(
            (item.absolute_contribution for item in report.global_feature_importance),
            default=1.0,
        )
        global_rows = "".join(
            self._render_global_row(item, max_global_abs)
            for item in report.global_feature_importance
        )

        sample_blocks = "".join(
            self._render_sample_block(sample)
            for sample in report.sample_explanations
        )

        summary_items = [
            ("backend", escape(report.backend)),
            ("global_features", str(len(report.global_feature_importance))),
            ("sample_explanations", str(len(report.sample_explanations))),
            ("fp_hotspots", str(len(report.false_positive_hotspots))),
        ]
        summary_cards = (
            "<div class='cards'>"
            + "".join(
                (
                    "<div class='card stat'>"
                    f"<div class='k'>{label}</div>"
                    f"<div class='v'>{value}</div>"
                    "</div>"
                )
                for label, value in summary_items
            )
            + "</div>"
        )

        hotspot_columns = self._resolve_hotspot_columns(report.false_positive_hotspots)
        hotspot_rows = "".join(
            "<tr>" + self._render_hotspot_cells(item, hotspot_columns) + "</tr>"
            for item in report.false_positive_hotspots
        )

        hotspot_headers = "".join(
            f"<th>{escape(str(key))}</th>"
            for key in hotspot_columns
        )

        member_section = self._render_member_explainability(report.member_explainability)

        metadata_rows = "".join(
            "<tr>"
            f"<td>{escape(str(key))}</td>"
            f"<td>{escape(str(value))}</td>"
            "</tr>"
            for key, value in sorted(report.metadata.items(), key=lambda pair: pair[0])
        )

        methodology_section = self._render_methodology_tr(report)

        return (
            "<html><head><meta charset='utf-8'>"
            "<title>PathoLogic Explainability Report</title>"
            "<style>"
            "body{font-family:Arial,sans-serif;margin:24px;line-height:1.4;color:#102a43;background:#f7fafc;}"
            "h1,h2{margin-bottom:8px;color:#0b2540;}"
            "table{border-collapse:collapse;width:100%;margin-bottom:16px;}"
            "th,td{border:1px solid #d9e2ec;padding:6px;text-align:left;background:#fff;}"
            "th{background:#e6edf4;font-weight:600;}"
            ".cards{display:grid;"
            "grid-template-columns:repeat(auto-fit,minmax(160px,1fr));"
            "gap:10px;margin:14px 0 18px 0;}"
            ".card{border:1px solid #d9e2ec;padding:10px;"
            "background:#fff;border-radius:8px;margin-bottom:12px;}"
            ".card.stat{margin-bottom:0;}"
            ".k{text-transform:uppercase;font-size:11px;color:#486581;letter-spacing:0.6px;}"
            ".v{font-size:20px;font-weight:700;color:#102a43;}"
            ".bar-track{height:8px;width:100%;background:#d9e2ec;border-radius:999px;overflow:hidden;}"
            ".bar-fill{height:100%;background:linear-gradient(90deg,#2f80ed,#27ae60);border-radius:999px;}"
            "</style></head><body>"
            "<h1>PathoLogic Explainability Report</h1>"
            f"{summary_cards}"
            "<h2>Yontem Ozeti (Turkce)</h2>"
            f"{methodology_section}"
            "<h2>Global Feature Importance</h2>"
            "<table><thead><tr>"
            "<th>feature</th><th>contribution</th><th>abs_contribution</th><th>biological_label</th><th>relative_strength</th>"
            "</tr></thead><tbody>"
            f"{global_rows}"
            "</tbody></table>"
            f"{member_section}"
            "<h2>Sample Explanations</h2>"
            f"{sample_blocks}"
            "<h2>False-Positive Hotspots</h2>"
            "<table><thead><tr>"
            f"{hotspot_headers}"
            "</tr></thead><tbody>"
            f"{hotspot_rows}"
            "</tbody></table>"
            "<h2>Metadata</h2>"
            "<table><thead><tr><th>key</th><th>value</th></tr></thead><tbody>"
            f"{metadata_rows}"
            "</tbody></table>"
            "</body></html>"
        )

    @staticmethod
    def _render_methodology_tr(report: ExplainabilityReport) -> str:
        metadata = report.metadata
        backend_policy = escape(str(metadata.get("backend_policy", "auto")))
        background_size = int(metadata.get("background_size", 0))
        top_k_features = int(metadata.get("top_k_features", 0))
        top_k_samples = int(metadata.get("top_k_samples", 0))
        group_columns_raw = metadata.get("group_columns", [])
        group_columns = (
            ", ".join(str(item) for item in group_columns_raw)
            if isinstance(group_columns_raw, list) and group_columns_raw
            else "gene_id, domain_id, protein_family"
        )

        lines = [
            "<li>Bu rapor, model tahminleri uzerinden ozellik katkilari hesaplayarak "
            "yorumlanabilirlik sunar.</li>",
            "<li>Atif backend secimi metadata alanindaki backend_policy degeriyle "
            "yonetilir; aktif backend rapor ozetinde verilir.</li>",
            f"<li>Arka plan orneklemi: {background_size}; global onem ve ornek "
            "aciklamalari bu dagilim uzerinden uretilir.</li>",
            f"<li>Global onem siralamasi, ozelliklerin mutlak katki degerlerine gore "
            f"yapilir (top_k_features={top_k_features}).</li>",
            f"<li>Ornek bazli aciklamada her satir icin en etkili ozellikler listelenir "
            f"(top_k_samples={top_k_samples}).</li>",
            "<li>False-positive hotspot analizi, grup bazli false_positive_rate / "
            "overall_false_positive_rate risk oranini raporlar.</li>",
            f"<li>Hotspot grup kolonlari: {escape(group_columns)}.</li>",
            f"<li>Bu kosu icin backend policy: {backend_policy}.</li>",
        ]
        return "<div class='card'><ul>" + "".join(lines) + "</ul></div>"

    @staticmethod
    def _safe_percentage(value: float, max_value: float) -> float:
        if max_value <= 0.0:
            return 0.0
        return max(min((value / max_value) * 100.0, 100.0), 0.0)

    @staticmethod
    def _resolve_hotspot_columns(rows: list[dict[str, float | int | str]]) -> list[str]:
        if not rows:
            return []

        priority = [
            "group_column",
            "gene_id",
            "Protein change",
            "false_positive_count",
            "negative_count",
            "false_positive_rate",
            "overall_false_positive_rate",
            "false_positive_risk_ratio",
        ]
        all_keys: set[str] = set()
        for item in rows:
            all_keys.update(str(key) for key in item.keys())

        ordered = [key for key in priority if key in all_keys]
        remaining = sorted(key for key in all_keys if key not in set(ordered))
        return ordered + remaining

    @staticmethod
    def _render_hotspot_cells(
        item: dict[str, float | int | str],
        columns: list[str],
    ) -> str:
        rendered: list[str] = []
        for key in columns:
            value = item.get(key, "")
            if key == "false_positive_risk_ratio":
                numeric_value = float(value) if value != "" else 0.0
                bar_width = min(numeric_value * 25.0, 100.0)
                rendered.append(
                    "<td>"
                    + (f"{numeric_value:.4f}" if value != "" else "")
                    + "<div class='bar-track'>"
                    + "<div class='bar-fill' style='width:"
                    + f"{bar_width:.2f}%"
                    + ";'></div>"
                    + "</div>"
                    + "</td>"
                )
            else:
                rendered.append(f"<td>{escape(str(value))}</td>")
        return "".join(rendered)

    @staticmethod
    def _render_member_explainability(member_payload: object) -> str:
        if not isinstance(member_payload, dict):
            return ""

        members_raw = member_payload.get("members")
        if not isinstance(members_raw, dict) or not members_raw:
            return "<h2>Member Explainability</h2><div class='card'>No member entries.</div>"

        blocks: list[str] = []
        for alias, payload in members_raw.items():
            if not isinstance(payload, dict):
                continue
            status = escape(str(payload.get("status", "unknown")))
            backend = escape(str(payload.get("backend", "unknown")))
            weight_value = payload.get("weight")
            weight_text = (
                f"<div><strong>weight:</strong> {float(weight_value):.4f}</div>"
                if isinstance(weight_value, (int, float))
                else ""
            )
            score_value = payload.get("weight_score")
            score_text = (
                f"<div><strong>weight_score:</strong> {float(score_value):.4f}</div>"
                if isinstance(score_value, (int, float))
                else ""
            )
            features = payload.get("global_feature_importance")
            top_features = features if isinstance(features, list) else []

            feature_rows = "".join(
                "<tr>"
                f"<td>{escape(str(item.get('feature', '')))}</td>"
                f"<td>{float(item.get('absolute_contribution', 0.0)):.6f}</td>"
                f"<td>{escape(str(item.get('biological_label', '')))}</td>"
                "</tr>"
                for item in top_features[:5]
                if isinstance(item, dict)
            )
            if not feature_rows:
                feature_rows = "<tr><td colspan='3'>No feature attributions</td></tr>"

            blocks.append(
                "<div class='card'>"
                f"<h3>{escape(str(alias))}</h3>"
                f"<div><strong>status:</strong> {status}</div>"
                f"<div><strong>backend:</strong> {backend}</div>"
                f"{weight_text}"
                f"{score_text}"
                "<table><thead><tr><th>feature</th><th>abs_contribution</th><th>biological_label</th></tr></thead><tbody>"
                f"{feature_rows}"
                "</tbody></table>"
                "</div>"
            )

        if not blocks:
            return ""
        return "<h2>Member Explainability</h2>" + "".join(blocks)

    def _render_global_row(self, item: FeatureAttribution, max_global_abs: float) -> str:
        relative_strength = self._safe_percentage(item.absolute_contribution, max_global_abs)
        return (
            "<tr>"
            f"<td>{escape(item.feature)}</td>"
            f"<td>{item.contribution:.6f}</td>"
            f"<td>{item.absolute_contribution:.6f}</td>"
            f"<td>{escape(item.biological_label)}</td>"
            "<td>"
            "<div class='bar-track'>"
            "<div class='bar-fill' style='width:"
            f"{relative_strength:.2f}%"
            ";'></div>"
            "</div>"
            "</td>"
            "</tr>"
        )

    @staticmethod
    def _render_sample_block(sample: SampleExplanation) -> str:
        top_features_html = ""
        for item in sample.top_features:
            top_features_html += (
                "<li>"
                f"{escape(item.feature)}: {item.contribution:.6f} "
                f"({escape(item.biological_label)})"
                "</li>"
            )

        return (
            "<div class='card'>"
            f"<div><strong>row_index:</strong> {int(sample.row_index)}</div>"
            f"<div><strong>score:</strong> {float(sample.score):.6f}</div>"
            f"<div><strong>predicted_label:</strong> {int(sample.predicted_label)}</div>"
            f"<div><strong>narrative:</strong> {escape(sample.narrative)}</div>"
            "<div><strong>top_features:</strong><ul>"
            f"{top_features_html}"
            "</ul></div></div>"
        )
