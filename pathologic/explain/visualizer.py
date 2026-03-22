"""Minimal HTML visual report builder for explainability outputs."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from pathologic.explain.schemas import ExplainabilityReport, FeatureAttribution, SampleExplanation


class ExplainabilityVisualizer:
    """Render explainability reports into lightweight HTML for auditing."""

    def render_error_report_html(self, results: dict[str, Any], output_path: str) -> str:
        """Renders a standalone error analysis report."""
        patterns = results.get("summary", {}).get("pattern_concentration", {}) or results.get("pattern_concentration", {})
        biological = results.get("summary", {}).get("biological_context", []) or results.get("biological_context", [])
        artifacts = results.get("artifacts", {}) if isinstance(results.get("artifacts"), dict) else {}
        plot_file = (
            results.get("summary", {}).get("surrogate_tree", {}).get("plot_file")
            or "surrogate_error_tree.png"
        )
        
        # Build the HTML
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f4f7f6; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #2980b9; margin-top: 30px; }}
                .card {{ background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
                th {{ background-color: #f8f9fa; color: #2c3e50; font-weight: 600; }}
                tr:hover {{ background-color: #f1f1f1; }}
                .error-type-fn {{ color: #e74c3c; font-weight: bold; }}
                .error-type-fp {{ color: #e67e22; font-weight: bold; }}
                .img-container {{ text-align: center; margin: 20px 0; }}
                img {{ max-width: 100%; border-radius: 8px; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>PathoLogic Hata Analiz Raporu</h1>
            <p>Bu rapor, modelin yanlış tahmin yaptığı varyantlar üzerindeki sistematik desenleri incelemektedir.</p>
            
            {self._render_pattern_analysis(patterns)}
            
            <div class='card'>
                <h2>Hata Karar Agaci (Surrogate Tree)</h2>
                <p>Aşağıdaki karar ağacı, modelin FP (False Positive) ve FN (False Negative) hataları arasındaki ayrımı hangi özelliklere dayanarak yaptığını gösterir.</p>
                <div class='img-container'>
                    <img src='{escape(str(plot_file))}' alt='Error Decision Tree'>
                </div>
                <p><i>Not: Agacın sol dalları düşük değerleri, sağ dalları yüksek değerleri temsil eder. Renk derinliği hata türünün yoğunluğunu belirtir (Turuncu: FP, Mavi: FN).</i></p>
            </div>
            
            <div class='card'>
                <h2>Biyolojik Baglam (Gene/Family/Domain)</h2>
                <p>Hata yapan varyantların protein aileleri ve domainleri bazlı dağılımı:</p>
                {self._render_error_analysis(biological)}
            </div>

            {self._render_error_image_gallery(artifacts, fallback_tree=plot_file)}
        </body>
        </html>
        """
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        return output_path

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
            f"<li>Analiz grup kolonlari: {escape(group_columns)}.</li>",
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

    @staticmethod
    def _render_error_analysis(errors: list[dict[str, Any]]) -> str:
        """Render biological context errors."""
        if not errors:
            return "<p>No systematic biological context errors detected.</p>"
        
        # Priority columns for biological context
        columns = ["gene_id", "protein_family", "domain_id", "fp_count", "fn_count", "total_errors"]
        
        # Discover any other columns
        all_keys = set()
        for err in errors:
            all_keys.update(err.keys())
        
        ordered_cols = [c for c in columns if c in all_keys]
        remaining = sorted([c for c in all_keys if c not in columns])
        final_cols = ordered_cols + remaining
        
        headers = "".join(f"<th>{escape(str(c))}</th>" for c in final_cols)
        rows = ""
        for err in errors:
            cells = "".join(f"<td>{escape(str(err.get(c, '')))}</td>" for c in final_cols)
            rows += f"<tr>{cells}</tr>"
            
        return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"

    @staticmethod
    def _render_pattern_analysis(patterns: dict[str, Any] | None) -> str:
        if not patterns:
            return ""

        sections: list[str] = []

        # Global stats
        if "error_type_distribution" in patterns:
            dist = patterns["error_type_distribution"]
            sections.append(
                "<div class='card'>"
                "<h3>Hata Turu Dagilimi</h3>"
                f"<div><strong>Toplam Hata:</strong> {dist['total']}</div>"
                f"<div><strong>False Positive (FP):</strong> {dist['fp']} (Benign varyantın Pathogenic tahmin edilmesi)</div>"
                f"<div><strong>False Negative (FN):</strong> {dist['fn']} (Pathogenic varyantın Benign tahmin edilmesi)</div>"
                "</div>"
            )

        # 1. Population Frequency
        if "population_frequency" in patterns:
            af_rows = ""
            for k, v in patterns["population_frequency"].items():
                if isinstance(v, dict):
                    af_rows += f"<tr><td>{escape(str(k))}</td><td>{v['fp']}</td><td>{v['fn']}</td><td>{v['total']}</td></tr>"
                else: 
                   af_rows += f"<tr><td>{escape(str(k))}</td><td colspan='3'>{int(v)}</td></tr>"
            
            sections.append(
                "<div class='card'>"
                "<h3>Populasyon Sikligi (gnomAD_AF) Bazlı Hata Analizi</h3>"
                "<table><thead><tr><th>Frekans Araligi</th><th>FP Sayısı</th><th>FN Sayısı</th><th>Toplam Hata</th></tr></thead>"
                f"<tbody>{af_rows}</tbody></table></div>"
            )

        # 2. In-Silico Conflicts
        if "insilico_conflicts" in patterns:
            conf = patterns["insilico_conflicts"]
            conf_rows = ""
            for scenario, stats in conf.items():
                label = "REVEL High (>0.5) / CADD Low (<15)" if scenario == "revel_high_cadd_low" else "REVEL Low (<0.2) / CADD High (>25)"
                if isinstance(stats, dict):
                    conf_rows += f"<tr><td>{label}</td><td>{stats['fp']}</td><td>{stats['fn']}</td><td>{stats['total']}</td></tr>"
                else:
                    conf_rows += f"<tr><td>{label}</td><td colspan='3'>{int(stats)}</td></tr>"

            sections.append(
                "<div class='card'>"
                "<h3>In-Silico Skor Celiskileri Bazlı Hata Analizi</h3>"
                "<table><thead><tr><th>Senaryo</th><th>FP Sayısı</th><th>FN Sayısı</th><th>Toplam Hata</th></tr></thead>"
                f"<tbody>{conf_rows}</tbody></table></div>"
            )

        # 3. Biochemical Patterns
        if "biochemical_patterns" in patterns:
            bio = patterns["biochemical_patterns"]
            bio_rows = ""
            for category, pattern_stats in bio.items():
                for label, stats in pattern_stats.items():
                    if isinstance(stats, dict):
                        bio_rows += f"<tr><td>{escape(category.capitalize())}</td><td>{escape(str(label))}</td><td>{stats['fp']}</td><td>{stats['fn']}</td><td>{stats['total']}</td></tr>"
                    else:
                        bio_rows += f"<tr><td>{escape(category.capitalize())}</td><td>{escape(str(label))}</td><td colspan='3'>{int(stats)}</td></tr>"
            
            sections.append(
                "<div class='card'>"
                "<h3>Biyokimyasal Degisim Desenleri Bazlı Hata Analizi</h3>"
                "<table><thead><tr><th>Kategori</th><th>Desen</th><th>FP Sayısı</th><th>FN Sayısı</th><th>Toplam Hata</th></tr></thead>"
                f"<tbody>{bio_rows}</tbody></table></div>"
            )

        if not sections:
            return ""

        return "<h2>Hata Analizi Desenleri (Pattern Concentration Analysis)</h2>" + "".join(sections)

    @staticmethod
    def _render_error_image_gallery(artifacts: dict[str, Any], *, fallback_tree: str) -> str:
        if not artifacts:
            return ""

        ordered_keys = [
            "surrogate_tree_plot_png",
            "surrogate_tree_shap_png",
            "errors_tsne_kmeans_png",
            "errors_tsne_dbscan_png",
            "errors_umap_kmeans_png",
            "errors_umap_dbscan_png",
        ]
        seen: set[str] = set()
        rows: list[str] = []

        for key in ordered_keys:
            value = artifacts.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            filename = Path(value).name
            if filename in seen:
                continue
            seen.add(filename)
            rows.append(
                "<div class='card'>"
                f"<h3>{escape(filename)}</h3>"
                "<div class='img-container'>"
                f"<img src='{escape(filename)}' alt='{escape(filename)}'>"
                "</div></div>"
            )

        if fallback_tree and fallback_tree not in seen:
            rows.insert(
                0,
                "<div class='card'>"
                f"<h3>{escape(str(fallback_tree))}</h3>"
                "<div class='img-container'>"
                f"<img src='{escape(str(fallback_tree))}' alt='{escape(str(fallback_tree))}'>"
                "</div></div>",
            )

        if not rows:
            return ""
        return "<h2>Hata Analizi Gorselleri</h2>" + "".join(rows)
