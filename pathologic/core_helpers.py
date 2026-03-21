"""Shared helper functions used by PathoLogic core orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from pathologic.models.hybrid import normalize_model_alias, parse_hybrid_alias
from pathologic.models.registry import list_registered_models


def load_config_from_path(*, config_dir: Path, path_value: str) -> dict[str, Any]:
    """Load YAML config from absolute path or path relative to config directory."""
    raw = Path(path_value)
    config_path = raw if raw.is_absolute() else config_dir / raw

    if not config_path.exists():
        raise ValueError(f"Model config path does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Model config file must be a mapping: {config_path}")
    return loaded


def load_defaults(*, config_dir: Path) -> dict[str, Any]:
    """Load top-level defaults and resolve modular runtime section references."""
    defaults_path = config_dir / "defaults.yaml"
    with defaults_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Default config must be a mapping.")

    for section_name in ("train", "test", "tune", "explain", "finetune"):
        section = loaded.get(section_name)
        if isinstance(section, str):
            loaded[section_name] = load_config_from_path(config_dir=config_dir, path_value=section)
        elif isinstance(section, dict):
            config_path_raw = section.get("config_path")
            if isinstance(config_path_raw, str) and config_path_raw.strip():
                base_section = load_config_from_path(
                    config_dir=config_dir,
                    path_value=config_path_raw,
                )
                inline_overrides = {
                    key: value for key, value in section.items() if key != "config_path"
                }
                loaded[section_name] = {**base_section, **inline_overrides}

    return loaded


def mlp_constructor_keys() -> set[str]:
    """Allowed constructor keys for MLP wrapper when reading model block."""
    return {
        "hidden_layer_sizes",
        "activation",
        "solver",
        "alpha",
        "max_epochs",
        "batch_size",
        "learning_rate_init",
        "optimizer",
        "scheduler",
        "early_stopping",
        "weight_decay",
        "max_iter",
    }


def resolve_model_config(
    *,
    defaults: dict[str, Any],
    model_name: str,
    runtime_model_config: dict[str, Any],
    config_dir: Path,
) -> dict[str, Any]:
    """Resolve model config from runtime payload or defaults section references."""
    if runtime_model_config:
        return dict(runtime_model_config)

    models_section = defaults.get("models", {})
    if not isinstance(models_section, dict):
        raise ValueError("Config field 'models' must be a mapping.")

    entry = models_section.get(model_name, {})
    if entry is None:
        resolved: dict[str, Any] = {}
    elif isinstance(entry, str):
        loaded = load_config_from_path(config_dir=config_dir, path_value=entry)
        if model_name == "mlp" and isinstance(loaded.get("model"), dict):
            model_block = cast(dict[str, Any], loaded["model"])
            resolved = {
                "architecture_path": entry,
                "tuning_search_space": model_block.get("tuning_search_space"),
            }
            for key in mlp_constructor_keys():
                if key in model_block:
                    resolved[key] = model_block[key]
        else:
            resolved = dict(loaded)
    elif isinstance(entry, dict):
        config_path_raw = entry.get("config_path")
        if isinstance(config_path_raw, str) and config_path_raw.strip():
            loaded = load_config_from_path(config_dir=config_dir, path_value=config_path_raw)
            if model_name == "mlp" and isinstance(loaded.get("model"), dict):
                model_block = cast(dict[str, Any], loaded["model"])
                loaded = {
                    "architecture_path": config_path_raw,
                    "tuning_search_space": model_block.get("tuning_search_space"),
                }
                for key in mlp_constructor_keys():
                    if key in model_block:
                        loaded[key] = model_block[key]
            inline_overrides = {k: v for k, v in entry.items() if k != "config_path"}
            resolved = {**dict(loaded), **inline_overrides}
        else:
            resolved = dict(entry)
    else:
        raise ValueError(f"Config field 'models.{model_name}' must be a mapping or path string.")

    return resolved


def model_params_from_resolved_config(
    *,
    model_name: str,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Normalize resolved config into constructor params only."""
    params = dict(model_config)
    params.pop("tuning_search_space", None)
    params.pop("config_path", None)
    if model_name == "mlp":
        params.pop("type", None)
        params.pop("architecture", None)
    return params


def apply_mlp_train_fallbacks(
    *,
    model_params: dict[str, Any],
    train_config: dict[str, Any],
) -> None:
    """Apply train-level fallback params to MLP config without overriding model-level values."""
    if "max_epochs" not in model_params and "max_iter" not in model_params:
        if "epochs" in train_config:
            model_params["max_epochs"] = int(train_config["epochs"])

    if "batch_size" not in model_params and "batch_size" in train_config:
        model_params["batch_size"] = int(train_config["batch_size"])

    if "optimizer" not in model_params:
        optimizer_cfg = train_config.get("optimizer")
        if isinstance(optimizer_cfg, dict):
            model_params["optimizer"] = dict(optimizer_cfg)

    if "scheduler" not in model_params:
        scheduler_cfg = train_config.get("scheduler")
        if isinstance(scheduler_cfg, dict):
            model_params["scheduler"] = dict(scheduler_cfg)

    if "learning_rate_init" not in model_params:
        optimizer_cfg = train_config.get("optimizer")
        if isinstance(optimizer_cfg, dict) and "lr" in optimizer_cfg:
            model_params["learning_rate_init"] = float(optimizer_cfg["lr"])

    if "early_stopping" not in model_params:
        early_stopping_cfg = train_config.get("early_stopping")
        if isinstance(early_stopping_cfg, dict):
            model_params["early_stopping"] = dict(early_stopping_cfg)


def validate_model_name(model_name: str) -> None:
    """Validate model alias against registry or plus-composed hybrid syntax."""
    if not model_name:
        raise ValueError("Unsupported model ''. Supported models: at least one alias is required.")

    registered = set(list_registered_models())
    if "+" in model_name:
        members = parse_hybrid_alias(model_name)
        unknown = [alias for alias in members if normalize_model_alias(alias) not in registered]
        if unknown:
            available = ", ".join(sorted(registered))
            raise ValueError(
                "Unsupported model '"
                + model_name
                + "'. Unknown hybrid members: "
                + ", ".join(unknown)
                + ". Available models: "
                + available
            )
        return

    normalized = normalize_model_alias(model_name)
    if normalized not in registered:
        available = ", ".join(sorted(registered))
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {available}")


def resolve_split_config(
    *,
    defaults: dict[str, Any],
    train_config: dict[str, Any],
    tune_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve split config with precedence: tune.split > train.split > defaults.split."""
    top_level = defaults.get("split")
    base = dict(top_level) if isinstance(top_level, dict) else {}

    train_split = train_config.get("split")
    if isinstance(train_split, dict):
        base = {**base, **train_split}

    if tune_config is not None:
        tune_split = tune_config.get("split")
        if isinstance(tune_split, dict):
            base = {**base, **tune_split}

    return base


def resolve_preprocess_config(
    *,
    defaults: dict[str, Any],
    train_config: dict[str, Any],
    tune_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve preprocess config with precedence: tune > train > defaults."""
    top_level = defaults.get("preprocess")
    base = dict(top_level) if isinstance(top_level, dict) else {}

    train_preprocess = train_config.get("preprocess")
    if isinstance(train_preprocess, dict):
        base = {**base, **train_preprocess}

    if tune_config is not None:
        tune_preprocess = tune_config.get("preprocess")
        if isinstance(tune_preprocess, dict):
            base = {**base, **tune_preprocess}

    return base


def resolve_explain_config(
    *,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Resolve explain runtime config from defaults and data-level analysis columns."""
    explain = defaults.get("explain")
    resolved = dict(explain) if isinstance(explain, dict) else {}

    data_config_raw = defaults.get("data")
    data_config = data_config_raw if isinstance(data_config_raw, dict) else {}
    error_columns_raw = data_config.get("error_analysis_columns", [])
    if error_columns_raw is None:
        error_columns: list[str] = []
    elif isinstance(error_columns_raw, list):
        error_columns = [str(column) for column in error_columns_raw if str(column)]
    else:
        raise ValueError("Config field 'data.error_analysis_columns' must be a list.")

    if not error_columns:
        return resolved

    def _merge_group_columns(existing: Any) -> list[str]:
        base = [str(item) for item in existing] if isinstance(existing, list) else []
        merged = list(base)
        seen = set(base)
        for column in error_columns:
            if column not in seen:
                merged.append(column)
                seen.add(column)
        return merged

    resolved["group_columns"] = _merge_group_columns(resolved.get("group_columns"))

    false_positive_raw = resolved.get("false_positive")
    false_positive = dict(false_positive_raw) if isinstance(false_positive_raw, dict) else {}
    false_positive["group_columns"] = _merge_group_columns(false_positive.get("group_columns"))
    resolved["false_positive"] = false_positive
    return resolved


def resolve_finetune_config(
    *,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Resolve fine-tune runtime config from defaults."""
    fine_tune = defaults.get("finetune")
    return dict(fine_tune) if isinstance(fine_tune, dict) else {}


def normalize_preprocess_for_dataset(
    *,
    preprocess_config: dict[str, Any],
    dataset_columns: Any,
    gene_column: str,
) -> dict[str, Any]:
    """Normalize preprocess config for available dataset schema."""
    normalized = dict(preprocess_config)
    per_gene_enabled = bool(normalized.get("per_gene", True))
    if not per_gene_enabled:
        return normalized

    if gene_column in dataset_columns:
        return normalized

    policy = str(normalized.get("on_missing_gene_column", "disable")).strip().lower()
    if policy not in {"disable", "error"}:
        raise ValueError(
            "Config field 'preprocess.on_missing_gene_column' "
            "must be one of: disable, error"
        )
    if policy == "error":
        raise ValueError(
            "Gene-aware preprocessing requested but gene column is missing: "
            f"'{gene_column}'"
        )

    normalized["per_gene"] = False
    return normalized


def validate_preprocess_options(preprocess_config: dict[str, Any]) -> tuple[str, str, str]:
    """Validate preprocess options and return normalized strategy values."""
    missing_value_policy_raw = str(preprocess_config.get("missing_value_policy", "impute"))
    impute_strategy_raw = str(preprocess_config.get("impute_strategy", "none"))
    scaler_raw = str(preprocess_config.get("scaler", "standard"))
    allowed_policies = {"impute", "drop_rows"}
    allowed_imputers = {"none", "mean", "median", "most_frequent"}
    allowed_scalers = {"standard", "minmax"}
    if missing_value_policy_raw not in allowed_policies:
        raise ValueError(
            "Config field 'preprocess.missing_value_policy' must be one of: "
            + ", ".join(sorted(allowed_policies))
        )
    if impute_strategy_raw not in allowed_imputers:
        raise ValueError(
            "Config field 'preprocess.impute_strategy' must be one of: "
            + ", ".join(sorted(allowed_imputers))
        )
    if scaler_raw not in allowed_scalers:
        raise ValueError(
            "Config field 'preprocess.scaler' must be one of: "
            + ", ".join(sorted(allowed_scalers))
        )
    add_missing_indicators = preprocess_config.get("add_missing_indicators")
    if add_missing_indicators is not None and not isinstance(add_missing_indicators, bool):
        raise ValueError(
            "Config field 'preprocess.add_missing_indicators' must be a boolean."
        )

    missing_indicator_features = preprocess_config.get("missing_indicator_features")
    if missing_indicator_features is not None and not isinstance(missing_indicator_features, list):
        raise ValueError(
            "Config field 'preprocess.missing_indicator_features' must be a list."
        )

    return missing_value_policy_raw, impute_strategy_raw, scaler_raw


def deep_merge_mappings(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Deep merge two mappings and return a new dictionary."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge_mappings(current, value)
        else:
            merged[key] = value
    return merged


def merge_config_overrides(
    *,
    base_config: dict[str, Any],
    overrides: Mapping[str, Any],
    allowed_keys: set[str],
    context_name: str,
    deep_merge_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Merge validated overrides into config with optional deep-merge keys."""
    unknown = sorted(set(overrides) - allowed_keys)
    if unknown:
        raise ValueError(
            f"Unsupported {context_name} overrides: "
            + ", ".join(unknown)
            + "."
        )

    merged = dict(base_config)
    deep_keys = deep_merge_keys or set()
    for key, value in overrides.items():
        if (
            key in deep_keys
            and isinstance(merged.get(key), Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge_mappings(
                cast(Mapping[str, Any], merged[key]),
                cast(Mapping[str, Any], value),
            )
        else:
            merged[key] = value
    return merged
