"""Core high-level API for PathoLogic.

Phase 1 intentionally provides a stable API skeleton.
Model training/prediction internals are implemented in later phases.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Unpack, cast

import numpy as np
from sklearn.model_selection import train_test_split

from pathologic.api_overrides import (
    EvaluateOverrideKwargs,
    ExplainOverrideKwargs,
    FineTuneOverrideKwargs,
    PredictOverrideKwargs,
    TrainOverrideKwargs,
    TuneOverrideKwargs,
)
from pathologic.core_helpers import (
    apply_mlp_train_fallbacks,
    load_config_from_path,
    load_defaults,
    merge_config_overrides,
    mlp_constructor_keys,
    model_params_from_resolved_config,
    normalize_preprocess_for_dataset,
    resolve_explain_config,
    resolve_finetune_config,
    resolve_model_config,
    resolve_preprocess_config,
    resolve_split_config,
    validate_model_name,
    validate_preprocess_options,
)
from pathologic.data.loader import (
    build_folds,
    build_holdout_split,
    load_dataset,
    summarize_folds,
    summarize_holdout_split,
    validate_schema,
)
from pathologic.data.preprocessor import (
    FoldPreprocessor,
    ImputeName,
    MissingValuePolicy,
    ScalerName,
)
from pathologic.engine import Evaluator, Trainer, TrainerConfig, Tuner
from pathologic.explain import ExplainabilityService, ExplainabilityVisualizer
from pathologic.models import create_model, get_model_metadata
from pathologic.models.builder import ModelBuilder
from pathologic.models.hybrid import normalize_model_alias, parse_hybrid_alias
from pathologic.models.zoo.mlp import extract_mlp_preprocess_hints
from pathologic.utils.colorstr import info_text, success_text
from pathologic.utils.hardware import detect_preferred_device
from pathologic.utils.logger import get_logger
from pathologic.utils.progress import is_progress_enabled, step_progress


class PathoLogic:
    """High-level framework API.

    Parameters
    ----------
    model_name:
        Registered model alias. In Phase 1 this validates against a fixed list.
    """

    def __init__(
        self,
        model_name: str,
        *,
        runtime_model_config: dict[str, Any] | None = None,
    ) -> None:
        normalized_model_name = model_name.strip().lower()
        self._validate_model_name(normalized_model_name)

        self.model_name = normalized_model_name
        self._runtime_model_config = dict(runtime_model_config or {})
        self.defaults = self._load_defaults()
        self._apply_ui_runtime_config(self.defaults.get("ui"))
        self._logger = get_logger("pathologic.core")
        self.device = detect_preferred_device()
        self.is_trained = False
        self.last_train_source: str | None = None
        self.last_split_summary: list[dict[str, Any]] = []
        self.last_train_metrics: dict[str, float] = {}
        self.last_tune_result: dict[str, Any] | None = None
        self.last_eval_report: dict[str, Any] | None = None
        self.last_explain_report: dict[str, Any] | None = None
        self.last_fine_tune_report: dict[str, Any] | None = None
        self._trained_model: Any | None = None
        self._feature_columns: list[str] = []
        self._preprocessor: FoldPreprocessor | None = None
        self._explain_background: np.ndarray | None = None

    @classmethod
    def builder(cls) -> ModelBuilder:
        """Create a fluent model builder for runtime hybrid definitions."""
        return ModelBuilder()

    @classmethod
    def from_builder(cls, builder: ModelBuilder) -> PathoLogic:
        """Instantiate PathoLogic from ModelBuilder without extra config files."""
        spec = builder.build()
        return cls(spec.alias, runtime_model_config=spec.to_model_config())

    def train(self, data: str, **overrides: Unpack[TrainOverrideKwargs]) -> PathoLogic:
        """Mark the model as trained in Phase 1 skeleton mode.

        Parameters
        ----------
        data:
            Path-like identifier to the training dataset.
        """
        if not data.strip():
            raise ValueError("Training data path must be a non-empty string.")
        self._logger.info(
            info_text(
                f"[train] model={self.model_name} source={data} device={self.device}"
            )
        )

        dataset = load_dataset(data)
        data_defaults = self.defaults.get("data")
        train_defaults = self.defaults.get("train")
        data_config = data_defaults if isinstance(data_defaults, dict) else {}
        train_config = train_defaults if isinstance(train_defaults, dict) else {}

        data_override_keys = {
            "label_column",
            "gene_column",
            "required_features",
            "excluded_columns",
            "feature_routing",
        }
        train_override_keys = {
            "epochs",
            "batch_size",
            "validation_split",
            "mixed_precision",
            "split",
            "preprocess",
            "early_stopping",
            "class_imbalance",
            "optimizer",
            "scheduler",
            "gpu_ids",
            "ddp",
        }
        extra_override_keys = {"learning_rate", "weight_decay", "model_params"}
        allowed_override_keys = data_override_keys | train_override_keys | extra_override_keys
        self._validate_override_keys(
            overrides=overrides,
            allowed_keys=allowed_override_keys,
            method_name="train",
        )

        data_overrides = {
            key: value for key, value in overrides.items() if key in data_override_keys
        }
        train_overrides = {
            key: value for key, value in overrides.items() if key in train_override_keys
        }

        data_config = merge_config_overrides(
            base_config=data_config,
            overrides=data_overrides,
            allowed_keys=data_override_keys,
            context_name="train(data)",
        )
        train_config = merge_config_overrides(
            base_config=train_config,
            overrides=train_overrides,
            allowed_keys=train_override_keys,
            context_name="train(config)",
            deep_merge_keys={
                "split",
                "preprocess",
                "early_stopping",
                "class_imbalance",
                "optimizer",
                "scheduler",
                "ddp",
            },
        )
        self._apply_ui_runtime_config(train_config.get("ui"))

        if "learning_rate" in overrides or "weight_decay" in overrides:
            optimizer_cfg = train_config.get("optimizer")
            merged_optimizer = dict(optimizer_cfg) if isinstance(optimizer_cfg, dict) else {}
            if "learning_rate" in overrides:
                merged_optimizer["lr"] = float(overrides["learning_rate"])
            if "weight_decay" in overrides:
                merged_optimizer["weight_decay"] = float(overrides["weight_decay"])
            train_config["optimizer"] = merged_optimizer

        split_config = self._resolved_split_config(train_config=train_config)
        preprocess_config = self._resolved_preprocess_config(train_config=train_config)

        label_column = str(data_config.get("label_column", "label"))
        gene_column = str(data_config.get("gene_column", "gene_id"))
        required_features = self._resolve_required_features(data_config)
        active_features, member_feature_map = self._resolve_feature_routing(
            data_config=data_config,
            required_features=required_features,
        )

        preprocess_config = self._normalize_preprocess_for_dataset(
            preprocess_config,
            dataset_columns=dataset.columns,
            gene_column=gene_column,
        )
        preprocess_config = self._apply_tabnet_missingness_policy(
            preprocess_config=preprocess_config,
            active_features=active_features,
        )

        validate_schema(
            dataset,
            label_column=label_column,
            gene_column=gene_column,
            require_gene_column=bool(preprocess_config.get("per_gene", False)),
            required_feature_columns=active_features,
        )

        split_mode = str(split_config.get("mode", "cross_validation")).strip().lower()
        if split_mode in {"cv", "cross_validation"}:
            cv_defaults = split_config.get("cross_validation")
            cv_config = cv_defaults if isinstance(cv_defaults, dict) else {}
            n_splits_value = cv_config.get("n_splits", split_config.get("n_splits", 3))
            if n_splits_value is None:
                n_splits_value = 3
            stratified_value = cv_config.get("stratified", True)
            if stratified_value is None:
                stratified_value = True
            folds = build_folds(
                dataset,
                label_column=label_column,
                gene_column=gene_column,
                n_splits=int(n_splits_value),
                stratified=bool(stratified_value),
                random_state=int(self.defaults.get("seed", 42)),
            )
            self.last_split_summary = summarize_folds(
                dataset,
                folds,
                label_column=label_column,
                gene_column=gene_column,
            )
        elif split_mode == "holdout":
            holdout_defaults = split_config.get("holdout")
            holdout_config = holdout_defaults if isinstance(holdout_defaults, dict) else {}
            test_size_value = holdout_config.get("test_size", split_config.get("test_size", 0.2))
            if test_size_value is None:
                test_size_value = 0.2
            val_size_value = holdout_config.get("val_size", split_config.get("val_size", 0.2))
            if val_size_value is None:
                val_size_value = 0.2
            holdout_stratified = holdout_config.get("stratified", True)
            if holdout_stratified is None:
                holdout_stratified = True
            split_indices = build_holdout_split(
                dataset,
                label_column=label_column,
                gene_column=gene_column,
                test_size=float(test_size_value),
                val_size=float(val_size_value),
                stratified=bool(holdout_stratified),
                random_state=int(self.defaults.get("seed", 42)),
            )
            self.last_split_summary = [
                summarize_holdout_split(
                    dataset,
                    split_indices,
                    label_column=label_column,
                    gene_column=gene_column,
                )
            ]
        else:
            raise ValueError(
                "Config field 'split.mode' must be one of: cross_validation, holdout"
            )

        preprocess_config = dict(preprocess_config)
        model_params = self._model_params_from_defaults()
        model_params_override = overrides.get("model_params")
        if isinstance(model_params_override, dict):
            model_params = {**model_params, **model_params_override}

        if self.model_name == "mlp":
            if "epochs" in overrides:
                model_params["max_epochs"] = int(overrides["epochs"])
            if "batch_size" in overrides:
                model_params["batch_size"] = int(overrides["batch_size"])
            if "learning_rate" in overrides:
                model_params["learning_rate_init"] = float(overrides["learning_rate"])
            if "weight_decay" in overrides:
                model_params["weight_decay"] = float(overrides["weight_decay"])
        elif self.model_name in {"xgboost", "catboost", "tabnet"}:
            if "learning_rate" in overrides:
                model_params["learning_rate"] = float(overrides["learning_rate"])
            if "weight_decay" in overrides and self.model_name == "tabnet":
                model_params["weight_decay"] = float(overrides["weight_decay"])

        if self.model_name == "mlp":
            self._apply_mlp_train_fallbacks(model_params, train_config)
            architecture_path_raw = model_params.get("architecture_path")
            if isinstance(architecture_path_raw, str) and architecture_path_raw.strip():
                hints = extract_mlp_preprocess_hints(architecture_path_raw)
                if "scaler" in hints:
                    preprocess_config["scaler"] = hints["scaler"]
                if "per_gene" in hints:
                    preprocess_config["per_gene"] = hints["per_gene"]
                if "per_gene_features" in hints:
                    preprocess_config["per_gene_features"] = hints["per_gene_features"]
                if "scaler_features" in hints:
                    preprocess_config["scaler_features"] = hints["scaler_features"]

        (
            missing_value_policy_raw,
            impute_strategy_raw,
            scaler_raw,
        ) = self._validate_preprocess_options(preprocess_config)

        processor = FoldPreprocessor(
            numeric_features=active_features,
            gene_column=gene_column,
            missing_value_policy=cast(MissingValuePolicy, missing_value_policy_raw),
            impute_strategy=cast(ImputeName, impute_strategy_raw),
            scaler=cast(ScalerName, scaler_raw),
            per_gene=bool(preprocess_config.get("per_gene", True)),
            per_gene_features=(
                [str(value) for value in preprocess_config.get("per_gene_features", [])]
                if isinstance(preprocess_config.get("per_gene_features"), list)
                else None
            ),
            scaler_features=(
                [str(value) for value in preprocess_config.get("scaler_features", [])]
                if isinstance(preprocess_config.get("scaler_features"), list)
                else None
            ),
            add_missing_indicators=bool(preprocess_config.get("add_missing_indicators", False)),
            missing_indicator_features=(
                [str(value) for value in preprocess_config.get("missing_indicator_features", [])]
                if isinstance(preprocess_config.get("missing_indicator_features"), list)
                else None
            ),
        )
        processed = processor.fit_transform(dataset)
        training_feature_columns = self._augment_feature_columns_with_missing_indicators(
            base_features=active_features,
            processor=processor,
        )
        x = processed[training_feature_columns].to_numpy(dtype=float)
        y = processed[label_column].to_numpy(dtype=int)

        model_params = self._with_runtime_model_params(
            model_params=model_params,
            y=y,
            early_stopping_config=train_config.get("early_stopping"),
            class_imbalance_config=train_config.get("class_imbalance"),
        )

        model_params = self._with_device_model_params(model_params)
        if member_feature_map is not None and "+" in self.model_name:
            model_params["member_feature_map"] = {
                key: list(value) for key, value in member_feature_map.items()
            }
            model_params["feature_names"] = list(active_features)

        model = create_model(
            self.model_name,
            random_state=int(self.defaults.get("seed", 42)),
            model_params=model_params,
        )

        val_fraction = float(train_config.get("validation_split", 0.0))
        x_train = x
        y_train = y
        x_val: np.ndarray | None = None
        y_val: np.ndarray | None = None
        if 0.0 < val_fraction < 1.0 and len(x) > 4:
            stratify_target: np.ndarray | None = None
            if np.unique(y).size > 1:
                stratify_target = y
            x_train, x_val, y_train, y_val = train_test_split(
                x,
                y,
                test_size=val_fraction,
                random_state=int(self.defaults.get("seed", 42)),
                stratify=stratify_target,
            )

        trainer = Trainer(self._trainer_config_from_defaults(train_config))
        train_result = trainer.fit(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
        )

        self._trained_model = train_result.model
        self._preprocessor = processor
        self._feature_columns = training_feature_columns
        self._explain_background = np.asarray(x_train, dtype=float)
        self.last_train_metrics = dict(train_result.metrics)
        self.is_trained = True
        self.last_train_source = data
        self._logger.info(success_text("[train] completed"))
        return self

    def evaluate(
        self,
        data: str,
        **overrides: Unpack[EvaluateOverrideKwargs],
    ) -> dict[str, Any]:
        """Evaluate trained model on labeled dataset and return metric report."""
        self._logger.info(
            info_text(
                f"[evaluate] model={self.model_name} source={data} device={self.device}"
            )
        )
        if not self.is_trained:
            raise RuntimeError("Call train(...) before evaluate(...).")
        if self._trained_model is None:
            raise RuntimeError("No trained model instance found. Call train(...) again.")
        if self._preprocessor is None:
            raise RuntimeError("No preprocessing state found. Call train(...) again.")

        dataset = load_dataset(data)
        data_defaults = self.defaults.get("data")
        data_config = data_defaults if isinstance(data_defaults, dict) else {}
        label_column = str(data_config.get("label_column", "label"))
        gene_column = str(data_config.get("gene_column", "gene_id"))

        if label_column not in dataset.columns:
            raise ValueError(f"Evaluation dataset must contain label column '{label_column}'.")
        missing = [column for column in self._feature_columns if column not in dataset.columns]
        if missing:
            raise ValueError(
                "Evaluation dataset is missing required feature columns: " + ", ".join(missing)
            )

        transformed = self._preprocessor.transform(dataset)
        x = transformed[self._feature_columns].to_numpy(dtype=float)
        y_true = transformed[label_column].to_numpy(dtype=int)

        probabilities = np.asarray(self._trained_model.predict_proba(x))
        if probabilities.ndim == 1:
            y_score = probabilities
        else:
            y_score = probabilities[:, -1]

        test_defaults = self.defaults.get("test")
        test_config = test_defaults if isinstance(test_defaults, dict) else {}
        eval_override_keys = {
            "threshold",
            "group_column",
            "metrics",
            "top_k_hotspots",
            "batch_size",
        }
        self._validate_override_keys(
            overrides=overrides,
            allowed_keys=eval_override_keys,
            method_name="evaluate",
        )
        config_overrides = {
            key: value
            for key, value in overrides.items()
            if key in {"threshold", "metrics", "top_k_hotspots", "batch_size"}
        }
        test_config = merge_config_overrides(
            base_config=test_config,
            overrides=config_overrides,
            allowed_keys={"threshold", "metrics", "top_k_hotspots", "batch_size"},
            context_name="evaluate(config)",
        )

        resolved_threshold = float(test_config.get("threshold", 0.5))
        y_pred = (y_score >= resolved_threshold).astype(int)

        metrics_raw = test_config.get("metrics")
        metric_names = (
            [str(item) for item in metrics_raw]
            if isinstance(metrics_raw, list)
            else ["roc_auc", "auprc", "f1", "mcc", "precision", "recall"]
        )

        evaluator = Evaluator(metric_names=metric_names)
        group_column_raw = overrides.get("group_column")
        resolved_group_column = (
            str(group_column_raw) if group_column_raw is not None else gene_column
        )
        groups: np.ndarray | None = None
        if resolved_group_column in transformed.columns:
            groups = transformed[resolved_group_column].astype(str).to_numpy()

        report = evaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            group_values=groups,
            group_name=resolved_group_column,
            top_k_hotspots=int(test_config.get("top_k_hotspots", 10)),
        )
        self.last_eval_report = report.to_dict()
        self._logger.info(success_text("[evaluate] completed"))
        return report.to_dict()

    def tune(self, data: str, **overrides: Unpack[TuneOverrideKwargs]) -> dict[str, Any]:
        """Tune model hyperparameters from config-defined search space."""
        self._logger.info(
            info_text(
                f"[tune] model={self.model_name} source={data} device={self.device}"
            )
        )
        dataset = load_dataset(data)

        data_defaults = self.defaults.get("data")
        data_config = data_defaults if isinstance(data_defaults, dict) else {}

        label_column = str(data_config.get("label_column", "label"))
        gene_column = str(data_config.get("gene_column", "gene_id"))
        required_features = self._resolve_required_features(data_config)
        active_features, member_feature_map = self._resolve_feature_routing(
            data_config=data_config,
            required_features=required_features,
        )

        tune_defaults = self.defaults.get("tune")
        tune_config = tune_defaults if isinstance(tune_defaults, dict) else {}
        tune_override_keys = {
            "engine",
            "n_trials",
            "max_trials",
            "objective",
            "timeout_minutes",
            "early_stopping",
            "class_imbalance",
            "callbacks",
            "split",
            "preprocess",
        }
        self._validate_override_keys(
            overrides=overrides,
            allowed_keys=tune_override_keys,
            method_name="tune",
        )
        tune_config = merge_config_overrides(
            base_config=tune_config,
            overrides=overrides,
            allowed_keys=tune_override_keys,
            context_name="tune(config)",
            deep_merge_keys={"early_stopping", "class_imbalance", "split", "preprocess"},
        )
        train_defaults = self.defaults.get("train")
        train_config = train_defaults if isinstance(train_defaults, dict) else {}
        split_config = self._resolved_split_config(
            train_config=train_config,
            tune_config=tune_config,
        )
        preprocess_config = self._resolved_preprocess_config(
            train_config=train_config,
            tune_config=tune_config,
        )
        preprocess_config = self._normalize_preprocess_for_dataset(
            preprocess_config,
            dataset_columns=dataset.columns,
            gene_column=gene_column,
        )
        preprocess_config = self._apply_tabnet_missingness_policy(
            preprocess_config=preprocess_config,
            active_features=active_features,
        )
        (
            missing_value_policy_raw,
            impute_strategy_raw,
            scaler_raw,
        ) = self._validate_preprocess_options(preprocess_config)

        validate_schema(
            dataset,
            label_column=label_column,
            gene_column=gene_column,
            require_gene_column=bool(preprocess_config.get("per_gene", False)),
            required_feature_columns=active_features,
        )

        model_config = self._resolve_model_config()

        search_space_raw = model_config.get("tuning_search_space")
        if not isinstance(search_space_raw, dict) or not search_space_raw:
            raise ValueError(
                "Config field 'models."
                f"{self.model_name}.tuning_search_space' must be a non-empty mapping."
            )

        split_mode = str(split_config.get("mode", "cross_validation")).strip().lower()
        if split_mode in {"cv", "cross_validation"}:
            cv_defaults = split_config.get("cross_validation")
            cv_config = cv_defaults if isinstance(cv_defaults, dict) else {}
            n_splits_value = cv_config.get("n_splits", split_config.get("n_splits", 3))
            if n_splits_value is None:
                n_splits_value = 3
            stratified_value = cv_config.get("stratified", True)
            if stratified_value is None:
                stratified_value = True
            folds = build_folds(
                dataset,
                label_column=label_column,
                gene_column=gene_column,
                n_splits=int(n_splits_value),
                stratified=bool(stratified_value),
                random_state=int(self.defaults.get("seed", 42)),
            )
        elif split_mode == "holdout":
            holdout_defaults = split_config.get("holdout")
            holdout_config = holdout_defaults if isinstance(holdout_defaults, dict) else {}
            test_size_value = holdout_config.get("test_size", split_config.get("test_size", 0.2))
            if test_size_value is None:
                test_size_value = 0.2
            val_size_value = holdout_config.get("val_size", split_config.get("val_size", 0.2))
            if val_size_value is None:
                val_size_value = 0.2
            holdout_stratified = holdout_config.get("stratified", True)
            if holdout_stratified is None:
                holdout_stratified = True
            split_indices = build_holdout_split(
                dataset,
                label_column=label_column,
                gene_column=gene_column,
                test_size=float(test_size_value),
                val_size=float(val_size_value),
                stratified=bool(holdout_stratified),
                random_state=int(self.defaults.get("seed", 42)),
            )
            train_val = np.concatenate([split_indices["train"], split_indices["val"]])
            folds = [(train_val, split_indices["test"])]
        else:
            raise ValueError("Config field 'split.mode' must be one of: cross_validation, holdout")

        objective_name = str(tune_config.get("objective", "roc_auc")).strip().lower()

        base_model_params = self._model_params_from_defaults()
        if member_feature_map is not None and "+" in self.model_name:
            base_model_params["member_feature_map"] = {
                key: list(value) for key, value in member_feature_map.items()
            }
            base_model_params["feature_names"] = list(active_features)

        def objective_fn(trial_params: dict[str, Any]) -> float:
            fold_scores: list[float] = []
            with step_progress(
                total=len(folds),
                desc="tune folds",
                enabled=is_progress_enabled(),
            ) as fold_bar:
                for train_idx, val_idx in folds:
                    train_df = dataset.iloc[train_idx]
                    val_df = dataset.iloc[val_idx]

                    processor = FoldPreprocessor(
                        numeric_features=active_features,
                        gene_column=gene_column,
                        missing_value_policy=cast(MissingValuePolicy, missing_value_policy_raw),
                        impute_strategy=cast(ImputeName, impute_strategy_raw),
                        scaler=cast(ScalerName, scaler_raw),
                        per_gene=bool(preprocess_config.get("per_gene", True)),
                        per_gene_features=(
                            [str(value) for value in preprocess_config.get("per_gene_features", [])]
                            if isinstance(preprocess_config.get("per_gene_features"), list)
                            else None
                        ),
                        scaler_features=(
                            [str(value) for value in preprocess_config.get("scaler_features", [])]
                            if isinstance(preprocess_config.get("scaler_features"), list)
                            else None
                        ),
                        add_missing_indicators=bool(
                            preprocess_config.get("add_missing_indicators", False)
                        ),
                        missing_indicator_features=(
                            [
                                str(value)
                                for value in preprocess_config.get("missing_indicator_features", [])
                            ]
                            if isinstance(preprocess_config.get("missing_indicator_features"), list)
                            else None
                        ),
                    )
                    train_processed = processor.fit_transform(train_df)
                    val_processed = processor.transform(val_df)

                    fold_feature_columns = self._augment_feature_columns_with_missing_indicators(
                        base_features=active_features,
                        processor=processor,
                    )

                    x_train = train_processed[fold_feature_columns].astype(float)
                    y_train = train_processed[label_column].to_numpy(dtype=int)
                    x_val = val_processed[fold_feature_columns].astype(float)
                    y_val = val_processed[label_column].to_numpy(dtype=int)

                    if len(x_train) == 0 or len(x_val) == 0:
                        raise ValueError(
                            "preprocess.missing_value_policy='drop_rows' removed all rows "
                            "in at least one tuning fold."
                        )

                    model_params = {**base_model_params, **trial_params}
                    model_params = self._with_runtime_model_params(
                        model_params=model_params,
                        y=y_train,
                        early_stopping_config=tune_config.get("early_stopping"),
                        class_imbalance_config=tune_config.get("class_imbalance"),
                    )
                    model_params = self._with_device_model_params(model_params)
                    model = create_model(
                        self.model_name,
                        random_state=int(self.defaults.get("seed", 42)),
                        model_params=model_params,
                    )
                    model.fit(x_train, y_train)

                    y_pred = np.asarray(model.predict(x_val)).reshape(-1)
                    y_score: np.ndarray | None = None
                    if hasattr(model, "predict_proba"):
                        proba = np.asarray(model.predict_proba(x_val))
                        y_score = proba[:, -1] if proba.ndim > 1 else proba

                    evaluator = Evaluator(metric_names=[objective_name, "f1"])
                    report = evaluator.evaluate(y_true=y_val, y_pred=y_pred, y_score=y_score)
                    score = report.metrics.get(objective_name)
                    if score is None or np.isnan(score):
                        score = report.metrics.get("f1", 0.0)
                    fold_scores.append(float(score))
                    fold_bar.update(1)
                    fold_bar.set_postfix(score=f"{float(score):.4f}")

            return float(np.mean(fold_scores))

        tuner = Tuner(
            engine=str(tune_config.get("engine", "random")),
            random_state=int(self.defaults.get("seed", 42)),
        )
        configured_trials = int(tune_config.get("n_trials", 20))
        budget_max_trials = int(tune_config.get("max_trials", configured_trials))
        effective_trials = min(configured_trials, budget_max_trials)

        tune_early_raw = tune_config.get("early_stopping")
        tune_early_stopping = tune_early_raw if isinstance(tune_early_raw, dict) else {}
        tune_callbacks_raw = tune_config.get("callbacks")
        tune_callbacks = (
            [item for item in tune_callbacks_raw if callable(item)]
            if isinstance(tune_callbacks_raw, list)
            else None
        )

        result = tuner.tune(
            objective=objective_fn,
            search_space=cast(dict[str, dict[str, Any]], search_space_raw),
            n_trials=effective_trials,
            timeout_seconds=float(tune_config.get("timeout_minutes", 60)) * 60.0,
            direction="maximize",
            callbacks=tune_callbacks,
            early_stopping=tune_early_stopping,
        )

        result_dict = {
            "engine": result.engine,
            "best_params": dict(result.best_params),
            "best_score": float(result.best_score),
            "trials": [dict(item) for item in result.trials],
        }
        self.last_tune_result = result_dict
        self._logger.info(success_text("[tune] completed"))
        return result_dict

    def explain(
        self,
        data: str,
        **overrides: Unpack[ExplainOverrideKwargs],
    ) -> dict[str, Any]:
        """Generate explainability report with attribution and grouped FP analysis."""
        self._logger.info(info_text(f"[explain] model={self.model_name} source={data}"))
        if not self.is_trained:
            raise RuntimeError("Call train(...) before explain(...).")
        if self._trained_model is None:
            raise RuntimeError("No trained model instance found. Call train(...) again.")
        if self._preprocessor is None:
            raise RuntimeError("No preprocessing state found. Call train(...) again.")

        dataset = load_dataset(data)
        data_defaults = self.defaults.get("data")
        data_config = data_defaults if isinstance(data_defaults, dict) else {}
        label_column = str(data_config.get("label_column", "label"))
        if label_column not in dataset.columns:
            raise ValueError(f"Explain dataset must contain label column '{label_column}'.")

        missing = [column for column in self._feature_columns if column not in dataset.columns]
        if missing:
            raise ValueError(
                "Explain dataset is missing required feature columns: " + ", ".join(missing)
            )

        transformed = self._preprocessor.transform(dataset)
        x = transformed[self._feature_columns].to_numpy(dtype=float)
        y_true = transformed[label_column].to_numpy(dtype=int)

        probabilities = np.asarray(self._trained_model.predict_proba(x))
        y_score = probabilities if probabilities.ndim == 1 else probabilities[:, -1]

        explain_config = self._resolved_explain_config()
        explain_override_keys = {
            "threshold",
            "backend",
            "background_size",
            "top_k_features",
            "top_k_samples",
            "group_columns",
            "biological_mapping",
            "false_positive",
            "visual_report",
        }
        self._validate_override_keys(
            overrides=overrides,
            allowed_keys=explain_override_keys,
            method_name="explain",
        )
        explain_config = merge_config_overrides(
            base_config=explain_config,
            overrides={key: value for key, value in overrides.items() if key != "threshold"},
            allowed_keys=explain_override_keys - {"threshold"},
            context_name="explain(config)",
            deep_merge_keys={"false_positive", "visual_report"},
        )
        test_defaults = self.defaults.get("test")
        test_config = test_defaults if isinstance(test_defaults, dict) else {}
        threshold_override = overrides.get("threshold")
        resolved_threshold = (
            float(threshold_override)
            if threshold_override is not None
            else float(test_config.get("threshold", 0.5))
        )
        y_pred = (y_score >= resolved_threshold).astype(int)

        explain_service = ExplainabilityService(
            config=explain_config,
            seed=int(self.defaults.get("seed", 42)),
        )
        background_matrix = (
            np.asarray(self._explain_background, dtype=float)
            if self._explain_background is not None
            else x
        )
        report = explain_service.build_report(
            model=self._trained_model,
            feature_names=list(self._feature_columns),
            x_background=background_matrix,
            x_target=x,
            y_score=y_score,
            y_pred=y_pred,
            y_true=y_true,
            dataset=transformed,
        )
        payload = report.to_dict()
        payload["metadata"]["model_name"] = self.model_name
        payload["metadata"]["background_source"] = (
            "train_split" if self._explain_background is not None else "input_dataset"
        )
        payload["metadata"]["device"] = self.device

        visual_config_raw = explain_config.get("visual_report")
        visual_config = visual_config_raw if isinstance(visual_config_raw, dict) else {}
        if bool(visual_config.get("enabled", True)):
            visualizer = ExplainabilityVisualizer()
            html_report = visualizer.render_html(report)
            payload["visual_report_html"] = html_report

            save_path = visual_config.get("save_path")
            if isinstance(save_path, str) and save_path.strip():
                output_path = Path(save_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(html_report, encoding="utf-8")
                payload["metadata"]["visual_report_path"] = str(output_path)

        self.last_explain_report = payload
        self._logger.info(success_text("[explain] completed"))
        return payload

    def fine_tune(
        self,
        data: str,
        **overrides: Unpack[FineTuneOverrideKwargs],
    ) -> dict[str, Any]:
        """Fine-tune the trained model on a domain-specific dataset."""
        self._logger.info(info_text(f"[fine_tune] model={self.model_name} source={data}"))
        if not self.is_trained:
            raise RuntimeError("Call train(...) before fine_tune(...).")
        if self._trained_model is None:
            raise RuntimeError("No trained model instance found. Call train(...) again.")
        if self._preprocessor is None:
            raise RuntimeError("No preprocessing state found. Call train(...) again.")

        fine_tune_config = self._resolved_finetune_config()
        fine_tune_override_keys = {
            "seed",
            "epochs",
            "batch_size",
            "learning_rate",
            "freeze_layers",
            "validation_split",
            "scheduler",
            "metric_delta",
        }
        self._validate_override_keys(
            overrides=overrides,
            allowed_keys=fine_tune_override_keys,
            method_name="fine_tune",
        )
        fine_tune_config = merge_config_overrides(
            base_config=fine_tune_config,
            overrides=overrides,
            allowed_keys=fine_tune_override_keys,
            context_name="fine_tune(config)",
            deep_merge_keys={"scheduler", "metric_delta"},
        )

        effective_freeze = str(fine_tune_config.get("freeze_layers", "none"))
        effective_lr = float(fine_tune_config.get("learning_rate", 0.0005))
        effective_epochs = int(fine_tune_config.get("epochs", 10))
        scheduler_config_raw = fine_tune_config.get("scheduler")
        scheduler_config = (
            dict(scheduler_config_raw)
            if isinstance(scheduler_config_raw, dict)
            else None
        )

        metadata = get_model_metadata(self.model_name) if "+" not in self.model_name else None
        if metadata is not None and not metadata.supports_layer_freezing:
            if effective_freeze.strip().lower() not in {"none", "", "all_trainable"}:
                raise ValueError(
                    "Selected model does not support layer freezing in fine_tune: "
                    f"{self.model_name}"
                )

        before_report = self.evaluate(data)

        dataset = load_dataset(data)
        data_defaults = self.defaults.get("data")
        data_config = data_defaults if isinstance(data_defaults, dict) else {}
        label_column = str(data_config.get("label_column", "label"))

        if label_column not in dataset.columns:
            raise ValueError(f"Fine-tune dataset must contain label column '{label_column}'.")

        missing = [column for column in self._feature_columns if column not in dataset.columns]
        if missing:
            raise ValueError(
                "Fine-tune dataset is missing required feature columns: " + ", ".join(missing)
            )

        transformed = self._preprocessor.transform(dataset)
        x = transformed[self._feature_columns].to_numpy(dtype=float)
        y = transformed[label_column].to_numpy(dtype=int)

        model_obj = self._trained_model
        if hasattr(model_obj, "fine_tune"):
            try:
                model_obj.fine_tune(
                    x,
                    y,
                    freeze_layers=effective_freeze,
                    learning_rate=effective_lr,
                    epochs=effective_epochs,
                    scheduler_config=scheduler_config,
                )
            except TypeError:
                model_obj.fine_tune(x, y)
        else:
            model_obj.fit(x, y)

        self._trained_model = model_obj
        self._explain_background = np.asarray(x, dtype=float)

        after_report = self.evaluate(data)

        metric_delta_cfg_raw = fine_tune_config.get("metric_delta")
        metric_delta_cfg = metric_delta_cfg_raw if isinstance(metric_delta_cfg_raw, dict) else {}
        delta_metrics_raw = metric_delta_cfg.get("metrics", ["roc_auc", "f1"])
        delta_metrics = (
            [str(item) for item in delta_metrics_raw]
            if isinstance(delta_metrics_raw, list)
            else ["roc_auc", "f1"]
        )

        before_metrics = before_report.get("metrics", {})
        after_metrics = after_report.get("metrics", {})
        metric_delta: dict[str, float] = {}
        if isinstance(before_metrics, dict) and isinstance(after_metrics, dict):
            for metric_name in delta_metrics:
                before_value = before_metrics.get(metric_name)
                after_value = after_metrics.get(metric_name)
                if before_value is None or after_value is None:
                    continue
                metric_delta[metric_name] = float(after_value) - float(before_value)

        report = {
            "model_name": self.model_name,
            "freeze_layers": effective_freeze,
            "learning_rate": effective_lr,
            "epochs": effective_epochs,
            "before": before_report,
            "after": after_report,
            "metric_delta": metric_delta,
        }
        if "+" in self.model_name:
            report["hybrid_policy"] = "all_members_common"
        self.last_fine_tune_report = report
        self._logger.info(success_text("[fine_tune] completed"))
        return report

    def predict(
        self,
        data: str,
        **overrides: Unpack[PredictOverrideKwargs],
    ) -> list[dict[str, Any]]:
        """Run inference and return prediction rows with probabilities."""
        self._logger.info(info_text(f"[predict] model={self.model_name} source={data}"))
        if not self.is_trained:
            raise RuntimeError("Call train(...) before predict(...).")
        if self._trained_model is None:
            raise RuntimeError("No trained model instance found. Call train(...) again.")
        if self._preprocessor is None:
            raise RuntimeError("No preprocessing state found. Call train(...) again.")
        if not data.strip():
            raise ValueError("Prediction data path must be a non-empty string.")

        dataset = load_dataset(data)
        missing = [column for column in self._feature_columns if column not in dataset.columns]
        if missing:
            raise ValueError(
                "Prediction dataset is missing required feature columns: "
                + ", ".join(missing)
            )

        self._validate_override_keys(
            overrides=overrides,
            allowed_keys={"threshold"},
            method_name="predict",
        )

        transformed = self._preprocessor.transform(dataset)
        x = transformed[self._feature_columns].to_numpy(dtype=float)
        probabilities = np.asarray(self._trained_model.predict_proba(x))
        if probabilities.ndim == 1:
            scores = probabilities
        else:
            scores = probabilities[:, -1]

        threshold_override = overrides.get("threshold")
        if threshold_override is None:
            labels = np.asarray(self._trained_model.predict(x)).reshape(-1)
        else:
            labels = (scores >= float(threshold_override)).astype(int)

        rows: list[dict[str, Any]] = []
        for idx, (label, score) in enumerate(zip(labels, scores, strict=True)):
            rows.append(
                {
                    "source": data,
                    "row_index": idx,
                    "predicted_label": str(int(label)),
                    "score": float(score),
                    "model_name": self.model_name,
                    "device": self.device,
                }
            )
        self._logger.info(success_text("[predict] completed"))
        return rows

    @staticmethod
    def _validate_override_keys(
        *,
        overrides: dict[str, Any],
        allowed_keys: set[str],
        method_name: str,
    ) -> None:
        unknown = sorted(set(overrides) - allowed_keys)
        if unknown:
            raise ValueError(
                f"Unsupported {method_name}() override keys: "
                + ", ".join(unknown)
                + "."
            )

    @staticmethod
    def _apply_ui_runtime_config(ui_config_raw: Any) -> None:
        """Apply UI config values as process env flags for shared progress/color utils."""
        if not isinstance(ui_config_raw, dict):
            return

        mapping = {
            "colored_output": "PATHOLOGIC_COLORED_OUTPUT",
            "show_progress": "PATHOLOGIC_SHOW_PROGRESS",
            "show_batch_progress": "PATHOLOGIC_SHOW_BATCH_PROGRESS",
        }
        for config_key, env_name in mapping.items():
            value = ui_config_raw.get(config_key)
            if isinstance(value, bool):
                os.environ[env_name] = "1" if value else "0"

    @staticmethod
    def _load_defaults() -> dict[str, Any]:
        return load_defaults(config_dir=Path(__file__).parent / "configs")

    @staticmethod
    def _load_config_from_path(path_value: str) -> dict[str, Any]:
        """Load YAML config from path relative to package config directory."""
        return load_config_from_path(
            config_dir=Path(__file__).parent / "configs",
            path_value=path_value,
        )

    def _resolve_model_config(self) -> dict[str, Any]:
        """Resolve model config from inline mapping or file path reference."""
        return resolve_model_config(
            defaults=self.defaults,
            model_name=self.model_name,
            runtime_model_config=self._runtime_model_config,
            config_dir=Path(__file__).parent / "configs",
        )

    def _model_params_from_defaults(self) -> dict[str, Any]:
        """Load model-specific params for the selected alias from defaults."""
        return model_params_from_resolved_config(
            model_name=self.model_name,
            model_config=self._resolve_model_config(),
        )

    @staticmethod
    def _mlp_constructor_keys() -> set[str]:
        """Allowed constructor keys for MLP wrapper when reading model block."""
        return mlp_constructor_keys()

    @staticmethod
    def _apply_mlp_train_fallbacks(
        model_params: dict[str, Any],
        train_config: dict[str, Any],
    ) -> None:
        """Apply train-level fallback params to MLP config without overriding model-level values."""
        apply_mlp_train_fallbacks(model_params=model_params, train_config=train_config)

    def _trainer_config_from_defaults(self, train_config: dict[str, Any]) -> TrainerConfig:
        """Build trainer config from defaults while preserving compatibility."""
        ddp_raw = train_config.get("ddp")
        ddp_config = ddp_raw if isinstance(ddp_raw, dict) else {}
        gpu_ids_raw = train_config.get("gpu_ids", ddp_config.get("gpu_ids"))
        gpu_ids = (
            [int(value) for value in gpu_ids_raw]
            if isinstance(gpu_ids_raw, list)
            else None
        )

        requested_device = str(train_config.get("device", self.defaults.get("device", "auto")))
        if requested_device == "auto":
            requested_device = self.device

        return TrainerConfig(
            device=requested_device,
            mixed_precision=bool(
                train_config.get(
                    "mixed_precision",
                    self.defaults.get("hardware", {}).get("mixed_precision", False),
                )
            ),
            ddp_enabled=bool(ddp_config.get("enabled", False)),
            ddp_backend=str(ddp_config.get("backend", "nccl")),
            rank=int(ddp_config.get("rank", 0)),
            world_size=int(ddp_config.get("world_size", 1)),
            gpu_ids=gpu_ids,
        )

    @staticmethod
    def _validate_model_name(model_name: str) -> None:
        """Validate model alias against registry or plus-composed hybrid syntax."""
        validate_model_name(model_name)

    def _resolved_split_config(
        self,
        *,
        train_config: dict[str, Any],
        tune_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve split config with precedence: tune.split > train.split > defaults.split."""
        return resolve_split_config(
            defaults=self.defaults,
            train_config=train_config,
            tune_config=tune_config,
        )

    def _resolved_preprocess_config(
        self,
        *,
        train_config: dict[str, Any],
        tune_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve preprocess config with precedence.

        Priority: tune.preprocess > train.preprocess > defaults.preprocess.
        """
        return resolve_preprocess_config(
            defaults=self.defaults,
            train_config=train_config,
            tune_config=tune_config,
        )

    def _resolved_explain_config(self) -> dict[str, Any]:
        """Resolve explain runtime config from defaults."""
        return resolve_explain_config(defaults=self.defaults)

    def _resolved_finetune_config(self) -> dict[str, Any]:
        """Resolve fine-tune runtime config from defaults."""
        return resolve_finetune_config(defaults=self.defaults)

    @staticmethod
    def _resolve_required_features(data_config: dict[str, Any]) -> list[str]:
        """Resolve required features after applying explicit exclusion list."""
        required_features_raw = data_config.get("required_features", [])
        if not isinstance(required_features_raw, list):
            raise ValueError("Config field 'data.required_features' must be a list.")
        required_features = [str(feature) for feature in required_features_raw]
        if not required_features:
            raise ValueError("Config field 'data.required_features' must not be empty.")

        excluded_columns_raw = data_config.get("excluded_columns", [])
        if excluded_columns_raw is None:
            excluded_columns: list[str] = []
        elif isinstance(excluded_columns_raw, list):
            excluded_columns = [str(column) for column in excluded_columns_raw]
        else:
            raise ValueError("Config field 'data.excluded_columns' must be a list.")

        excluded_set = {column for column in excluded_columns if column}
        filtered_features = [
            feature for feature in required_features if feature not in excluded_set
        ]
        if not filtered_features:
            raise ValueError(
                "All data.required_features were filtered out by data.excluded_columns."
            )
        return filtered_features

    def _resolve_feature_routing(
        self,
        *,
        data_config: dict[str, Any],
        required_features: list[str],
    ) -> tuple[list[str], dict[str, list[str]] | None]:
        """Resolve optional feature routing for single and hybrid model modes."""
        feature_routing_raw = data_config.get("feature_routing")
        if not isinstance(feature_routing_raw, dict):
            return required_features, None

        required_set = set(required_features)
        single_raw = feature_routing_raw.get("single")
        single = single_raw if isinstance(single_raw, dict) else {}
        hybrid_raw = feature_routing_raw.get("hybrid")
        hybrid = hybrid_raw if isinstance(hybrid_raw, dict) else {}

        if "+" not in self.model_name:
            single_key = normalize_model_alias(self.model_name)
            selected_raw = single.get(single_key)
            if selected_raw is None:
                return required_features, None
            if not isinstance(selected_raw, list) or not selected_raw:
                raise ValueError(
                    "Config field 'data.feature_routing.single."
                    f"{single_key}' must be a non-empty list."
                )
            selected = [str(feature) for feature in selected_raw]
            unknown = [feature for feature in selected if feature not in required_set]
            if unknown:
                raise ValueError(
                    "Routed features must be subset of data.required_features. "
                    f"Unknown: {', '.join(unknown)}"
                )
            return list(dict.fromkeys(selected)), None

        members = parse_hybrid_alias(self.model_name)
        canonical_alias = "+".join(members)
        sorted_alias = "+".join(sorted(members))
        routing_raw = (
            hybrid.get(self.model_name)
            or hybrid.get(canonical_alias)
            or hybrid.get(sorted_alias)
        )
        if routing_raw is None:
            return required_features, None
        if not isinstance(routing_raw, dict):
            raise ValueError(
                "Config field 'data.feature_routing.hybrid.<alias>' must be a mapping."
            )

        member_map_raw = routing_raw.get("members")
        member_map_source = member_map_raw if isinstance(member_map_raw, dict) else routing_raw
        if not isinstance(member_map_source, dict):
            raise ValueError(
                "Config field 'data.feature_routing.hybrid.<alias>.members' "
                "must be a mapping."
            )

        normalized_member_map: dict[str, list[str]] = {}
        for member_alias in members:
            candidate_keys = {member_alias, normalize_model_alias(member_alias)}
            selected_raw: Any = None
            for key in candidate_keys:
                if key in member_map_source:
                    selected_raw = member_map_source[key]
                    break

            if not isinstance(selected_raw, list) or not selected_raw:
                raise ValueError(
                    "Hybrid member routing must define a non-empty feature list for "
                    f"'{member_alias}'."
                )

            selected = [str(feature) for feature in selected_raw]
            unknown = [feature for feature in selected if feature not in required_set]
            if unknown:
                raise ValueError(
                    "Routed features must be subset of data.required_features. "
                    f"Unknown: {', '.join(unknown)}"
                )
            normalized_member_map[member_alias] = list(dict.fromkeys(selected))

        union_set = {
            feature for values in normalized_member_map.values() for feature in values
        }
        active_features = [feature for feature in required_features if feature in union_set]
        if not active_features:
            raise ValueError("Hybrid feature routing produced an empty active feature set.")
        return active_features, normalized_member_map

    @staticmethod
    def _normalize_preprocess_for_dataset(
        preprocess_config: dict[str, Any],
        *,
        dataset_columns: Any,
        gene_column: str,
    ) -> dict[str, Any]:
        """Normalize preprocess config for available dataset schema."""
        return normalize_preprocess_for_dataset(
            preprocess_config=preprocess_config,
            dataset_columns=dataset_columns,
            gene_column=gene_column,
        )

    @staticmethod
    def _validate_preprocess_options(
        preprocess_config: dict[str, Any],
    ) -> tuple[str, str, str]:
        """Validate preprocess options and return normalized strategy names."""
        return validate_preprocess_options(preprocess_config)

    def _apply_tabnet_missingness_policy(
        self,
        *,
        preprocess_config: dict[str, Any],
        active_features: list[str],
    ) -> dict[str, Any]:
        """Apply optional TabNet-specific missingness defaults.

        TabNet does not reliably handle NaN features. In auto mode we enforce
        imputation and add configured missing-indicator features.
        """
        resolved = dict(preprocess_config)
        if self.model_name != "tabnet":
            return resolved

        mode = str(resolved.get("tabnet_missingness_mode", "auto")).strip().lower()
        if mode == "off":
            return resolved

        configured_indicator_raw = resolved.get(
            "tabnet_missing_indicator_features",
            ["feature__GERP_Score", "GERP_Score"],
        )
        configured_indicator = (
            [str(value) for value in configured_indicator_raw if str(value)]
            if isinstance(configured_indicator_raw, list)
            else []
        )
        available = set(active_features)
        preferred_indicator_features = [
            feature
            for feature in configured_indicator
            if feature in available
        ]

        if mode == "manual":
            if (
                bool(resolved.get("add_missing_indicators", False))
                and "missing_indicator_features" not in resolved
                and preferred_indicator_features
            ):
                resolved["missing_indicator_features"] = preferred_indicator_features
            return resolved

        # mode=auto
        resolved["missing_value_policy"] = "impute"
        auto_impute = str(
            resolved.get(
                "tabnet_impute_strategy",
                resolved.get("impute_strategy", "median"),
            )
        ).strip().lower()
        resolved["impute_strategy"] = auto_impute
        resolved["add_missing_indicators"] = True

        existing_indicator_raw = resolved.get("missing_indicator_features")
        existing_indicator = (
            [str(value) for value in existing_indicator_raw if str(value)]
            if isinstance(existing_indicator_raw, list)
            else []
        )
        merged_indicator = list(dict.fromkeys([*existing_indicator, *preferred_indicator_features]))
        resolved["missing_indicator_features"] = [
            feature
            for feature in merged_indicator
            if feature in available
        ]
        return resolved

    @staticmethod
    def _augment_feature_columns_with_missing_indicators(
        *,
        base_features: list[str],
        processor: FoldPreprocessor,
    ) -> list[str]:
        """Append generated missing-indicator columns to model feature list."""
        augmented = list(base_features)
        for feature in processor.resolved_missing_indicator_features:
            if feature not in augmented:
                augmented.append(feature)
        return augmented

    def _with_device_model_params(self, model_params: dict[str, Any]) -> dict[str, Any]:
        """Attach GPU-oriented model params when CUDA is available and supported."""
        merged = dict(model_params)
        if self.device != "cuda":
            # LightGBM GPU backend is OpenCL-based and can work even when torch CUDA detector is false.
            if "+" in self.model_name:
                for alias in parse_hybrid_alias(self.model_name):
                    if alias == "lightgbm":
                        merged.setdefault(f"member__{alias}__device", "gpu")
            elif self.model_name == "lightgbm":
                merged.setdefault("device", "gpu")
            return merged

        if "+" in self.model_name:
            for alias in parse_hybrid_alias(self.model_name):
                if alias == "xgboost":
                    merged.setdefault(f"member__{alias}__device", "cuda")
                    merged.setdefault(f"member__{alias}__tree_method", "hist")
                elif alias == "lightgbm":
                    merged.setdefault(f"member__{alias}__device", "gpu")
                elif alias == "catboost":
                    merged.setdefault(f"member__{alias}__task_type", "GPU")
                elif alias == "tabnet":
                    merged.setdefault(f"member__{alias}__device_name", "cuda")
            return merged

        if self.model_name == "xgboost":
            merged.setdefault("device", "cuda")
            merged.setdefault("tree_method", "hist")
        elif self.model_name == "lightgbm":
            merged.setdefault("device", "gpu")
        elif self.model_name == "catboost":
            merged.setdefault("task_type", "GPU")
        elif self.model_name == "tabnet":
            merged.setdefault("device_name", "cuda")

        return merged

    def _with_runtime_model_params(
        self,
        *,
        model_params: dict[str, Any],
        y: np.ndarray,
        early_stopping_config: Any,
        class_imbalance_config: Any,
    ) -> dict[str, Any]:
        """Attach runtime model params derived from train/tune config."""
        merged = dict(model_params)
        self._apply_early_stopping_model_params(merged, early_stopping_config)
        self._apply_class_imbalance_model_params(merged, y, class_imbalance_config)
        return merged

    def _apply_early_stopping_model_params(
        self,
        model_params: dict[str, Any],
        early_stopping_config: Any,
    ) -> None:
        if not isinstance(early_stopping_config, dict):
            return

        if not bool(early_stopping_config.get("enabled", False)):
            return

        if "+" in self.model_name:
            for alias in parse_hybrid_alias(self.model_name):
                if alias in {"xgboost", "catboost", "tabnet", "mlp"}:
                    model_params.setdefault(
                        f"member__{alias}__early_stopping",
                        dict(early_stopping_config),
                    )
            return

        if self.model_name in {"xgboost", "catboost", "tabnet", "mlp"}:
            model_params.setdefault("early_stopping", dict(early_stopping_config))

    def _apply_class_imbalance_model_params(
        self,
        model_params: dict[str, Any],
        y: np.ndarray,
        class_imbalance_config: Any,
    ) -> None:
        if not isinstance(class_imbalance_config, dict):
            return

        if not bool(class_imbalance_config.get("enabled", False)):
            return

        mode = str(class_imbalance_config.get("mode", "balanced")).strip().lower()
        if mode != "balanced":
            return

        explicit_weight = class_imbalance_config.get("positive_class_weight")
        if explicit_weight is not None:
            scale_pos_weight = float(explicit_weight)
        else:
            scale_pos_weight = self._compute_scale_pos_weight(y)

        if "+" in self.model_name:
            for alias in parse_hybrid_alias(self.model_name):
                if alias == "xgboost":
                    model_params.setdefault(f"member__{alias}__scale_pos_weight", scale_pos_weight)
                elif alias in {"catboost", "logreg", "random_forest", "tabnet"}:
                    model_params.setdefault(f"member__{alias}__class_weight", "balanced")
            return

        if self.model_name == "xgboost":
            model_params.setdefault("scale_pos_weight", scale_pos_weight)
        elif self.model_name in {"catboost", "logreg", "random_forest", "tabnet"}:
            model_params.setdefault("class_weight", "balanced")

    @staticmethod
    def _compute_scale_pos_weight(y: np.ndarray) -> float:
        """Compute neg/pos ratio for binary imbalance handling with safe clipping."""
        y_flat = np.asarray(y).reshape(-1)
        positive = float(np.sum(y_flat == 1))
        negative = float(np.sum(y_flat == 0))
        if positive <= 0.0:
            return 1.0
        ratio = negative / positive
        return float(np.clip(ratio, 0.1, 50.0))
