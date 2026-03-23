"""Typed kwargs schemas for PathoLogic public API overrides."""

from __future__ import annotations

from typing import Any, TypedDict


class SplitCrossValidationOverrides(TypedDict, total=False):
    n_splits: int
    stratified: bool


class SplitHoldoutOverrides(TypedDict, total=False):
    test_size: float
    val_size: float
    stratified: bool


class SplitOverrides(TypedDict, total=False):
    mode: str
    n_splits: int
    stratified: bool
    test_size: float
    val_size: float
    cross_validation: SplitCrossValidationOverrides
    holdout: SplitHoldoutOverrides


class PreprocessOverrides(TypedDict, total=False):
    missing_value_policy: str
    impute_strategy: str
    scaler: str
    per_gene: bool
    on_missing_gene_column: str
    per_gene_features: list[str]
    scaler_features: list[str]
    add_missing_indicators: bool
    missing_indicator_features: list[str]
    tabnet_missingness_mode: str
    tabnet_missing_indicator_features: list[str]
    tabnet_impute_strategy: str


class EarlyStoppingOverrides(TypedDict, total=False):
    enabled: bool
    patience: int
    min_delta: float
    validation_split: float
    restore_best_weights: bool


class ClassImbalanceOverrides(TypedDict, total=False):
    enabled: bool
    mode: str
    positive_class_weight: float


class OptimizerOverrides(TypedDict, total=False):
    name: str
    lr: float
    weight_decay: float


class SchedulerOverrides(TypedDict, total=False):
    name: str
    mode: str
    factor: float
    patience: int
    threshold: float
    min_lr: float
    step_size: int
    gamma: float


class DDPOverrides(TypedDict, total=False):
    enabled: bool
    backend: str
    rank: int
    world_size: int
    gpu_ids: list[int]


class TrainOverrideKwargs(TypedDict, total=False):
    label_column: str
    gene_column: str
    required_features: list[str]
    excluded_columns: list[str]
    feature_routing: dict[str, Any]
    epochs: int
    batch_size: int
    validation_split: float
    mixed_precision: bool
    split: SplitOverrides
    preprocess: PreprocessOverrides
    early_stopping: EarlyStoppingOverrides
    class_imbalance: ClassImbalanceOverrides
    optimizer: OptimizerOverrides
    scheduler: SchedulerOverrides
    gpu_ids: list[int]
    ddp: DDPOverrides
    model_params: dict[str, Any]
    validation_data: str
    learning_rate: float
    weight_decay: float


class PredictOverrideKwargs(TypedDict, total=False):
    threshold: float


class EvaluateOverrideKwargs(TypedDict, total=False):
    threshold: float
    group_column: str
    metrics: list[str]
    top_k_hotspots: int
    batch_size: int


class TuneOverrideKwargs(TypedDict, total=False):
    engine: str
    n_trials: int
    max_trials: int
    objective: str
    timeout_minutes: float
    early_stopping: dict[str, Any]
    class_imbalance: ClassImbalanceOverrides
    split: SplitOverrides
    preprocess: PreprocessOverrides


class ExplainOverrideKwargs(TypedDict, total=False):
    threshold: float
    backend: str
    background_size: int
    top_k_features: int
    top_k_samples: int
    group_columns: list[str]
    biological_mapping: dict[str, str]
    false_positive: dict[str, Any]
    visual_report: dict[str, Any]


class FineTuneOverrideKwargs(TypedDict, total=False):
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    freeze_layers: str
    validation_split: float
    scheduler: SchedulerOverrides
    metric_delta: dict[str, Any]
