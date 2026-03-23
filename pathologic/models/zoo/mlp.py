"""Torch-based MLP wrapper with dynamic architecture and preprocess hints."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pathologic.models.registry import register
from pathologic.utils.hardware import detect_preferred_device
from pathologic.utils.logger import get_logger
from pathologic.utils.progress import epoch_progress, step_progress


def _resolve_architecture_path(architecture_path: str) -> Path:
    """Resolve architecture path from absolute path or package config directory."""
    raw = Path(architecture_path)
    if raw.is_absolute() and raw.exists():
        return raw

    package_root = Path(__file__).resolve().parents[2]
    config_candidate = package_root / "configs" / "models" / "mlp.yaml"
    if config_candidate.exists():
        return config_candidate

    if raw.exists():
        return raw

    raise ValueError(
        "MLP architecture file was not found. "
        f"Tried: '{raw}' and '{config_candidate}'."
    )


def _read_architecture_config(architecture_path: str) -> dict[str, Any]:
    """Load and validate MLP architecture mapping from YAML file."""
    path = _resolve_architecture_path(architecture_path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ValueError("MLP architecture config must be a mapping.")

    model_block = loaded.get("model", loaded)
    if not isinstance(model_block, dict):
        raise ValueError("MLP architecture 'model' section must be a mapping.")
    return model_block


def _extract_layer_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract normalized layer specifications from architecture config."""
    architecture_block = config.get("architecture")
    if isinstance(architecture_block, dict):
        layers = architecture_block.get("layers", [])
    elif "layers" in config:
        layers = config.get("layers", [])
    elif "hidden_layers" in config:
        hidden_layers = config["hidden_layers"]
        if not isinstance(hidden_layers, list) or not all(
            isinstance(value, int) for value in hidden_layers
        ):
            raise ValueError("MLP architecture field 'hidden_layers' must be a list of integers.")
        return [{"type": "dense", "units": int(units)} for units in hidden_layers]
    else:
        return []

    if not isinstance(layers, list):
        raise ValueError("MLP architecture field 'layers' must be a list.")

    normalized: list[dict[str, Any]] = []
    for layer in layers:
        if isinstance(layer, int):
            normalized.append({"type": "dense", "units": int(layer)})
            continue
        if not isinstance(layer, dict):
            raise ValueError("Each MLP layer item must be a mapping.")

        layer_type_raw = layer.get("type")
        if layer_type_raw is None and "units" in layer:
            layer_type = "dense"
        elif isinstance(layer_type_raw, str):
            layer_type = layer_type_raw.strip().lower()
        else:
            raise ValueError("Each MLP layer must provide a string 'type' or integer 'units'.")

        normalized.append({**layer, "type": layer_type})

    return normalized


def extract_mlp_preprocess_hints(architecture_path: str) -> dict[str, Any]:
    """Extract preprocess hints from MLP architecture config.

    Supported hints:
    - scaler: standard
    - per_gene: true
    - per_gene_features: [...]
    - scaler_features: [...]
    """
    config = _read_architecture_config(architecture_path)
    layers = _extract_layer_specs(config)

    hints: dict[str, Any] = {}
    per_gene_features: list[str] = []
    scaler_features: list[str] = []

    for layer in layers:
        layer_type = str(layer.get("type", "")).strip().lower()
        if layer_type == "batch_norm":
            hints["scaler"] = "standard"
            features = layer.get("features")
            if isinstance(features, list):
                scaler_features.extend(str(feature) for feature in features)
        elif layer_type == "gene_batch_norm":
            hints["scaler"] = "standard"
            hints["per_gene"] = True
            features = layer.get("features")
            if not isinstance(features, list) or not features:
                raise ValueError("Layer type 'gene_batch_norm' requires non-empty 'features' list.")
            per_gene_features.extend(str(feature) for feature in features)

    if per_gene_features:
        hints["per_gene_features"] = list(dict.fromkeys(per_gene_features))

    if scaler_features:
        hints["scaler_features"] = list(dict.fromkeys(scaler_features))

    return hints


class _TorchMLPModule(nn.Module):
    """Torch module that materializes architecture from layer specs."""

    def __init__(
        self,
        *,
        input_dim: int,
        layer_specs: list[dict[str, Any]],
        default_activation: str,
    ) -> None:
        super().__init__()
        activation_map: dict[str, type[nn.Module]] = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "sigmoid": nn.Sigmoid,
            "identity": nn.Identity,
        }

        modules: list[nn.Module] = []
        current_dim = input_dim

        for layer in layer_specs:
            layer_type = str(layer.get("type", "dense")).strip().lower()
            if layer_type == "dense":
                units = layer.get("units")
                if not isinstance(units, int) or units <= 0:
                    raise ValueError("Dense layer requires positive integer 'units'.")
                modules.append(nn.Linear(current_dim, units))
                activation_name_raw = layer.get("activation", default_activation)
                activation_name = str(activation_name_raw).strip().lower()
                activation_cls = activation_map.get(activation_name)
                if activation_cls is None:
                    raise ValueError(
                        "Unsupported activation in MLP architecture: " + activation_name
                    )
                modules.append(activation_cls())
                current_dim = units
                continue

            if layer_type == "batch_norm":
                modules.append(nn.BatchNorm1d(current_dim))
                continue

            if layer_type == "dropout":
                probability = float(layer.get("p", layer.get("prob", 0.2)))
                if probability < 0.0 or probability >= 1.0:
                    raise ValueError("Dropout probability 'p' must satisfy 0 <= p < 1.")
                modules.append(nn.Dropout(probability))
                continue

            if layer_type == "gene_batch_norm":
                # gene-specific normalization is handled in preprocessing via hints.
                continue

            raise ValueError(f"Unsupported MLP architecture layer type: {layer_type}")

        self.backbone = nn.Sequential(*modules)
        self.output = nn.Linear(current_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x)
        logits = self.output(hidden)
        return cast(torch.Tensor, logits.squeeze(1))


def _build_optimizer(
    *,
    name: str,
    parameters: Any,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    normalized = name.strip().lower()
    if normalized == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if normalized == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if normalized == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    raise ValueError(f"Unsupported optimizer name: {name}")


def _build_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict[str, Any] | None,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if not scheduler_config:
        return None

    name = str(scheduler_config.get("name", "none")).strip().lower()
    if name in {"none", ""}:
        return None

    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_config.get("step_size", 10)),
            gamma=float(scheduler_config.get("gamma", 0.1)),
        )
    if name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_config.get("t_max", 10)),
            eta_min=float(scheduler_config.get("eta_min", 0.0)),
        )

    if name == "exponentiallr":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=float(scheduler_config.get("gamma", 0.99)),
        )

    if name in {"reduce_on_plateau", "reducelronplateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_config.get("mode", "min")),
            factor=float(scheduler_config.get("factor", 0.1)),
            patience=int(scheduler_config.get("patience", 10)),
            threshold=float(scheduler_config.get("threshold", 0.0001)),
            min_lr=float(scheduler_config.get("min_lr", 0.0)),
        )

    raise ValueError(f"Unsupported scheduler name: {name}")


@register(name="mlp", family="neural-network", supports_layer_freezing=True)
class MLPWrapper(ClassifierMixin, BaseEstimator):
    """Torch MLP wrapper exposing stable fit/predict/predict_proba API."""

    def __init__(
        self,
        *,
        architecture_path: str | None = None,
        hidden_layer_sizes: Sequence[int] | None = None,
        activation: str | None = None,
        solver: str | None = None,
        alpha: float | None = None,
        max_epochs: int = 30,
        batch_size: int = 64,
        learning_rate_init: float = 0.001,
        optimizer: dict[str, Any] | None = None,
        scheduler: dict[str, Any] | None = None,
        early_stopping: dict[str, Any] | None = None,
        weight_decay: float = 0.0,
        max_iter: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.architecture_path = architecture_path
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.weight_decay = weight_decay
        self.max_iter = max_iter
        self.random_state = random_state
        self._logger = get_logger("pathologic.models.zoo.mlp")

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.device = torch.device(detect_preferred_device())
        self.architecture_path = architecture_path
        self.random_state = random_state

        self._default_activation = (activation or "relu").strip().lower()
        self._max_epochs = int(max_iter if max_iter is not None else max_epochs)
        self._batch_size = int(batch_size)
        self._learning_rate = float(learning_rate_init)
        self._weight_decay = float(weight_decay)
        self._optimizer_cfg = dict(self.optimizer or {})
        self._scheduler_cfg = dict(self.scheduler or {})
        self._early_stopping_cfg = dict(self.early_stopping or {})

        if self.solver is not None and "name" not in self._optimizer_cfg:
            self._optimizer_cfg["name"] = self.solver

        if self.alpha is not None and self._weight_decay == 0.0:
            self._weight_decay = float(self.alpha)

        config_from_file: dict[str, Any] = {}
        if architecture_path:
            config_from_file = _read_architecture_config(architecture_path)

        layers = _extract_layer_specs(config_from_file)
        if not layers and hidden_layer_sizes is not None:
            layers = [{"type": "dense", "units": int(units)} for units in hidden_layer_sizes]
        if not layers:
            layers = [{"type": "dense", "units": 32}, {"type": "dense", "units": 16}]

        self.layer_specs = layers

        if "optimizer" in config_from_file and isinstance(config_from_file["optimizer"], dict):
            self._optimizer_cfg = {**config_from_file["optimizer"], **self._optimizer_cfg}
        if "scheduler" in config_from_file and isinstance(config_from_file["scheduler"], dict):
            self._scheduler_cfg = {**config_from_file["scheduler"], **self._scheduler_cfg}
        if "early_stopping" in config_from_file and isinstance(
            config_from_file["early_stopping"],
            dict,
        ):
            self._early_stopping_cfg = {
                **self._early_stopping_cfg,
                **config_from_file["early_stopping"],
            }

        if "max_epochs" in config_from_file:
            self._max_epochs = int(config_from_file["max_epochs"])
        if "batch_size" in config_from_file:
            self._batch_size = int(config_from_file["batch_size"])
        if "learning_rate_init" in config_from_file:
            self._learning_rate = float(config_from_file["learning_rate_init"])
        if "activation" in config_from_file:
            self._default_activation = str(config_from_file["activation"]).strip().lower()

        self.model: _TorchMLPModule | None = None
        self._trained_epochs: int = 0
        # Expose sklearn-style estimator field for hybrid ensemble compatibility.
        self.estimator = self
        self._logger = get_logger("pathologic.models.zoo.mlp")

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> MLPWrapper:
        return self._fit_impl(
            x,
            y,
            reset_model=True,
            freeze_layers="none",
            learning_rate_override=None,
            epochs_override=None,
            scheduler_override=None,
            external_x_val=x_val,
            external_y_val=y_val,
        )

    def fine_tune(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        freeze_layers: str = "backbone_last2",
        learning_rate: float | None = None,
        epochs: int | None = None,
        scheduler_config: dict[str, Any] | None = None,
    ) -> MLPWrapper:
        return self._fit_impl(
            x,
            y,
            reset_model=False,
            freeze_layers=freeze_layers,
            learning_rate_override=learning_rate,
            epochs_override=epochs,
            scheduler_override=scheduler_config,
            external_x_val=None,
            external_y_val=None,
        )

    def _fit_impl(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        reset_model: bool,
        freeze_layers: str,
        learning_rate_override: float | None,
        epochs_override: int | None,
        scheduler_override: dict[str, Any] | None,
        external_x_val: np.ndarray | None = None,
        external_y_val: np.ndarray | None = None,
    ) -> MLPWrapper:
        x_array = np.asarray(x, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32).reshape(-1)
        self.classes_ = np.unique(y_array.astype(int))

        if x_array.ndim != 2:
            raise ValueError("Input feature matrix must be 2-dimensional.")

        input_dim = int(x_array.shape[1])
        if reset_model or self.model is None:
            self.model = _TorchMLPModule(
                input_dim=input_dim,
                layer_specs=self.layer_specs,
                default_activation=self._default_activation,
            ).to(self.device)

        if self.model is None:
            raise RuntimeError("Model initialization failed.")

        self._apply_freeze_strategy(freeze_layers)

        optimizer_name = str(self._optimizer_cfg.get("name", "adam"))
        base_learning_rate = float(self._optimizer_cfg.get("lr", self._learning_rate))
        learning_rate = (
            float(learning_rate_override)
            if learning_rate_override is not None
            else base_learning_rate
        )
        weight_decay = float(self._optimizer_cfg.get("weight_decay", self._weight_decay))
        max_epochs = int(epochs_override) if epochs_override is not None else self._max_epochs

        scheduler_config = (
            dict(scheduler_override)
            if scheduler_override is not None
            else dict(self._scheduler_cfg)
        )

        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        if not trainable_params:
            raise ValueError("Fine-tune left no trainable parameters after freeze strategy.")

        optimizer = _build_optimizer(
            name=optimizer_name,
            parameters=trainable_params,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = _build_scheduler(optimizer=optimizer, scheduler_config=scheduler_config)
        criterion = nn.BCEWithLogitsLoss()

        early_stopping_enabled = bool(self._early_stopping_cfg.get("enabled", False))
        patience = int(self._early_stopping_cfg.get("patience", 10))
        min_delta = float(self._early_stopping_cfg.get("min_delta", 0.0))
        validation_split = float(self._early_stopping_cfg.get("validation_split", 0.2))
        restore_best_weights = bool(self._early_stopping_cfg.get("restore_best_weights", True))

        train_x = x_array
        train_y = y_array
        val_tensor_x: torch.Tensor | None = None
        val_tensor_y: torch.Tensor | None = None
        if (
            early_stopping_enabled
            and external_x_val is not None
            and external_y_val is not None
            and len(external_x_val) > 0
        ):
            val_x_array = np.asarray(external_x_val, dtype=np.float32)
            val_y_array = np.asarray(external_y_val, dtype=np.float32).reshape(-1)
            val_tensor_x = torch.from_numpy(val_x_array).to(self.device)
            val_tensor_y = torch.from_numpy(val_y_array).to(self.device)

        if (
            early_stopping_enabled
            and val_tensor_x is None
            and val_tensor_y is None
            and 0.0 < validation_split < 1.0
            and len(x_array) > 4
        ):
            indices = np.arange(len(x_array))
            stratify_values: np.ndarray | None = None
            if np.unique(y_array).shape[0] > 1:
                stratify_values = y_array
            try:
                train_idx, val_idx = train_test_split(
                    indices,
                    test_size=validation_split,
                    random_state=self.random_state,
                    stratify=stratify_values,
                )
            except ValueError:
                train_idx, val_idx = train_test_split(
                    indices,
                    test_size=validation_split,
                    random_state=self.random_state,
                    shuffle=True,
                )

            if len(train_idx) > 0 and len(val_idx) > 0:
                train_x = x_array[train_idx]
                train_y = y_array[train_idx]
                val_tensor_x = torch.from_numpy(x_array[val_idx]).to(self.device)
                val_tensor_y = torch.from_numpy(y_array[val_idx]).to(self.device)

        dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state_dict: dict[str, torch.Tensor] | None = None

        self.model.train()
        self._logger.info(
            "Starting MLP training: epochs=%d batch_size=%d freeze=%s",
            max_epochs,
            self._batch_size,
            freeze_layers,
        )
        with epoch_progress(total=max_epochs, desc="MLP epochs") as epoch_bar:
            for epoch in range(max_epochs):
                epoch_loss_total = 0.0
                epoch_batches = 0
                with step_progress(
                    total=len(loader),
                    desc=f"epoch {epoch + 1}/{max_epochs}",
                ) as batch_bar:
                    for batch_x, batch_y in loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        optimizer.zero_grad()
                        logits = self.model(batch_x)
                        loss = criterion(logits, batch_y)
                        loss.backward()
                        optimizer.step()

                        loss_value = float(loss.item())
                        epoch_loss_total += loss_value
                        epoch_batches += 1
                        batch_bar.update(1)
                        batch_bar.set_postfix(loss=f"{loss_value:.4f}")

                train_loss = epoch_loss_total / max(epoch_batches, 1)
                self._trained_epochs = epoch + 1

                current_val_loss: float | None = None
                if (
                    early_stopping_enabled
                    and val_tensor_x is not None
                    and val_tensor_y is not None
                ):
                    self.model.eval()
                    with torch.no_grad():
                        val_logits = self.model(val_tensor_x)
                        current_val_loss = float(criterion(val_logits, val_tensor_y).item())
                    self.model.train()

                    if (best_val_loss - current_val_loss) > min_delta:
                        best_val_loss = current_val_loss
                        epochs_without_improvement = 0
                        if restore_best_weights:
                            best_state_dict = {
                                key: value.detach().cpu().clone()
                                for key, value in self.model.state_dict().items()
                            }
                    else:
                        epochs_without_improvement += 1

                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if current_val_loss is not None:
                            scheduler.step(current_val_loss)
                    else:
                        scheduler.step()

                if current_val_loss is None:
                    epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss="-")
                else:
                    epoch_bar.set_postfix(
                        train_loss=f"{train_loss:.4f}",
                        val_loss=f"{current_val_loss:.4f}",
                    )
                epoch_bar.update(1)

                self._logger.info(
                    "Epoch %d/%d train_loss=%.6f val_loss=%s",
                    epoch + 1,
                    max_epochs,
                    train_loss,
                    "-" if current_val_loss is None else f"{current_val_loss:.6f}",
                )

                if early_stopping_enabled and epochs_without_improvement >= patience:
                    self._logger.info(
                        "Stopping early at epoch %d/%d after %d non-improving epochs",
                        epoch + 1,
                        max_epochs,
                        epochs_without_improvement,
                    )
                    break

        if best_state_dict is not None and self.model is not None:
            self.model.load_state_dict(best_state_dict)

        return self

    def _apply_freeze_strategy(self, freeze_layers: str) -> None:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        strategy = freeze_layers.strip().lower()
        if strategy in {"none", "", "all_trainable"}:
            self.model.requires_grad_(True)
            return

        if strategy in {"backbone_last1", "last1"}:
            self._freeze_all_but_last_n(1)
            return

        if strategy in {"backbone_last2", "last2"}:
            self._freeze_all_but_last_n(2)
            return

        raise ValueError(
            "Unsupported freeze strategy for MLPWrapper: "
            f"{freeze_layers}. Supported: none, backbone_last1, backbone_last2"
        )

    def _freeze_all_but_last_n(self, n_last_layers: int) -> None:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        self.model.requires_grad_(False)
        linear_layers: list[nn.Linear] = []
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                linear_layers.append(module)

        if not linear_layers:
            raise ValueError("No linear layers found for freeze strategy.")

        for layer in linear_layers[-max(1, n_last_layers):]:
            layer.requires_grad_(True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(x)[:, 1]
        return (probabilities >= 0.5).astype(int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit(...) first.")

        x_array = np.asarray(x, dtype=np.float32)
        x_tensor = torch.from_numpy(x_array).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs_pos = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

        probs = np.column_stack([1.0 - probs_pos, probs_pos])
        return probs
