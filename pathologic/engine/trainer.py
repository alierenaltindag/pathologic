"""Trainer engine with device fallback, optional AMP, and optional DDP setup."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from pathologic.engine.evaluator import Evaluator
from pathologic.utils.hardware import detect_preferred_device
from pathologic.utils.logger import get_logger
from pathologic.utils.progress import epoch_progress, step_progress


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for trainer runtime behavior."""

    device: str = "auto"
    mixed_precision: bool = False
    ddp_enabled: bool = False
    ddp_backend: str = "nccl"
    rank: int = 0
    world_size: int = 1
    gpu_ids: list[int] | None = None


@dataclass(frozen=True)
class TrainerResult:
    """Result payload after training."""

    model: Any
    metrics: dict[str, float]
    device: str


class Trainer:
    """Orchestrate training and validation on top of model wrappers."""

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self.logger = get_logger("pathologic.engine.trainer")
        self._configure_gpu_visibility()
        self.device = self._select_device(config.device)

    def fit(
        self,
        *,
        model: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> TrainerResult:
        """Fit model and optionally evaluate validation predictions."""
        model.fit(x_train, y_train)
        metrics: dict[str, float] = {}
        if x_val is not None and y_val is not None:
            y_pred = np.asarray(model.predict(x_val)).reshape(-1)
            y_score: np.ndarray | None = None
            if hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(x_val))
                if proba.ndim == 1:
                    y_score = proba
                else:
                    y_score = proba[:, -1]
            evaluator = Evaluator(
                metric_names=["f1", "mcc", "precision", "recall", "roc_auc", "auprc"]
            )
            report = evaluator.evaluate(y_true=np.asarray(y_val), y_pred=y_pred, y_score=y_score)
            metrics = report.metrics

        return TrainerResult(model=model, metrics=metrics, device=self.device)

    def train_torch_module(
        self,
        *,
        model: nn.Module,
        train_loader: DataLoader[Any],
        optimizer: Any,
        loss_fn: Any,
        epochs: int,
        scheduler: Any | None = None,
        val_loader: DataLoader[Any] | None = None,
    ) -> dict[str, float]:
        """Train a native torch module with optional AMP and DDP wrapping."""
        import torch

        ddp_model = self._wrap_ddp_model(model)
        scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled())

        last_train_loss = 0.0
        last_val_loss = 0.0
        self.logger.info("Starting torch training loop: epochs=%d", epochs)
        show_progress = not self._is_distributed() or self.config.rank == 0
        with epoch_progress(total=epochs, desc="Torch epochs", enabled=show_progress) as epoch_bar:
            for epoch in range(epochs):
                if self._is_distributed() and hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)

                ddp_model.train()
                train_loss_total = 0.0
                train_batches = 0
                with step_progress(
                    total=len(train_loader),
                    desc=f"epoch {epoch + 1}/{epochs}",
                    enabled=show_progress,
                ) as batch_bar:
                    for batch in train_loader:
                        batch_x, batch_y = batch
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        optimizer.zero_grad()
                        with torch.amp.autocast("cuda", enabled=self._amp_enabled()):
                            logits = ddp_model(batch_x)
                            loss = loss_fn(logits, batch_y)

                        if self._amp_enabled():
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                        loss_value = float(loss.item())
                        train_loss_total += loss_value
                        train_batches += 1
                        batch_bar.update(1)
                        batch_bar.set_postfix(loss=f"{loss_value:.4f}")

                if scheduler is not None:
                    scheduler.step()

                last_train_loss = train_loss_total / max(train_batches, 1)

                if val_loader is not None:
                    ddp_model.eval()
                    val_loss_total = 0.0
                    val_batches = 0
                    with torch.no_grad():
                        for batch in val_loader:
                            batch_x, batch_y = batch
                            batch_x = batch_x.to(self.device)
                            batch_y = batch_y.to(self.device)
                            with torch.amp.autocast("cuda", enabled=self._amp_enabled()):
                                logits = ddp_model(batch_x)
                                loss = loss_fn(logits, batch_y)
                            val_loss_total += float(loss.item())
                            val_batches += 1
                    last_val_loss = val_loss_total / max(val_batches, 1)

                epoch_bar.set_postfix(
                    train_loss=f"{last_train_loss:.4f}",
                    val_loss=f"{last_val_loss:.4f}" if val_loader is not None else "-",
                )
                epoch_bar.update(1)
                self.logger.info(
                    "Epoch %d/%d train_loss=%.6f val_loss=%s",
                    epoch + 1,
                    epochs,
                    last_train_loss,
                    "-" if val_loader is None else f"{last_val_loss:.6f}",
                )

        return {"train_loss": last_train_loss, "val_loss": last_val_loss}

    def initialize_ddp(self) -> None:
        """Initialize process group if DDP is requested."""
        if not self._is_distributed():
            return
        import torch.distributed as dist

        if dist.is_initialized():
            return

        backend = self.config.ddp_backend
        # MAC_OPTIMIZATION: Automatically switch to 'gloo' for Apple Silicon since 
        # 'nccl' is strictly CUDA-only. This prevents crash on MacOS multi-process runs.
        if self.device != "cuda" and backend == "nccl":
            backend = "gloo"
        dist.init_process_group(
            backend=backend,
            rank=self.config.rank,
            world_size=self.config.world_size,
        )

    def finalize_ddp(self) -> None:
        """Destroy process group if initialized."""
        if not self._is_distributed():
            return
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()

    def _wrap_ddp_model(self, model: nn.Module) -> nn.Module:
        """Wrap model in DDP when enabled and process group is initialized."""
        if not self._is_distributed():
            return model.to(self.device)

        import torch
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        module = model.to(self.device)
        if not dist.is_initialized():
            return module

        if self.device == "cuda":
            current_index = torch.cuda.current_device()
            return DDP(module, device_ids=[current_index], output_device=current_index)
        return DDP(module)

    def _select_device(self, requested: str) -> str:
        # MAC_OPTIMIZATION: Safely resolves hardware requested devices ensuring 
        # MPS (Metal Performance Shaders) maps correctly if Apple Silicon is detected.
        if requested != "auto":
            preferred = requested.strip().lower()
            if preferred in {"cuda", "mps", "cpu"}:
                detected = detect_preferred_device()
                if preferred == "cuda" and detected != "cuda":
                    self.logger.warning(
                        "Requested CUDA device unavailable; falling back to %s",
                        detected,
                    )
                    return detected
                if preferred == "mps" and detected not in {"mps", "cuda"}:
                    self.logger.warning(
                        "Requested MPS device unavailable; falling back to %s",
                        detected,
                    )
                    return detected
                return preferred
            raise ValueError("Trainer device must be one of: auto, cuda, mps, cpu")
        return detect_preferred_device()

    def _configure_gpu_visibility(self) -> None:
        """Allow user-defined GPU pinning through CUDA_VISIBLE_DEVICES."""
        gpu_ids = self.config.gpu_ids
        if gpu_ids is None:
            return
        if not gpu_ids:
            return

        normalized: list[int] = []
        for value in gpu_ids:
            if int(value) < 0:
                raise ValueError("All gpu_ids must be >= 0.")
            normalized.append(int(value))

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in normalized)
        self.logger.info(
            "Pinned training to CUDA_VISIBLE_DEVICES=%s",
            os.environ["CUDA_VISIBLE_DEVICES"],
        )

    def _amp_enabled(self) -> bool:
        return self.config.mixed_precision and self.device == "cuda"

    def _is_distributed(self) -> bool:
        return self.config.ddp_enabled and self.config.world_size > 1
