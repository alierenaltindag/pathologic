"""Compute-cost profiling helpers for candidate-level reporting."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import os
import platform
import subprocess
from threading import Event, Thread
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


_CPU_MODEL_CACHE: str | None = None


@dataclass(frozen=True)
class InferenceLatency:
    """Latency summary for practical inference execution."""

    single_sample_ms: float
    batch_total_ms: float
    batch_per_sample_ms: float
    batch_size: int
    full_dataset_ms: float
    full_dataset_size: int


class ProcessMemoryMonitor:
    """Track process RSS usage and peak during a monitored stage."""

    def __init__(self, *, sample_interval_seconds: float = 0.05) -> None:
        self.sample_interval_seconds = max(float(sample_interval_seconds), 0.01)
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._process: Any | None = None
        self._psutil: Any | None = None
        self._start_rss_bytes: int | None = None
        self._peak_rss_bytes: int | None = None
        self._system_used_start_bytes: int | None = None
        self._status = "unavailable"
        self._reason: str | None = None

        try:
            psutil = import_module("psutil")
            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
            self._status = "ok"
        except Exception as exc:
            self._reason = str(exc)

    @property
    def is_available(self) -> bool:
        return self._status == "ok" and self._process is not None

    def start(self) -> None:
        if not self.is_available:
            return
        assert self._process is not None
        process_mem = self._process.memory_info()
        self._start_rss_bytes = int(process_mem.rss)
        self._peak_rss_bytes = int(process_mem.rss)
        if self._psutil is not None:
            vm = self._psutil.virtual_memory()
            self._system_used_start_bytes = int(vm.total - vm.available)

        self._stop_event.clear()
        self._thread = Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()

    def _sampling_loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            if not self.is_available:
                continue
            assert self._process is not None
            try:
                rss = int(self._process.memory_info().rss)
            except Exception:
                continue
            if self._peak_rss_bytes is None or rss > self._peak_rss_bytes:
                self._peak_rss_bytes = rss

    def stop(self) -> dict[str, Any]:
        if not self.is_available:
            return {
                "status": "unavailable",
                "reason": self._reason or "psutil_unavailable",
            }

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        assert self._process is not None
        end_rss = int(self._process.memory_info().rss)
        peak_rss = int(self._peak_rss_bytes if self._peak_rss_bytes is not None else end_rss)
        start_rss = int(self._start_rss_bytes if self._start_rss_bytes is not None else end_rss)

        payload: dict[str, Any] = {
            "status": "ok",
            "sample_interval_seconds": float(self.sample_interval_seconds),
            "process_rss_start_mb": float(start_rss / (1024 * 1024)),
            "process_rss_end_mb": float(end_rss / (1024 * 1024)),
            "process_rss_delta_mb": float((end_rss - start_rss) / (1024 * 1024)),
            "process_rss_peak_mb": float(peak_rss / (1024 * 1024)),
            "process_rss_peak_delta_mb": float((peak_rss - start_rss) / (1024 * 1024)),
        }

        if self._psutil is not None:
            vm = self._psutil.virtual_memory()
            used_end = int(vm.total - vm.available)
            used_start = int(
                self._system_used_start_bytes
                if self._system_used_start_bytes is not None
                else used_end
            )
            payload["system_used_start_mb"] = float(used_start / (1024 * 1024))
            payload["system_used_end_mb"] = float(used_end / (1024 * 1024))
            payload["system_used_delta_mb"] = float((used_end - used_start) / (1024 * 1024))

        return payload


def _safe_module_version(module_name: str) -> str | None:
    try:
        module = import_module(module_name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


def _read_psutil_memory() -> dict[str, Any]:
    try:
        psutil = import_module("psutil")
        process = psutil.Process(os.getpid())
        process_mem = process.memory_info()
        vm = psutil.virtual_memory()
        return {
            "status": "ok",
            "process_rss_mb": float(process_mem.rss / (1024 * 1024)),
            "process_vms_mb": float(process_mem.vms / (1024 * 1024)),
            "system_total_mb": float(vm.total / (1024 * 1024)),
            "system_available_mb": float(vm.available / (1024 * 1024)),
            "system_used_mb": float((vm.total - vm.available) / (1024 * 1024)),
            "system_percent": float(vm.percent),
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": str(exc),
        }


def _detect_cpu_model() -> str:
    global _CPU_MODEL_CACHE
    if isinstance(_CPU_MODEL_CACHE, str) and _CPU_MODEL_CACHE:
        return _CPU_MODEL_CACHE

    def _looks_generic(value: str) -> bool:
        lowered = value.lower()
        return lowered.startswith("amd64 family") or lowered.startswith("x86 family")

    processor = (platform.processor() or "").strip()
    if (
        processor
        and processor.lower() not in {"unknown", "amd64", "x86_64"}
        and not _looks_generic(processor)
    ):
        _CPU_MODEL_CACHE = processor
        return _CPU_MODEL_CACHE

    if os.name == "nt":
        try:
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
            ]
            completed = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=3)
            candidate = (completed.stdout or "").strip()
            if candidate and not _looks_generic(candidate):
                _CPU_MODEL_CACHE = candidate
                return _CPU_MODEL_CACHE
        except Exception:
            pass

        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if isinstance(value, str) and value.strip():
                    candidate = value.strip()
                    if not _looks_generic(candidate):
                        _CPU_MODEL_CACHE = candidate
                        return _CPU_MODEL_CACHE
        except Exception:
            pass

    identifier = (os.environ.get("PROCESSOR_IDENTIFIER") or "").strip()
    if identifier and not _looks_generic(identifier):
        _CPU_MODEL_CACHE = identifier
        return _CPU_MODEL_CACHE

    if os.path.exists("/proc/cpuinfo"):
        try:
            for line in open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore"):
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                if key.strip().lower() == "model name" and value.strip():
                    _CPU_MODEL_CACHE = value.strip()
                    return _CPU_MODEL_CACHE
        except Exception:
            pass

    uname_processor = (platform.uname().processor or "").strip()
    if uname_processor:
        _CPU_MODEL_CACHE = uname_processor
        return _CPU_MODEL_CACHE
    _CPU_MODEL_CACHE = processor or "unknown_cpu_model"
    return _CPU_MODEL_CACHE


def collect_system_info() -> dict[str, Any]:
    """Collect system and hardware metadata for reporting."""
    physical_cores: int | None = None
    try:
        psutil = import_module("psutil")
        value = psutil.cpu_count(logical=False)
        if isinstance(value, int):
            physical_cores = int(value)
    except Exception:
        physical_cores = None

    info: dict[str, Any] = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "cpu": {
            "model": _detect_cpu_model(),
            "logical_cores": int(os.cpu_count() or 0),
            "physical_cores": physical_cores,
        },
        "ram": _read_psutil_memory(),
    }

    gpu_info = collect_gpu_memory_snapshot()
    info["gpu"] = {
        "status": str(gpu_info.get("status", "unavailable")),
        "device": gpu_info.get("device"),
        "device_name": gpu_info.get("device_name"),
        "vram_total_mb": gpu_info.get("vram_total_mb"),
        "reason": gpu_info.get("reason"),
    }
    return info


def collect_framework_versions() -> dict[str, str]:
    """Collect framework versions for reproducibility and transparency."""
    module_map = {
        "pathologic": "pathologic",
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit_learn": "sklearn",
        "torch": "torch",
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
        "shap": "shap",
    }
    versions: dict[str, str] = {}
    for key, module_name in module_map.items():
        value = _safe_module_version(module_name)
        if value is not None:
            versions[key] = value
    return versions


def collect_gpu_memory_snapshot() -> dict[str, Any]:
    """Capture current GPU memory and VRAM metadata when CUDA is available."""
    try:
        torch = import_module("torch")
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"torch_import_error: {exc}",
        }

    if not bool(getattr(torch, "cuda", None)) or not torch.cuda.is_available():
        return {
            "status": "unavailable",
            "reason": "cuda_unavailable",
        }

    try:
        device_index = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(device_index)
        allocated_mb = float(torch.cuda.memory_allocated(device_index) / (1024 * 1024))
        reserved_mb = float(torch.cuda.memory_reserved(device_index) / (1024 * 1024))
        peak_allocated_mb = float(torch.cuda.max_memory_allocated(device_index) / (1024 * 1024))
        return {
            "status": "ok",
            "device": f"cuda:{device_index}",
            "device_name": str(props.name),
            "vram_total_mb": float(props.total_memory / (1024 * 1024)),
            "vram_allocated_mb": allocated_mb,
            "vram_reserved_mb": reserved_mb,
            "vram_peak_allocated_mb": peak_allocated_mb,
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": str(exc),
        }


def reset_gpu_peak_memory_stats() -> None:
    """Reset CUDA peak memory counters when available."""
    try:
        torch = import_module("torch")
        if bool(getattr(torch, "cuda", None)) and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
    except Exception:
        return


def collect_reproducibility_settings(*, seed: int, model: Any) -> dict[str, Any]:
    """Capture seed and deterministic execution knobs."""
    payload: dict[str, Any] = {
        "seed": int(seed),
        "device": str(getattr(model, "device", "unknown")),
    }

    try:
        torch = import_module("torch")
        payload["torch_deterministic_algorithms"] = bool(
            torch.are_deterministic_algorithms_enabled()
        )
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            payload["torch_cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
            payload["torch_cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    except Exception:
        payload["torch_deterministic_algorithms"] = None

    hardware_cfg = model.defaults.get("hardware") if isinstance(model.defaults, dict) else None
    if isinstance(hardware_cfg, dict):
        payload["mixed_precision"] = bool(hardware_cfg.get("mixed_precision", False))
    return payload


def resolve_batch_size(*, model: Any, selected_params: dict[str, Any]) -> int | None:
    """Resolve effective batch size from selected params or train defaults."""
    if "batch_size" in selected_params:
        try:
            return int(selected_params["batch_size"])
        except Exception:
            return None

    train_cfg = model.defaults.get("train") if isinstance(model.defaults, dict) else None
    if isinstance(train_cfg, dict) and "batch_size" in train_cfg:
        try:
            return int(train_cfg["batch_size"])
        except Exception:
            return None
    return None


def extract_iteration_metadata(*, model: Any, train_seconds: float) -> dict[str, Any]:
    """Extract iteration/epoch metadata and per-iteration timing when available."""
    trained_model = getattr(model, "_trained_model", None)
    estimator = getattr(trained_model, "estimator", trained_model)

    iteration_count: int | None = None
    for attr in ("best_iteration_", "n_estimators_", "n_iter_", "tree_count_"):
        value = getattr(estimator, attr, None)
        if isinstance(value, (int, np.integer)) and int(value) > 0:
            iteration_count = int(value)
            break

    if iteration_count is None and hasattr(estimator, "get_best_iteration"):
        try:
            value = estimator.get_best_iteration()
            if isinstance(value, (int, np.integer)) and int(value) > 0:
                iteration_count = int(value)
        except Exception:
            iteration_count = None

    payload: dict[str, Any] = {
        "train_total_seconds": float(train_seconds),
        "iteration_count": iteration_count,
        "iteration_seconds": None,
    }
    if isinstance(iteration_count, int) and iteration_count > 0:
        payload["iteration_seconds"] = float(train_seconds / iteration_count)
    return payload


def _benchmark_callable_ms(*, fn: Any, runs: int) -> float:
    if runs <= 0:
        return 0.0
    start = perf_counter()
    for _ in range(runs):
        fn()
    return float((perf_counter() - start) * 1000.0 / runs)


def _predict_proba_robust(*, model: Any, x_values: pd.DataFrame) -> np.ndarray:
    try:
        return np.asarray(model.predict_proba(x_values))
    except Exception:
        return np.asarray(model.predict_proba(x_values.to_numpy(dtype=float)))


def benchmark_inference_latency(
    *,
    model: Any,
    dataset: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    single_runs: int,
    batch_runs: int,
    warmup_runs: int,
    batch_size: int,
) -> InferenceLatency:
    """Measure single-sample and batch inference latencies."""
    preprocessor = getattr(model, "_preprocessor", None)
    trained_model = getattr(model, "_trained_model", None)
    if preprocessor is None or trained_model is None:
        raise RuntimeError("Model must be trained before inference benchmarking.")

    transformed = preprocessor.transform(dataset)
    model_feature_columns = getattr(model, "_feature_columns", None)
    resolved_feature_columns = (
        list(model_feature_columns) if model_feature_columns else list(feature_columns)
    )
    x_values = transformed[resolved_feature_columns].astype(float)

    if len(x_values) == 0:
        raise ValueError("Inference benchmark dataset is empty.")

    single_x = x_values.iloc[:1]
    effective_batch_size = max(1, min(int(batch_size), int(len(x_values))))
    batch_x = x_values.iloc[:effective_batch_size]

    def _predict_single() -> np.ndarray:
        return _predict_proba_robust(model=trained_model, x_values=single_x)

    def _predict_batch() -> np.ndarray:
        return _predict_proba_robust(model=trained_model, x_values=batch_x)

    def _predict_full() -> np.ndarray:
        return _predict_proba_robust(model=trained_model, x_values=x_values)

    for _ in range(max(int(warmup_runs), 0)):
        _predict_single()
        _predict_batch()

    single_sample_ms = _benchmark_callable_ms(fn=_predict_single, runs=max(int(single_runs), 1))
    batch_total_ms = _benchmark_callable_ms(fn=_predict_batch, runs=max(int(batch_runs), 1))
    full_dataset_ms = _benchmark_callable_ms(fn=_predict_full, runs=1)

    return InferenceLatency(
        single_sample_ms=single_sample_ms,
        batch_total_ms=batch_total_ms,
        batch_per_sample_ms=float(batch_total_ms / effective_batch_size),
        batch_size=effective_batch_size,
        full_dataset_ms=full_dataset_ms,
        full_dataset_size=int(len(x_values)),
    )


def create_process_memory_monitor(*, sample_interval_seconds: float = 0.05) -> ProcessMemoryMonitor:
    """Factory for process memory monitoring during long-running stages."""
    return ProcessMemoryMonitor(sample_interval_seconds=sample_interval_seconds)
