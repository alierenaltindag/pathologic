"""Unit tests for fallback warning logs in optional model wrappers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import pathologic.models.zoo.catboost_model as catboost_module
import pathologic.models.zoo.tabnet as tabnet_module
import pathologic.models.zoo.xgboost_model as xgboost_module
from pathologic.models.zoo.catboost_model import CatBoostWrapper
from pathologic.models.zoo.tabnet import TabNetWrapper
from pathologic.models.zoo.xgboost_model import XGBoostWrapper


@pytest.mark.parametrize(
    ("module_obj", "wrapper_cls"),
    [
        (xgboost_module, XGBoostWrapper),
        (catboost_module, CatBoostWrapper),
        (tabnet_module, TabNetWrapper),
    ],
)
def test_wrapper_logs_warning_when_optional_backend_missing(
    module_obj: object,
    wrapper_cls: type[object],
) -> None:
    with (
        patch.object(module_obj.importlib, "import_module", side_effect=ImportError("missing")),
        patch.object(module_obj._LOGGER, "warning") as warning_mock,
    ):
        wrapper = wrapper_cls(random_state=42)

    assert wrapper._using_fallback is True
    warning_mock.assert_called_once()
