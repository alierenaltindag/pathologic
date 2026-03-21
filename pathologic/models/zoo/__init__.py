"""Model zoo registrations."""

from pathologic.models.zoo.catboost_model import CatBoostWrapper
from pathologic.models.zoo.lightgbm_model import LightGBMWrapper
from pathologic.models.zoo.mlp import MLPWrapper
from pathologic.models.zoo.sklearn_models import (
    HistGradientBoostingWrapper,
    LogisticRegressionWrapper,
    RandomForestWrapper,
)
from pathologic.models.zoo.tabnet import TabNetWrapper
from pathologic.models.zoo.xgboost_model import XGBoostWrapper

__all__ = [
    "CatBoostWrapper",
    "HistGradientBoostingWrapper",
    "LightGBMWrapper",
    "LogisticRegressionWrapper",
    "MLPWrapper",
    "RandomForestWrapper",
    "TabNetWrapper",
    "XGBoostWrapper",
]
