from typing import Dict

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def build_classifier(model_cfg: Dict):
    model_type = model_cfg.get("type", "random_forest")
    params = model_cfg.get("params", {})

    if model_type == "random_forest":
        return RandomForestClassifier(**params)
    if model_type == "hist_gb":
        return HistGradientBoostingClassifier(**params)

    raise ValueError(f"Unsupported classifier type: {model_type}")


def build_regressor(model_cfg: Dict):
    model_type = model_cfg.get("type", "random_forest")
    params = model_cfg.get("params", {})

    if model_type == "random_forest":
        return RandomForestRegressor(**params)
    if model_type == "hist_gb":
        return HistGradientBoostingRegressor(**params)

    raise ValueError(f"Unsupported regressor type: {model_type}")
