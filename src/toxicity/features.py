from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessConfig:
    scale_numeric: bool = True
    encode_categorical: bool = True


def build_preprocessor(X, cfg: PreprocessConfig) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns

    transformers = []
    if len(numeric_features) > 0:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if cfg.scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), numeric_features))

    if len(categorical_features) > 0:
        if cfg.encode_categorical:
            cat_steps = [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
            transformers.append(("cat", Pipeline(cat_steps), categorical_features))
        else:
            transformers.append(("cat", "passthrough", categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if transformer == "passthrough":
            feature_names.extend(cols)
        else:
            if hasattr(transformer, "get_feature_names_out"):
                names = transformer.get_feature_names_out(cols)
                feature_names.extend(names.tolist())
            else:
                feature_names.extend(cols)
    return feature_names


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, indices: Optional[List[int]] = None):
        self.indices = indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.indices is None:
            return X
        if hasattr(X, "iloc"):
            return X.iloc[:, self.indices]
        return X[:, self.indices]


def select_top_k_by_shap(shap_values: np.ndarray, k: int) -> List[int]:
    # shap_values: [n_samples, n_features] for tree models
    mean_abs = np.abs(shap_values).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=1)
    top_idx = np.argsort(mean_abs)[::-1][:k]
    return [int(i) for i in np.asarray(top_idx).ravel().tolist()]
