import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score

from .data import load_csv, split_features_targets, coerce_numeric
from .utils import load_config, ensure_dir


def eval_classification(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probs)),
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds)),
    }


def eval_regression(model, X, y):
    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y, preds)),
    }


def eval_split(name, df, tox_col, ld50_col, id_col, drop_cols, clf, reg):
    X, y_class, y_reg = split_features_targets(df, tox_col, ld50_col, id_col, drop_cols)
    X = coerce_numeric(X)
    y_class = pd.to_numeric(y_class, errors="coerce")
    y_reg = pd.to_numeric(y_reg, errors="coerce")
    valid = y_class.notna() & y_reg.notna()
    X = X.loc[valid].reset_index(drop=True)
    y_class = y_class.loc[valid].astype(int).reset_index(drop=True)
    y_reg = y_reg.loc[valid].reset_index(drop=True)
    return {
        "classification": eval_classification(clf, X, y_class),
        "regression": eval_regression(reg, X, y_reg),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_dir = cfg["paths"]["model_dir"]
    report_dir = cfg["paths"]["report_dir"]
    ensure_dir(report_dir)

    clf = joblib.load(os.path.join(model_dir, "toxicity_classifier.joblib"))
    reg = joblib.load(os.path.join(model_dir, "ld50_regressor.joblib"))

    dropped_path = os.path.join(model_dir, "dropped_columns.json")
    dropped_cols = []
    if os.path.exists(dropped_path):
        with open(dropped_path, "r") as f:
            dropped_cols = json.load(f).get("dropped_all_nan_columns", [])

    tox_col = cfg["columns"]["tox_label_col"]
    ld50_col = cfg["columns"]["ld50_col"]
    id_col = cfg["columns"].get("id_col")
    drop_cols = cfg["columns"].get("drop_cols", [])

    results = {}

    test_path = cfg["paths"].get("test_csv")
    if test_path and os.path.exists(test_path):
        test_df = load_csv(test_path)
        if dropped_cols:
            test_df = test_df.drop(columns=[c for c in dropped_cols if c in test_df.columns])
        results["test"] = eval_split("test", test_df, tox_col, ld50_col, id_col, drop_cols, clf, reg)

    ext_path = cfg["paths"].get("external_csv")
    if ext_path and os.path.exists(ext_path):
        ext_df = load_csv(ext_path)
        if dropped_cols:
            ext_df = ext_df.drop(columns=[c for c in dropped_cols if c in ext_df.columns])
        results["external"] = eval_split("external", ext_df, tox_col, ld50_col, id_col, drop_cols, clf, reg)

    with open(os.path.join(report_dir, "evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
