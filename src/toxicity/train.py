import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectFromModel

from .data import load_csv, split_features_targets, coerce_numeric
from .features import PreprocessConfig, build_preprocessor, get_feature_names, select_top_k_by_shap
from .modeling import build_classifier, build_regressor
from .utils import load_config, ensure_dir


def train_task(X, y, pre_cfg, model_cfg, fs_method, top_k, shap_sample, shap_plot, out_prefix, report_dir):
    preprocessor = build_preprocessor(X, pre_cfg)
    Xp = preprocessor.fit_transform(X)
    Xp = np.asarray(Xp)
    if Xp.ndim > 2:
        Xp = Xp.reshape(Xp.shape[0], -1)

    model = model_cfg["builder"](model_cfg)
    model.fit(Xp, y)

    feature_names = get_feature_names(preprocessor)

    if fs_method == "importance":
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model does not support feature_importances_")
        imp = model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:top_k]

        selector = SelectFromModel(model, prefit=True, max_features=top_k, threshold=-np.inf)
        X_sel = selector.transform(Xp)
        X_sel = np.asarray(X_sel)
        if X_sel.ndim > 2:
            X_sel = X_sel.reshape(X_sel.shape[0], -1)

        model_sel = model_cfg["builder"](model_cfg)
        model_sel.fit(X_sel, y)

        selected_names = [feature_names[i] for i in top_idx]
        imp_df = pd.DataFrame({"feature": feature_names, "importance": imp})
        imp_df.sort_values("importance", ascending=False, inplace=True)
        imp_df.to_csv(os.path.join(report_dir, f"{out_prefix}_importance.csv"), index=False)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("model", model_sel),
        ])

        return pipeline, {
            "selected_features": selected_names,
            "top_k": top_k,
            "importance_report": f"{out_prefix}_importance.csv",
        }

    if fs_method == "shap":
        explainer = shap.TreeExplainer(model)
        if shap_sample and Xp.shape[0] > shap_sample:
            rng = np.random.default_rng(42)
            idx = rng.choice(Xp.shape[0], size=shap_sample, replace=False)
            Xp_shap = Xp[idx]
        else:
            Xp_shap = Xp

        shap_values = explainer.shap_values(Xp_shap)
        if isinstance(shap_values, list):
            shap_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_matrix = shap_values
        if getattr(shap_matrix, "ndim", 0) == 3:
            shap_matrix = shap_matrix.mean(axis=2)

        top_idx = select_top_k_by_shap(shap_matrix, top_k)

        selector = SelectFromModel(model, prefit=True, max_features=top_k, threshold=-np.inf)
        X_sel = selector.transform(Xp)
        X_sel = np.asarray(X_sel)
        if X_sel.ndim > 2:
            X_sel = X_sel.reshape(X_sel.shape[0], -1)

        model_sel = model_cfg["builder"](model_cfg)
        model_sel.fit(X_sel, y)

        selected_names = [feature_names[i] for i in top_idx]

        imp = np.abs(shap_matrix).mean(axis=0)
        imp_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
        imp_df.sort_values("mean_abs_shap", ascending=False, inplace=True)
        imp_df.to_csv(os.path.join(report_dir, f"{out_prefix}_shap_importance.csv"), index=False)

        shap_plot_path = None
        if shap_plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_matrix, Xp_shap, feature_names=feature_names, show=False)
            shap_plot_path = os.path.join(report_dir, f"{out_prefix}_shap_summary.png")
            plt.tight_layout()
            plt.savefig(shap_plot_path, dpi=200)
            plt.close()

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("model", model_sel),
        ])

        return pipeline, {
            "selected_features": selected_names,
            "top_k": top_k,
            "shap_plot": shap_plot_path,
        }

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("identity", FunctionTransformer(lambda x: x)),
        ("model", model),
    ])

    return pipeline, {
        "selected_features": feature_names,
        "top_k": None,
        "shap_plot": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    train_df = load_csv(cfg["paths"]["train_csv"])
    tox_col = cfg["columns"]["tox_label_col"]
    ld50_col = cfg["columns"]["ld50_col"]
    id_col = cfg["columns"].get("id_col")

    drop_cols = cfg["columns"].get("drop_cols", [])
    X, y_class, y_reg = split_features_targets(train_df, tox_col, ld50_col, id_col, drop_cols)
    X = coerce_numeric(X)
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
    y_class = pd.to_numeric(y_class, errors="coerce")
    y_reg = pd.to_numeric(y_reg, errors="coerce")
    valid = y_class.notna() & y_reg.notna()
    X = X.loc[valid].reset_index(drop=True)
    y_class = y_class.loc[valid].astype(int).reset_index(drop=True)
    y_reg = y_reg.loc[valid].reset_index(drop=True)
    raw_feature_cols = list(X.columns)

    pre_cfg = PreprocessConfig(
        scale_numeric=cfg["preprocess"]["scale_numeric"],
        encode_categorical=cfg["preprocess"]["encode_categorical"],
    )

    model_dir = cfg["paths"]["model_dir"]
    report_dir = cfg["paths"]["report_dir"]
    ensure_dir(model_dir)
    ensure_dir(report_dir)

    fs_method = cfg["feature_selection"].get("method", "shap")
    top_k = cfg["feature_selection"]["top_k"]
    shap_sample = cfg["feature_selection"].get("shap_sample", 1000)
    shap_plot = cfg["feature_selection"].get("shap_plot", False)

    class_cfg = cfg["models"]["classifier"].copy()
    class_cfg["builder"] = build_classifier

    reg_cfg = cfg["models"]["regressor"].copy()
    reg_cfg["builder"] = build_regressor

    clf_pipeline, clf_info = train_task(
        X, y_class, pre_cfg, class_cfg, fs_method, top_k, shap_sample, shap_plot, "classification", report_dir
    )

    reg_pipeline, reg_info = train_task(
        X, y_reg, pre_cfg, reg_cfg, fs_method, top_k, shap_sample, shap_plot, "regression", report_dir
    )

    joblib.dump(clf_pipeline, os.path.join(model_dir, "toxicity_classifier.joblib"))
    joblib.dump(reg_pipeline, os.path.join(model_dir, "ld50_regressor.joblib"))

    with open(os.path.join(report_dir, "training_summary.json"), "w") as f:
        json.dump({"classification": clf_info, "regression": reg_info}, f, indent=2)

    with open(os.path.join(model_dir, "feature_columns.json"), "w") as f:
        json.dump({"raw_feature_columns": raw_feature_cols}, f, indent=2)

    with open(os.path.join(model_dir, "dropped_columns.json"), "w") as f:
        json.dump({"dropped_all_nan_columns": all_nan_cols}, f, indent=2)


if __name__ == "__main__":
    main()
