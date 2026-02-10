import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.toxicity.featurize import featurize_smiles

MODEL_DIR = os.getenv("MODEL_DIR", "models")

clf_path = os.path.join(MODEL_DIR, "toxicity_classifier.joblib")
reg_path = os.path.join(MODEL_DIR, "ld50_regressor.joblib")
feat_path = os.path.join(MODEL_DIR, "feature_columns.json")
dropped_path = os.path.join(MODEL_DIR, "dropped_columns.json")
meta_path = os.path.join(MODEL_DIR, "featuregen_meta.json")

st.set_page_config(page_title="Toxicity + LD50 Demo", layout="wide")

st.title("Toxicity Classification + LD50 Prediction")

if not os.path.exists(clf_path) or not os.path.exists(reg_path):
    st.error("Models not found. Train first and ensure models are in models/.")
    st.stop()

clf = joblib.load(clf_path)
reg = joblib.load(reg_path)

featuregen = {"use_padel": True, "use_mordred": True, "mordred_n": None}
if os.path.exists(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
        featuregen.update(meta.get("feature_generation", {}))

if os.path.exists(feat_path):
    with open(feat_path, "r") as f:
        raw_feature_columns = json.load(f).get("raw_feature_columns", [])
else:
    raw_feature_columns = []

if os.path.exists(meta_path) and not raw_feature_columns:
    with open(meta_path, "r") as f:
        raw_feature_columns = json.load(f).get("features", [])

if os.path.exists(dropped_path):
    with open(dropped_path, "r") as f:
        dropped_cols = set(json.load(f).get("dropped_all_nan_columns", []))
else:
    dropped_cols = set()

mode = st.radio("Input mode", ["SMILES", "Upload CSV", "Single JSON"], horizontal=True)

if mode == "SMILES":
    smiles = st.text_input("Enter SMILES")
    if st.button("Predict") and smiles:
        try:
            feat = featurize_smiles(
                [smiles],
                use_padel=featuregen.get("use_padel", True),
                use_mordred=featuregen.get("use_mordred", True),
                mordred_n=featuregen.get("mordred_n"),
            )
            if raw_feature_columns:
                for c in raw_feature_columns:
                    if c not in feat.columns:
                        feat[c] = pd.NA
                feat = feat[raw_feature_columns]
            if dropped_cols:
                feat = feat.drop(columns=[c for c in dropped_cols if c in feat.columns])
            feat = feat.apply(pd.to_numeric, errors="coerce")
            tox_prob = float(clf.predict_proba(feat)[:, 1][0])
            tox_label = int(tox_prob >= 0.5)
            ld50_pred = float(reg.predict(feat)[0])
            st.json({
                "toxicity_probability": tox_prob,
                "toxicity_label": tox_label,
                "ld50_prediction": ld50_pred,
            })
        except Exception as e:
            st.error(f"Failed to compute descriptors: {e}")

elif mode == "Upload CSV":
    file = st.file_uploader("Upload CSV with feature columns", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if raw_feature_columns:
            missing = [c for c in raw_feature_columns if c not in df.columns]
            if missing:
                st.error(f"Missing features: {missing}")
            else:
                X = df[raw_feature_columns].apply(pd.to_numeric, errors="coerce")
                if dropped_cols:
                    X = X.drop(columns=[c for c in dropped_cols if c in X.columns])
                tox_prob = clf.predict_proba(X)[:, 1]
                tox_label = (tox_prob >= 0.5).astype(int)
                ld50_pred = reg.predict(X)
                out = df.copy()
                out["toxicity_probability"] = tox_prob
                out["toxicity_label"] = tox_label
                out["ld50_prediction"] = ld50_pred
                st.dataframe(out.head(20))
                st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv")
        else:
            st.error("feature_columns.json not found; train the model to generate it.")

else:
    example = {c: 0 for c in (raw_feature_columns[:5] if raw_feature_columns else [])}
    raw = st.text_area("Enter JSON features", value=json.dumps(example, indent=2))
    if st.button("Predict"):
        try:
            features = json.loads(raw)
            if raw_feature_columns:
                missing = [c for c in raw_feature_columns if c not in features]
                if missing:
                    st.error(f"Missing features: {missing}")
                else:
                    row = {c: features[c] for c in raw_feature_columns}
                    for c in list(row.keys()):
                        if c in dropped_cols:
                            row.pop(c, None)
                    X = pd.DataFrame([row]).apply(pd.to_numeric, errors="coerce")
                    tox_prob = float(clf.predict_proba(X)[:, 1][0])
                    tox_label = int(tox_prob >= 0.5)
                    ld50_pred = float(reg.predict(X)[0])
                    st.json({
                        "toxicity_probability": tox_prob,
                        "toxicity_label": tox_label,
                        "ld50_prediction": ld50_pred,
                    })
            else:
                st.error("feature_columns.json not found; train the model to generate it.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
