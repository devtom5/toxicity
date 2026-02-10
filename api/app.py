import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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

if not os.path.exists(clf_path) or not os.path.exists(reg_path):
    raise RuntimeError("Models not found. Train first and ensure models are in models/.")

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


class PredictRequest(BaseModel):
    features: Dict[str, Any]


class SmilesRequest(BaseModel):
    smiles: str


app = FastAPI(title="Toxicity + LD50 API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    return {"raw_feature_columns": raw_feature_columns}


@app.post("/predict")
def predict(req: PredictRequest):
    if raw_feature_columns:
        missing = [c for c in raw_feature_columns if c not in req.features]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
        row = {c: req.features[c] for c in raw_feature_columns}
    else:
        row = req.features

    for c in list(row.keys()):
        if c in dropped_cols:
            row.pop(c, None)

    X = pd.DataFrame([row]).apply(pd.to_numeric, errors="coerce")
    tox_prob = float(clf.predict_proba(X)[:, 1][0])
    tox_label = int(tox_prob >= 0.5)
    ld50_pred = float(reg.predict(X)[0])

    return {
        "toxicity_probability": tox_prob,
        "toxicity_label": tox_label,
        "ld50_prediction": ld50_pred,
    }


@app.post("/predict_smiles")
def predict_smiles(req: SmilesRequest):
    feat = featurize_smiles(
        [req.smiles],
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

    return {
        "toxicity_probability": tox_prob,
        "toxicity_label": tox_label,
        "ld50_prediction": ld50_pred,
    }
