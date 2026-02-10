import argparse
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import load_config, ensure_dir
from .featurize import featurize_smiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_csv = cfg["paths"]["raw_csv"]
    smiles_col = cfg["columns"]["smiles_col"]
    tox_col = cfg["columns"]["tox_label_col"]
    ld50_col = cfg["columns"]["ld50_col"]
    id_col = cfg["columns"].get("id_col")

    fg = cfg.get("feature_generation", {})
    use_padel = fg.get("use_padel", True)
    use_mordred = fg.get("use_mordred", True)
    mordred_n = fg.get("mordred_n")
    padel_max_rows = fg.get("padel_max_rows")

    df = pd.read_csv(raw_csv)
    df = df[df[smiles_col].notna()].copy()

    smiles = df[smiles_col].tolist()
    feat = featurize_smiles(
        smiles,
        use_padel=use_padel,
        use_mordred=use_mordred,
        mordred_n=mordred_n,
        padel_max_rows=padel_max_rows,
    )

    out = feat.copy()
    out[tox_col] = pd.to_numeric(df[tox_col], errors="coerce")
    out[ld50_col] = pd.to_numeric(df[ld50_col], errors="coerce")
    if id_col and id_col in df.columns:
        out[id_col] = df[id_col].values

    out = out.dropna(subset=[tox_col, ld50_col]).reset_index(drop=True)

    stratify = out[tox_col] if out[tox_col].nunique() > 1 else None
    train_df, temp_df = train_test_split(
        out, test_size=0.30, random_state=42, stratify=stratify
    )
    stratify_temp = temp_df[tox_col] if temp_df[tox_col].nunique() > 1 else None
    test_df, ext_df = train_test_split(
        temp_df, test_size=1/3, random_state=42, stratify=stratify_temp
    )

    ensure_dir(cfg["paths"]["data_dir"])
    train_df.to_csv(cfg["paths"]["train_csv"], index=False)
    test_df.to_csv(cfg["paths"]["test_csv"], index=False)
    ext_df.to_csv(cfg["paths"]["external_csv"], index=False)

    meta = {
        "raw_csv": raw_csv,
        "smiles_col": smiles_col,
        "features": [c for c in out.columns if c not in [tox_col, ld50_col, id_col]],
        "rows": {
            "train": len(train_df),
            "test": len(test_df),
            "external": len(ext_df),
        },
        "feature_generation": {
            "use_padel": use_padel,
            "use_mordred": use_mordred,
            "mordred_n": mordred_n,
            "padel_max_rows": padel_max_rows,
        },
    }
    ensure_dir(cfg["paths"]["model_dir"])
    with open(os.path.join(cfg["paths"]["model_dir"], "featuregen_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
