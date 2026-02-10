import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def split_features_targets(df, tox_label_col, ld50_col, id_col=None, drop_cols=None):
    base_drop = [c for c in [tox_label_col, ld50_col, id_col] if c and c in df.columns]
    extra_drop = [c for c in (drop_cols or []) if c in df.columns]
    X = df.drop(columns=base_drop + extra_drop)
    y_class = df[tox_label_col]
    y_reg = df[ld50_col]
    return X, y_class, y_reg


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # Convert bool to int, then coerce to numeric
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "bool":
            out[col] = out[col].astype(int)
    return out.apply(pd.to_numeric, errors="coerce")
