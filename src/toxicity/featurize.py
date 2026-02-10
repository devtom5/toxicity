import os
import subprocess
import tempfile
import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors


def _safe_mol(smiles: str):
    if not isinstance(smiles, str):
        return None
    return Chem.MolFromSmiles(smiles)


def compute_mordred(smiles_list, mordred_n=None):
    mols = [_safe_mol(s) for s in smiles_list]
    base_calc = Calculator(descriptors, ignore_3D=True)
    if mordred_n is not None and mordred_n > 0:
        desc_list = base_calc.descriptors[:mordred_n]
        calc = Calculator(desc_list, ignore_3D=True)
    else:
        calc = base_calc

    valid_idx = [i for i, m in enumerate(mols) if m is not None]
    valid_mols = [mols[i] for i in valid_idx]

    if not valid_mols:
        return pd.DataFrame(index=range(len(smiles_list)))

    valid_df = calc.pandas(valid_mols, quiet=True)
    valid_df = valid_df.apply(pd.to_numeric, errors="coerce")

    full_df = pd.DataFrame(index=range(len(smiles_list)), columns=valid_df.columns)
    for row_idx, src_idx in enumerate(valid_idx):
        full_df.loc[src_idx] = valid_df.iloc[row_idx].values

    return full_df


def _run_padel(smiles_list, timeout_sec=600):
    import padelpy

    jar_path = os.path.join(os.path.dirname(padelpy.__file__), "PaDEL-Descriptor", "PaDEL-Descriptor.jar")
    if not os.path.exists(jar_path):
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        smi_path = os.path.join(tmpdir, "input.smi")
        out_path = os.path.join(tmpdir, "output.csv")
        with open(smi_path, "w") as f:
            for i, s in enumerate(smiles_list):
                if not isinstance(s, str):
                    s = ""
                f.write(f"{s}\tC{i}\n")

        cmd = [
            "java",
            "-Djava.awt.headless=true",
            "-jar",
            jar_path,
            "-2d",
            "-dir",
            smi_path,
            "-file",
            out_path,
            "-retainorder",
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout_sec)

        if not os.path.exists(out_path):
            return None

        df = pd.read_csv(out_path)
        if "Name" in df.columns:
            df = df.drop(columns=["Name"])
        df = df.apply(pd.to_numeric, errors="coerce")
        return df


def compute_padel(smiles_list, chunk_size=1000, timeout_sec=600, max_rows=None):
    try:
        import padelpy  # noqa: F401
    except Exception:
        return None

    n = len(smiles_list)
    limit = min(n, max_rows) if max_rows else n
    parts = []
    columns = None

    for i in range(0, limit, chunk_size):
        chunk = smiles_list[i:i + chunk_size]
        df = None
        for _ in range(2):
            try:
                df = _run_padel(chunk, timeout_sec=timeout_sec)
                break
            except Exception:
                df = None
        if df is None:
            if columns is None:
                continue
            empty = pd.DataFrame({c: [np.nan] * len(chunk) for c in columns})
            parts.append(empty)
            continue
        if columns is None:
            columns = list(df.columns)
        else:
            for c in columns:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[columns]
        parts.append(df)

    if not parts or columns is None:
        return None

    padel_df = pd.concat(parts, axis=0, ignore_index=True)

    # If we limited rows, pad remaining with NaNs
    if limit < n:
        extra = pd.DataFrame({c: [np.nan] * (n - limit) for c in columns})
        padel_df = pd.concat([padel_df, extra], axis=0, ignore_index=True)

    return padel_df


def featurize_smiles(smiles_list, use_padel=True, use_mordred=True, mordred_n=None, padel_max_rows=None):
    parts = []
    if use_padel:
        padel_df = compute_padel(smiles_list, max_rows=padel_max_rows)
        if padel_df is not None:
            parts.append(padel_df)
    if use_mordred:
        mordred_df = compute_mordred(smiles_list, mordred_n=mordred_n)
        parts.append(mordred_df)

    if not parts:
        raise RuntimeError("No descriptors computed (Padel/Mordred failed or disabled).")

    feat = pd.concat(parts, axis=1)
    feat = feat.loc[:, ~feat.columns.duplicated()]
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feat
