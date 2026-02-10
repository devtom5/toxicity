import pandas as pd
import numpy as np
from rdkit import Chem


def _safe_mol(smiles: str):
    if not isinstance(smiles, str):
        return None
    return Chem.MolFromSmiles(smiles)


def compute_mordred(smiles_list, mordred_n=None):
    # Lazy import to avoid import-time crashes in some environments
    from mordred import Calculator, descriptors

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


def compute_padel(smiles_list, chunk_size=200, timeout=120, maxruntime=5, threads=1):
    try:
        from padelpy import from_smiles
    except Exception:
        return None

    mols = [_safe_mol(s) for s in smiles_list]
    valid_idx = [i for i, m in enumerate(mols) if m is not None]
    valid_smiles = [smiles_list[i] for i in valid_idx]

    rows = []
    for i in range(0, len(valid_smiles), chunk_size):
        chunk = valid_smiles[i:i + chunk_size]
        try:
            data = from_smiles(
                chunk,
                fingerprints=False,
                descriptors=True,
                timeout=timeout,
                maxruntime=maxruntime,
                threads=threads,
            )
            if isinstance(data, list):
                rows.extend(data)
            else:
                rows.append(data)
        except Exception:
            for _ in chunk:
                rows.append({})

    if not rows:
        return pd.DataFrame(index=range(len(smiles_list)))

    valid_df = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")

    full_df = pd.DataFrame(index=range(len(smiles_list)), columns=valid_df.columns)
    for row_idx, src_idx in enumerate(valid_idx):
        full_df.loc[src_idx] = valid_df.iloc[row_idx].values

    return full_df


def featurize_smiles(smiles_list, use_padel=True, use_mordred=True, mordred_n=None):
    parts = []
    if use_padel:
        padel_df = compute_padel(smiles_list)
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
