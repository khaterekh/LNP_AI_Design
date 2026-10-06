from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "LNPDB" / "data" / "LNPDB_for_LiON" / "LNPDB.csv"
MODEL_PATH = PROJECT_ROOT / "results" / "models" / "lnp_rf_morgan_ratios.joblib"
METADATA_PATH = PROJECT_ROOT / "results" / "models" / "lnp_rf_morgan_ratios_metadata.json"

EXPERIMENT_METHOD = "luminescence_normalized"
TARGET = "Experiment_value"

RATIO_FEATURES = [
    "IL_molratio",
    "HL_molratio",
    "CHL_molratio",
    "PEG_molratio",
    "IL_to_nucleicacid_massratio",
]

FP_RADIUS = 2
FP_SIZE = 1024


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    lum = df[df["Experiment_method"] == EXPERIMENT_METHOD].copy()
    lum = lum.dropna(subset=[TARGET, "IL_SMILES"]).copy()

    print(f"Valid modeling rows: {len(lum):,}")
    print(f"Unique IL SMILES: {lum['IL_SMILES'].nunique():,}")

    generator = AllChem.GetMorganGenerator(
        radius=FP_RADIUS,
        fpSize=FP_SIZE,
    )

    fingerprint_cache = {}

    for smiles in lum["IL_SMILES"].unique():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")

        fp = generator.GetFingerprint(mol)

        arr = np.zeros((FP_SIZE,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprint_cache[smiles] = arr

    X_fp = np.vstack(
        [fingerprint_cache[s] for s in lum["IL_SMILES"]]
    )

    X_ratio = lum[RATIO_FEATURES].to_numpy(dtype=float)

    X = np.hstack([X_fp, X_ratio])
    y = lum[TARGET].to_numpy(dtype=float)

    print(f"Feature matrix: {X.shape}")
    print(f"Target vector: {y.shape}")

    model = RandomForestRegressor(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    print("Training final Random Forest...")
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_type": "RandomForestRegressor",
        "n_estimators": 300,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "random_state": 42,
        "training_rows": int(len(lum)),
        "n_features": int(X.shape[1]),
        "fingerprint_type": "Morgan",
        "radius": FP_RADIUS,
        "size": FP_SIZE,
        "chemical_feature_source": "IL_SMILES",
        "ratio_features": RATIO_FEATURES,
        "target": TARGET,
        "experiment_method": EXPERIMENT_METHOD,
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()
