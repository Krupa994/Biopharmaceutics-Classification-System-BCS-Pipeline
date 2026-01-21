# bcs_explainer.py
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED, GraphDescriptors
from catboost import Pool
import json
import pickle
from typing import Optional


def load_artifacts(model_dir: str = "./artifacts_bcs"):
    with open(f"{model_dir}/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    with open(f"{model_dir}/catboost_ovr.pkl", "rb") as f:
        clf_cb = pickle.load(f)

    with open(f"{model_dir}/descriptor_columns.json", "r") as f:
        cfg = json.load(f)

    descriptor_cols = cfg["descriptor_cols"]
    continuous_features = cfg["continuous_features"]
    count_features = cfg["count_features"]
    qed_feature = cfg["qed_feature"]
    labels = cfg["labels"]

    # ensure qed_feature is a list
    if isinstance(qed_feature, str):
        qed_feature = [qed_feature]

    return clf_cb, preprocessor, descriptor_cols, continuous_features, count_features, qed_feature, labels


# ------------------------------------------------------
# Descriptor meanings (human-friendly)
# ------------------------------------------------------
descriptor_gloss = {
    "MolWt": "Molecular weight (size/mass of the molecule)",
    "LabuteASA": "Approximate solvent-accessible surface area",
    "MolLogP": "Lipophilicity (octanol/water partition, logP)",
    "MolMR": "Molar refractivity (volume/polarizability)",
    "FractionCSP3": "Fraction of sp3 carbons (3D saturation)",
    "BalabanJ": "Balaban J topological index (shape/branching)",
    "BertzCT": "Bertz complexity index (molecular complexity)",
    "Ipc": "Information content index (branching/complexity)",
    "TPSA": "Topological polar surface area (polarity/H-bonding)",
    "logS": "Aqueous solubility (logS)",
    "HeavyAtomCount": "Count of heavy atoms (non-H)",
    "NumRotatableBonds": "Flexibility",
    "NumHAcceptors": "Hydrogen bond acceptors",
    "NumHDonors": "Hydrogen bond donors",
    "RingCount": "Number of rings",
    "FormalCharge": "Net formal charge",
    "RadicalElectrons": "Unpaired electrons",
    "QED": "Drug-likeness score"
}


# ------------------------------------------------------
# Descriptor Extraction
# ------------------------------------------------------
def smiles_to_descriptors(smiles: str, descriptor_cols, logS_override: Optional[float] = None):
    """
    Return a pandas Series containing descriptor_cols order.
    If logS_override is provided, it will be used for 'logS' descriptor.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    data = {
        'MolWt': Descriptors.MolWt(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'TPSA': Descriptors.TPSA(mol),
        'LabuteASA': rdMolDescriptors.CalcLabuteASA(mol),
        'MolLogP': rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
        'MolMR': rdMolDescriptors.CalcCrippenDescriptors(mol)[1],
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'RingCount': Descriptors.RingCount(mol),
        'QED': QED.qed(mol),
        'BalabanJ': GraphDescriptors.BalabanJ(mol),
        'BertzCT': GraphDescriptors.BertzCT(mol),
        'Ipc': GraphDescriptors.Ipc(mol),
        'FormalCharge': Chem.GetFormalCharge(mol),
        'RadicalElectrons': Descriptors.NumRadicalElectrons(mol),
        # use override if provided
        'logS': float(logS_override) if logS_override is not None else np.nan
    }

    # ensure requested columns exist
    series = pd.Series(data)
    missing = [c for c in descriptor_cols if c not in series.index]
    if missing:
        raise KeyError(f"Descriptor columns missing in generated data: {missing}")

    return series[descriptor_cols]


def smiles_to_ecfp4_array(smiles: str, n_bits: int = 1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp, dtype=np.int8)


# ------------------------------------------------------
# MAIN EXPLAINER
# ------------------------------------------------------
def explain_bcs(smiles: str,
                model_dir: str = "./artifacts_bcs",
                top_k_desc: int = 6,
                threshold: float = 0.5,
                logS_override: Optional[float] = None):
    """
    Returns a dictionary containing CatBoost OVR probabilities, predicted class,
    top descriptor contributions and other explanation metadata.
    """

    # 1. Load model + metadata
    clf_cb, preprocessor, descriptor_cols, continuous_features, count_features, qed_feature, labels = load_artifacts(model_dir)

    # Build descriptor feature names in same order used in training
    descriptor_feature_names = list(continuous_features) + list(qed_feature) + list(count_features)
    ecfp_feature_names = [f"ECFP4_{i}" for i in range(1024)]
    feature_names = descriptor_feature_names + ecfp_feature_names

    # 3. Extract desc + ECFP (uses logS_override if provided)
    desc = smiles_to_descriptors(smiles, descriptor_cols, logS_override=logS_override)
    desc_df = pd.DataFrame([desc])
    ecfp = smiles_to_ecfp4_array(smiles)

    # Basic cleaning (same as training)
    desc_df = desc_df.fillna(desc_df.median()).clip(-1e6, 1e6)

    # 4. Transform descriptors
    X_desc_scaled = preprocessor.transform(desc_df)  # expects shape (1, n_desc)
    X_hybrid = np.hstack([X_desc_scaled, ecfp.reshape(1, -1)])  # shape (1, n_features)

    # 5. Predict probabilities for each estimator in the OVR ensemble
    # clf_cb.estimators_ must be iterable of trained CatBoost binary estimators
    probs = np.array([est.predict_proba(X_hybrid)[0, 1] for est in clf_cb.estimators_])
    preds = (probs >= threshold).astype(int)
    cls_id = int(np.argmax(probs))
    target_label = labels[cls_id] if (len(labels) > cls_id) else str(cls_id)

    # 6. SHAP contributions: CatBoost's get_feature_importance with type="ShapValues"
    est = clf_cb.estimators_[cls_id]
    # returns shape (n_samples, n_features+1) where last col is expected base value (bias)
    shap_vals = est.get_feature_importance(Pool(X_hybrid, label=None), type="ShapValues")
    if shap_vals.ndim == 1:
        raise RuntimeError("Unexpected shap values shape from CatBoost")
    contrib = shap_vals[0, :-1]
    bias = float(shap_vals[0, -1])

    # Split desc vs ECFP
    n_desc = len(descriptor_feature_names)
    desc_contrib = contrib[:n_desc]
    ecfp_contrib = contrib[n_desc:]

    denom = (np.sum(np.abs(contrib)) + 1e-12)
    desc_share = float(np.sum(np.abs(desc_contrib)) / denom)
    ecfp_share = float(np.sum(np.abs(ecfp_contrib)) / denom)

    # 7. Top descriptor contributors
    order = np.argsort(np.abs(desc_contrib))[::-1][:top_k_desc]
    top_desc = []
    for j in order:
        if j >= len(descriptor_feature_names):
            continue
        name = descriptor_feature_names[j]
        top_desc.append({
            "name": name,
            "meaning": descriptor_gloss.get(name, "Descriptor"),
            "value": float(desc_df.iloc[0][name]),
            "contribution": float(desc_contrib[j]),
            "effect": "toward" if desc_contrib[j] >= 0 else "against"
        })

    # 8. Package explanation
    result = {
        "smiles": smiles,
        "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))},
        "predicted_class": target_label,
        "predicted_flag": int(preds[cls_id]),
        "descriptor_share": desc_share,
        "ecfp_share": ecfp_share,
        "top_descriptors": top_desc,
        "bias_term": bias
    }

    return result
