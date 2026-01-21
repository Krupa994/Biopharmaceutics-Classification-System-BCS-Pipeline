from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import shap
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED, GraphDescriptors, AllChem
from fastapi.middleware.cors import CORSMiddleware

import bcs_explainer

# ----------------------------
# Load artifacts for logS
# ----------------------------
model_lgbm = joblib.load("lgbm_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
feature_names = joblib.load("lgbm_feature_names.pkl")

# Setup SHAP for logS explanation
explainer = shap.TreeExplainer(model_lgbm)

# ----------------------------
# FastAPI + CORS
# ----------------------------
class SMILESInput(BaseModel):
    smiles: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # adjust for deployed origin if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Utility functions (logS flow)
# ----------------------------
def smiles_to_ecfp4(smiles, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp)

def get_molecular_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return {
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
        'RadicalElectrons': Descriptors.NumRadicalElectrons(mol)
    }

def generate_lgbm_explanation(instance_features, shap_values, feature_names, top_n=5):
    df_shap = pd.DataFrame({
        "feature": feature_names,
        "value": instance_features,
        "shap_value": shap_values
    })
    df_shap["abs_shap"] = df_shap["shap_value"].abs()
    df_top = df_shap.sort_values("abs_shap", ascending=False).head(top_n)

    explanations = []
    for _, row in df_top.iterrows():
        direction = "increases" if row["shap_value"] > 0 else "decreases"
        explanations.append(
            f"The feature '{row['feature']}' with value {row['value']:.3f} {direction} solubility."
        )
    return " ".join(explanations)

# ----------------------------
# API Route
# ----------------------------
@app.post("/predict_logS")
def predict(input: SMILESInput):
    smiles = input.smiles
    descriptors = get_molecular_descriptors(smiles)
    ecfp = smiles_to_ecfp4(smiles)

    if descriptors is None or ecfp is None:
        return {"error": "Invalid SMILES"}

    # --- logS prediction (unchanged) ---
    X_desc = pd.DataFrame([descriptors])
    X_proc = preprocessor.transform(X_desc)
    X_final = np.hstack([X_proc, ecfp.reshape(1, -1)])

    logS_pred = model_lgbm.predict(X_final)[0]
    shap_values = explainer.shap_values(X_final)[0]
    logS_explanation = generate_lgbm_explanation(X_final[0], shap_values, feature_names)

    # --- CatBoost classification & explanation (using bcs_explainer) ---
    # Call explain_bcs with logS_override so CatBoost uses our LGBM logS
    cb_result = bcs_explainer.explain_bcs(smiles,
                                         model_dir="./artifacts_bcs",
                                         top_k_desc=6,
                                         threshold=0.5,
                                         logS_override=float(logS_pred))

    # cb_result contains:
    #  - "probabilities": {label: prob}
    #  - "predicted_class": label
    #  - "predicted_flag": 0/1 for selected label
    #  - "descriptor_share", "ecfp_share", "top_descriptors", "bias_term", etc.

    # Build final response merging LGBM outputs + CatBoost explainer outputs
    response = {
        "Values": descriptors,
        "logS": float(logS_pred),
        "logP": float(descriptors['MolLogP']),
        "logS_explanation": logS_explanation,
        # CatBoost outputs
        "cb_probabilities": cb_result.get("probabilities"),
        "cb_predicted_class": cb_result.get("predicted_class"),
        "cb_predicted_flag": int(cb_result.get("predicted_flag", 0)),
        "cb_descriptor_share": cb_result.get("descriptor_share"),   
        "cb_ecfp_share": cb_result.get("ecfp_share"),
        "cb_top_descriptors": cb_result.get("top_descriptors"),
        "cb_bias": cb_result.get("bias_term"),
        # Human-friendly class explanation
        "class_explanation": f"Predicted BCS class (CatBoost): {cb_result.get('predicted_class')}"
    }

    return response
