import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, QED, GraphDescriptors
import shap

# ---------- Load Models & Preprocessor ----------
model_lgbm = joblib.load("lgbm_model.pkl")
feature_names = joblib.load("lgbm_feature_names.pkl")
preprocessor = joblib.load("preprocessor.pkl")  # saved during training

clf = joblib.load("random_forest_classifier.pkl")
rf_features = joblib.load("rf_feature_names.pkl")

# SHAP explainer (initialize once)
explainer = shap.TreeExplainer(model_lgbm)

# ---------- Descriptor & ECFP Utilities ----------
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

# ---------- Main Prediction Function ----------
def predict_logS(smiles, top_n=5):
    # Step 1: descriptors + fingerprint
    desc = get_molecular_descriptors(smiles)
    ecfp = smiles_to_ecfp4(smiles)
    if desc is None or ecfp is None:
        return {"error": "Invalid SMILES"}

    # Step 2: preprocess
    X_desc = pd.DataFrame([desc])
    X_scaled = preprocessor.transform(X_desc)
    X_final = np.hstack([X_scaled, ecfp.reshape(1, -1)])

    # Step 3: predict logS
    logS_pred = model_lgbm.predict(X_final)[0]

    # Step 4: SHAP explanation
    shap_vals = explainer.shap_values(X_final)[0]
    df_shap = pd.DataFrame({
        "feature": feature_names,
        "value": X_final[0],
        "shap_value": shap_vals
    })
    df_shap["abs_shap"] = df_shap["shap_value"].abs()
    top_features = df_shap.sort_values("abs_shap", ascending=False).head(top_n)

    explanations = []
    for _, row in top_features.iterrows():
        direction = "increases" if row["shap_value"] > 0 else "decreases"
        explanations.append(f"Feature '{row['feature']}' ({row['value']:.3f}) {direction} solubility.")

    explanation_text = " ".join(explanations)

    # ---------- Classification ----------
    logP = desc["MolLogP"]  # take directly from descriptors
    X_class = pd.DataFrame([[logS_pred, logP]], columns=rf_features)

    class_pred = clf.predict(X_class)[0]  # array like [0,1,0,0]

    class_labels = ['I', 'II', 'III', 'IV']
    predicted_classes = [cls for cls, flag in zip(class_labels, class_pred) if flag == 1]

    if not predicted_classes:
        predicted_classes = ["Uncertain"]

    return {
        "logS": float(logS_pred),
        "logP": float(logP),
        "class": predicted_classes,
        "explanations": {
            "logS": explanation_text,
            "class": f"Classification predicted using Random Forest on logS and logP"
        }
    }

# ---------- CLI for quick test ----------
if __name__ == "__main__":
    test_smiles = "CCO"  # Ethanol
    result = predict_logS(test_smiles)
    print(result)
