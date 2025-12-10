import joblib
import numpy as np
import sys
import os
from rdkit import Chem
from rdkit.Chem import AllChem

def test_prediction():
    print("Iniciando teste de predição...")
    
    # Path setup
    model_path = "qsar_app/xgboost_ptp1b_final.pkl"
    if not os.path.exists(model_path):
        model_path = "xgboost_ptp1b_final.pkl"
    
    if not os.path.exists(model_path):
        print(f"Erro: Modelo não encontrado em {model_path}")
        return

    # Load Model
    try:
        model = joblib.load(model_path)
        print("✅ Modelo carregado com sucesso.")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return

    # Dummy SMILES (Aspirin)
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    
    # Needs to convert to numpy array equivalent to training
    arr = np.zeros((1,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    
    # The XGBoost model usually expects a 2D array (n_samples, n_features)
    # RDKit ToNumpyArray gives a 1D array of size nBits
    # So we reshape or wrap it
    X_input = np.array([arr]) 
    
    print(f"Shape do input: {X_input.shape}")
    
    try:
        pred = model.predict(X_input)
        prob = model.predict_proba(X_input)
        print(f"✅ Predição realizada com sucesso!")
        print(f"Classe: {pred[0]}")
        print(f"Probabilidade: {prob[0]}")
    except Exception as e:
        print(f"❌ Erro na predição: {e}")

if __name__ == "__main__":
    test_prediction()