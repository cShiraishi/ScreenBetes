import joblib
import sys

try:
    model = joblib.load('qsar_app/qsar_model_SLTG2.pkl')
    print(f"Model Type: {type(model)}")
    print(f"Model Module: {model.__module__}")
except Exception as e:
    print(f"Error: {e}")
