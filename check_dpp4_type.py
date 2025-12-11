import joblib
import sys

try:
    model = joblib.load('qsar_app/model_ddp4_rf.pkl')
    print(f"Model Type: {type(model)}")
    print(f"Model Module: {model.__module__}")
except Exception as e:
    print(f"Error: {e}")
