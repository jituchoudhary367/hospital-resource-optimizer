import mlflow.pyfunc

MODEL_NAME = "HospitalLOSModel"

model_uri = f"models:/{MODEL_NAME}@production"

print("Loading production model from registry (alias)...")
model = mlflow.pyfunc.load_model(model_uri)

print("Model loaded:", model)
