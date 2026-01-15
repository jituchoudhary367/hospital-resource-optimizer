import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import sys
from pathlib import Path
from math import sqrt

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

from src.features.build_features import build_pipeline


# ---------------- CONFIG ----------------

DATA_PATH = ROOT / "data" / "processed" / "clean.csv"
TARGET = "time_in_hospital"


# ---------------- MODEL FACTORY ----------------

def get_model(model_name, trees):
    if model_name == "lr":
        return LinearRegression()
    elif model_name == "rf":
        return RandomForestRegressor(
            n_estimators=trees,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError("Model must be 'lr' or 'rf'")


# ---------------- TRAIN PIPELINE ----------------

def main(args):
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Building preprocessing pipeline...")
    preprocessor = build_pipeline()

    model = get_model(args.model, args.trees)

    full_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    mlflow.set_experiment("Hospital_LOS_Prediction")

    with mlflow.start_run(run_name=f"{args.model}_pipeline"):
        print("Training model...")
        full_pipeline.fit(X_train, y_train)

        preds = full_pipeline.predict(X_test)

        rmse = sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model", args.model)
        mlflow.log_param("trees", args.trees)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(full_pipeline, name="model")

        print("RMSE:", rmse)
        print("R2:", r2)

        # -------- SAVE FULL PIPELINE FOR DEPLOYMENT --------
        model_output_path = ROOT / "full_model_pipeline.joblib"
        dump(full_pipeline, model_output_path)
        print(f"Saved deployment model to: {model_output_path}")

    print("Pipeline run complete.")


# ---------------- CLI ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rf", choices=["lr", "rf"])
    parser.add_argument("--trees", type=int, default=100)

    args = parser.parse_args()
    main(args)
