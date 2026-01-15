<<<<<<< HEAD
import pandas as pd
import mlflow
import mlflow.sklearn

from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from math import sqrt


DATA_PATH = "data/processed/clean.csv"
PREPROCESSOR_PATH = "src/features/preprocess_pipeline.joblib"
TARGET = "time_in_hospital"


def train_and_log(model, model_name, pipeline, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):

        full_pipeline = Pipeline(
            steps=[
                ("preprocess", pipeline),
                ("model", model),
            ]
        )

        full_pipeline.fit(X_train, y_train)

        preds = full_pipeline.predict(X_test)

        rmse = sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(full_pipeline, name="model")

        print(f"{model_name} | RMSE: {rmse:.3f} | R2: {r2:.3f}")


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Loading preprocessing pipeline...")
    preprocessor = load(PREPROCESSOR_PATH)

    mlflow.set_experiment("Hospital_LOS_Prediction")

    print("Training models...")

    lr = LinearRegression()
    train_and_log(lr, "LinearRegression", preprocessor, X_train, X_test, y_train, y_test)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    train_and_log(rf, "RandomForest", preprocessor, X_train, X_test, y_train, y_test)

    print("Training complete.")


if __name__ == "__main__":
    main()
=======
import pandas as pd
import mlflow
import mlflow.sklearn

from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from math import sqrt


DATA_PATH = "data/processed/clean.csv"
PREPROCESSOR_PATH = "src/features/preprocess_pipeline.joblib"
TARGET = "time_in_hospital"


def train_and_log(model, model_name, pipeline, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):

        full_pipeline = Pipeline(
            steps=[
                ("preprocess", pipeline),
                ("model", model),
            ]
        )

        full_pipeline.fit(X_train, y_train)

        preds = full_pipeline.predict(X_test)

        rmse = sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(full_pipeline, name="model")

        print(f"{model_name} | RMSE: {rmse:.3f} | R2: {r2:.3f}")


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Loading preprocessing pipeline...")
    preprocessor = load(PREPROCESSOR_PATH)

    mlflow.set_experiment("Hospital_LOS_Prediction")

    print("Training models...")

    lr = LinearRegression()
    train_and_log(lr, "LinearRegression", preprocessor, X_train, X_test, y_train, y_test)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    train_and_log(rf, "RandomForest", preprocessor, X_train, X_test, y_train, y_test)

    print("Training complete.")


if __name__ == "__main__":
    main()
>>>>>>> 585363464613ca596c1f9d17ba9b907a36c366b2
