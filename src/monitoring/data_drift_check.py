import pandas as pd
import mlflow

TRAIN_PATH = "data/processed/clean.csv"
LIVE_PATH = "data/processed/live_inputs.csv"


def main():
    train = pd.read_csv(TRAIN_PATH)
    live = pd.read_csv(LIVE_PATH)

    drift_flags = {}

    numeric_cols = train.select_dtypes(exclude="object").columns

    for col in numeric_cols:
        train_mean = train[col].mean()
        live_mean = live[col].mean()

        diff = abs(train_mean - live_mean)
        threshold = train[col].std()

        drift_flags[col] = diff > threshold

    drifted = sum(drift_flags.values())

    with mlflow.start_run(run_name="data_drift_check"):
        mlflow.log_metric("drifted_features", drifted)

        for k, v in drift_flags.items():
            mlflow.log_metric(f"drift_{k}", int(v))

    print("Drifted numeric features:", drifted)
    for k, v in drift_flags.items():
        if v:
            print("Drift detected in:", k)


if __name__ == "__main__":
    main()
