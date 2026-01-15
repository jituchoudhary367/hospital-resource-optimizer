import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from joblib import dump


DATA_PATH = "data/processed/clean.csv"
PIPELINE_PATH = "src/features/preprocess_pipeline.joblib"


NUM_FEATURES = [
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

CAT_FEATURES = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "change",
    "diabetesMed",
]

TARGET = "time_in_hospital"


def build_pipeline():
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUM_FEATURES),
            ("cat", cat_pipeline, CAT_FEATURES),
        ]
    )

    return preprocessor


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Building preprocessing pipeline...")
    pipeline = build_pipeline()

    print("Fitting pipeline on training data...")
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    print("Saving pipeline...")
    dump(pipeline, PIPELINE_PATH)

    print("Done.")
    print("Train shape:", X_train_transformed.shape)
    print("Test shape:", X_test_transformed.shape)


if __name__ == "__main__":
    main()
