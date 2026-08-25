"""Data loading and preprocessing for Heart Disease and Diabetes datasets."""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import requests
import os

HEART_DISEASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
HEART_DISEASE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

DIABETES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
DIABETES_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "payer_code",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
    "readmitted",
]


def load_heart_disease(
    data_dir: str = "data", download: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Heart Disease dataset (Cleveland).

    Returns:
        X: Features (n_samples, 13)
        y: Binary targets (0 = no disease, 1 = disease)
    """
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "heart_disease.csv")

    if download and not os.path.exists(filepath):
        print("Downloading Heart Disease dataset...")
        try:
            response = requests.get(HEART_DISEASE_URL, timeout=30)
            with open(filepath, "w") as f:
                f.write(response.text)
            print(f"Saved to {filepath}")
        except Exception as e:
            print(f"Download failed: {e}")
            raise

    df = pd.read_csv(filepath, names=HEART_DISEASE_COLUMNS, na_values="?")

    df = df.dropna()

    y = (df["target"] > 0).astype(int).values
    X = df.drop("target", axis=1).values.astype(float)

    return X, y


def load_diabetes(
    data_dir: str = "data", download: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Diabetes 130-US Hospitals dataset.

    Returns:
        X: Features
        y: Binary targets (readmitted: <30 days = 1, >30/NO = 0)
    """
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "diabetes.csv")

    if download and not os.path.exists(filepath):
        print("Downloading Diabetes dataset...")
        print("Note: This dataset is large. Please download manually from:")
        print(
            "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008"
        )
        print("And place as data/diabetes.csv")
        raise FileNotFoundError("Diabetes dataset not found. Please download manually.")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Diabetes dataset not found at {filepath}")

    df = pd.read_csv(filepath)

    df = df.replace("?", np.nan)

    drop_cols = [
        "encounter_id",
        "patient_nbr",
        "weight",
        "payer_code",
        "medical_specialty",
        "diag_1",
        "diag_2",
        "diag_3",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    imputer = SimpleImputer(strategy="median")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    y = (df_imputed["readmitted"] == "<30").astype(int).values
    X = df_imputed.drop("readmitted", axis=1).values.astype(float)

    return X, y


def preprocess_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    Preprocess data: split and scale.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def one_hot_encode(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot encoding."""
    return np.eye(n_classes)[y]


def get_data_info(X: np.ndarray, y: np.ndarray) -> dict:
    """Get dataset information."""
    return {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "classes": np.unique(y).tolist(),
        "class_distribution": {int(c): int(np.sum(y == c)) for c in np.unique(y)},
        "feature_stats": {
            "mean": X.mean(axis=0).tolist(),
            "std": X.std(axis=0).tolist(),
            "min": X.min(axis=0).tolist(),
            "max": X.max(axis=0).tolist(),
        },
    }
