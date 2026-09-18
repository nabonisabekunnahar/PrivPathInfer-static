"""
datasets.py — dataset loaders for the fidelity / storage / subtree
experiments. Every loader returns (X, y) as numpy arrays. If a
required file is missing, raise a clear error naming the file and
where to put it — never silently substitute different data.
"""

import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


def _require_file(path, hint):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing dataset file: {path}\n{hint}"
        )


def _load_pima(data_dir):
    path = os.path.join(data_dir, "diabetes.csv")
    _require_file(
        path,
        "Place the PIMA Indians Diabetes CSV (768 rows, 8 features, "
        "'Outcome' column) at data/diabetes.csv.",
    )
    df = pd.read_csv(path)
    y = df["Outcome"].to_numpy()
    X = df.drop(columns=["Outcome"]).to_numpy(dtype=float)
    return X, y


def _load_breast_cancer():
    data = load_breast_cancer()
    return data.data, data.target


HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

UCI_HEART_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


def _load_heart(data_dir):
    path = os.path.join(data_dir, "processed.cleveland.data")
    if not os.path.isfile(path):
        try:
            import urllib.request
            os.makedirs(data_dir, exist_ok=True)
            urllib.request.urlretrieve(UCI_HEART_URL, path)
        except Exception as exc:
            raise FileNotFoundError(
                f"Missing dataset file: {path}\n"
                f"Could not fetch it from {UCI_HEART_URL} either "
                f"({exc}). Place the UCI Heart Disease Cleveland "
                f"processed file at data/processed.cleveland.data."
            ) from exc

    df = pd.read_csv(path, names=HEART_COLUMNS, na_values="?")
    # Impute missing values (present only in 'ca' and 'thal') with the
    # column median so all 303 records are retained, matching standard
    # practice for this dataset.
    for col in ["ca", "thal"]:
        df[col] = df[col].fillna(df[col].median())

    y = (df["num"].to_numpy(dtype=int) > 0).astype(int)  # binarize: disease present/absent
    X = df.drop(columns=["num"]).to_numpy(dtype=float)
    return X, y


def _load_framingham(data_dir):
    path = os.path.join(data_dir, "framingham.csv")
    _require_file(
        path,
        "Place the Framingham CHD risk CSV at data/framingham.csv "
        "('TenYearCHD' column, ~4,238 raw rows).",
    )
    df = pd.read_csv(path)
    df = df.dropna()
    y = df["TenYearCHD"].to_numpy(dtype=int)
    X = df.drop(columns=["TenYearCHD"]).to_numpy(dtype=float)
    return X, y


def load_dataset(name, data_dir="data"):
    """
    Load one of "pima", "breast_cancer", "heart", "framingham".

    Returns:
        (X, y): numpy arrays
    """
    if name == "pima":
        return _load_pima(data_dir)
    if name == "breast_cancer":
        return _load_breast_cancer()
    if name == "heart":
        return _load_heart(data_dir)
    if name == "framingham":
        return _load_framingham(data_dir)
    raise ValueError(f"Unknown dataset: {name!r}")


DATASET_NAMES = ("pima", "breast_cancer", "heart", "framingham")
