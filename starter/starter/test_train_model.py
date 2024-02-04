import numpy as np
import pytest
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def setup():
    data = pd.read_csv("../data/census.csv")

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, _ = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    yield X_train, y_train, encoder, lb


def test_train_model(setup):
    X_train, y_train, _, _ = setup
    model = train_model(X_train, y_train)
    assert type(model) is LogisticRegression


def test_compute_model_metrics(setup):
    X_train, y_train, _, _ = setup
    m1, m2, m3 = compute_model_metrics(y_train, y_train)
    assert type(m1) is np.float64


def test_inference(setup):
    X_train, y_train, _, _ = setup
    model = train_model(X_train, y_train)
    assert type(inference(model, X_train)) is np.ndarray
