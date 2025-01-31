# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference
from joblib import dump
import pandas as pd


# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv("../data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Proces the test data with the process_data function.

# Train and save a model.
model = train_model(X_train, y_train)
dump(model, "model.joblib")
dump(encoder, "encoder.joblib")
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, lb=lb, encoder=encoder
)
y_predict = inference(model, X_test)
print(f"These are the model metrics {compute_model_metrics(y_test, y_predict)}")
