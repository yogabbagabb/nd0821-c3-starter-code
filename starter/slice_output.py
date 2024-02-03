# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, performance_on_model_slices
from joblib import dump
import pandas as pd


# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv("./data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
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
    train, categorical_features=cat_features, label='salary', training=True
)
X_test, y_test, _, _ = process_data(
    data, categorical_features=cat_features, label='salary', training=True, encoder=encoder, lb=lb
)

# Proces the test data with the process_data function.

# Train and save a model.
model = train_model(X_train, y_train)

print(performance_on_model_slices(model, X_test, 'education', value=encoder.transform()))
