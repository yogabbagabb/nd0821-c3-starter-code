# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from starter.ml.data import process_data, cat_features
from starter.ml.model import train_model, performance_on_model_slices
from joblib import dump
import pandas as pd


# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv("./data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label='salary', training=True
)
# Proces the test data with the process_data function.

# Train and save a model.
model = train_model(X_train, y_train)

unique_values = data['education'].unique()
with open("./data/slice_output.txt", "w") as f:
    for edu in unique_values:
        performance_across_three_metrics = str(performance_on_model_slices(model, data, 'education', value=edu, lb=lb, encoder=encoder))
        f.write(f"When education is {edu}: ")
        f.write(performance_across_three_metrics)
        f.write("\n")
