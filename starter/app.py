from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from starter.ml.data import cat_features, process_data
import numpy as np

# Instantiate the app.
app = FastAPI()


class Record(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: int
    native_country: str


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


model = joblib.load("starter/model.joblib")
encoder = joblib.load("starter/encoder.joblib")


@app.post("/predict/")
async def items(record: Record):
    record_dict = record.dict()
    record_dict['education-num'] = record_dict.pop('education_num')
    record_dict['marital-status'] = record_dict.pop('marital_status')
    record_dict['capital-gain'] = record_dict.pop('capital_gain')
    record_dict['capital-loss'] = record_dict.pop('capital_loss')
    record_dict['hours-per-week'] = record_dict.pop('hours_per_week')
    record_dict['native-country'] = record_dict.pop('native_country')
    columns = ["age", "workclass", "fnlgt", "education", "education-num", "marital-status", "occupation",
               "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
    df = pd.DataFrame(columns=columns)
    df.loc[len(df)] = record_dict
    X, _, _, _ = process_data(df, categorical_features=cat_features, label=None, training=False, encoder=encoder)
    return {"value": str(model.predict(X))}

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
