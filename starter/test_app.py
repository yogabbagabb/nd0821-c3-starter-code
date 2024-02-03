from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}


def test_get_zero_label():
    r = client.post("/predict/", json=
        {
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
        }
    )
    assert r.status_code == 200
    assert r.json() == {"value": "[0]"}


def test_get_one_label():
    r = client.post("/predict/", json=
    {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Doctorate",
        "education_num": 13,
        "marital_status": "Married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "Asian",
        "sex": "Female",
        "capital_gain": 112174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
                    )
    assert r.status_code == 200
    assert r.json() == {"value": "[1]"}
