import requests

# Replace the following URL with the actual API endpoint
api_url = "https://nd0821-c3-starter-code-0icr.onrender.com/predict/"

# Replace the payload_data with your JSON payload
payload_data = {
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

# Make a POST request with JSON payload
response = requests.post(api_url, json=payload_data)

# Check the status code and print the response
if response.status_code == 200:
    print("Request successful. Response:")
    print(response.json())
else:
    print(f"Request failed with status code {response.status_code}. Response text:")
    print(response.text)
