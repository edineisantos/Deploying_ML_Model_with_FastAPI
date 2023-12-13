import requests

# URL of the deployed API
api_url = "https://ml-fast-api-deploy.uc.r.appspot.com/predict"
print(f"API URL: {api_url}")
print("...................")

# Test data for the POST requests
test_data_1 = {
    "age": 20,
    "workclass": "Private",
    "fnlgt": 50000,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Never-married",
    "occupation": "Handlers-cleaners",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

test_data_2 = {
    "age": 50,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 150000,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 10000,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States"
}

# Function to send POST request and print the result


def send_request(data):
    response = requests.post(api_url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")


# Send requests with both test data
print("Sending request with lower age and education...")
send_request(test_data_1)

print("\nSending request with higher age, education, and capital gain...")
send_request(test_data_2)
