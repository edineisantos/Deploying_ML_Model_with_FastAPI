from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_method():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting":
                               "Welcome to the Census Income Prediction API!"}


def test_post_lower_age_education():
    test_data = {
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

    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}


def test_post_higher_age_education_gain():
    test_data = {
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

    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}
