import pytest
from httpx import AsyncClient
from main import app

# Test with lower age and education go get a prediction of <=50K
@pytest.mark.asyncio
async def test_post_lower_age_education():
    test_data = {
        "age": 20,  # Lower age
        "workclass": "Private",
        "fnlgt": 50000,
        "education": "HS-grad",  # Lower education
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

    async with AsyncClient(app=app, base_url="http://127.0.0.1:8890") as ac:
        response = await ac.post("/predict", json=test_data)
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}

# Test with higher age, education, and capital gain to get a prediction of >50K
@pytest.mark.asyncio
async def test_post_higher_age_education_gain():
    test_data = {
        "age": 50,  # Higher age
        "workclass": "Self-emp-not-inc",
        "fnlgt": 150000,
        "education": "Masters",  # Higher education
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 10000,  # Higher capital gain
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }

    async with AsyncClient(app=app, base_url="http://127.0.0.1:8890") as ac:
        response = await ac.post("/predict", json=test_data)
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}
