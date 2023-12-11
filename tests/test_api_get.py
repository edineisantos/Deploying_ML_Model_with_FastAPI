from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)

def test_get_method():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting": 
    "Welcome to the Census Income Prediction API!"}

