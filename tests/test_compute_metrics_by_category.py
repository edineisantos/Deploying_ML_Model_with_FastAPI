import pytest
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from io import StringIO
import os
import sys

# Adjusting the path for 'data.py' import
starter_dir = os.path.join(os.path.dirname(__file__), '..', 'starter')
sys.path.insert(0, os.path.abspath(starter_dir))
ml_dir = os.path.join(os.path.dirname(__file__), '..', 'starter', 'ml')
sys.path.insert(0, os.path.abspath(ml_dir))

from data import process_data
from model import compute_model_metrics, inference
from compute_metrics_by_category import compute_metrics_by_category

def test_compute_metrics_by_category():
    # Define paths
    model_path = os.path.join(os.getcwd(), '..', 'model', 'trained_model.joblib')
    encoder_path = os.path.join(os.getcwd(), '..', 'model', 'encoder.joblib')
    lb_path = os.path.join(os.getcwd(), '..', 'model', 'label_binarizer.joblib')
    data_file = os.path.join(os.getcwd(), '..', 'data', 'cleaned_census.csv')

    # Load model, encoder, and label binarizer
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)

    # Load test data
    data = pd.read_csv(data_file)
    _, test = train_test_split(data, test_size=0.20, random_state=42)

    # Define categorical features
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]

    # Redirect output to a string buffer
    buffer = StringIO()
    for feature in cat_features:
        compute_metrics_by_category(
            test, test["salary"], model, feature, encoder, lb, 
            cat_features, buffer
        )

    # Check if metrics are computed and printed (basic check)
    buffer.seek(0)
    output = buffer.read()
    assert 'Metrics for' in output
    assert 'Precision' in output
    assert 'Recall' in output
    assert 'F1-Score' in output
