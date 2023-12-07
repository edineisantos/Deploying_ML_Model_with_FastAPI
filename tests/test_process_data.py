import os
import sys
import pytest
import pandas as pd
import numpy as np

# Adjusting the path for 'data.py' import
ml_dir = os.path.join(os.path.dirname(__file__), '..', 'starter', 'ml')
sys.path.insert(0, os.path.abspath(ml_dir))

from data import process_data


def test_process_data():
    # Create a sample DataFrame
    data = pd.DataFrame({
        'feature1': ['A', 'B', 'C'],
        'feature2': [1, 2, 3],
        'label': [0, 1, 0]
    })

    categorical_features = ['feature1']
    label = 'label'

    # Process the data
    X, y, encoder, lb = process_data(
        data, categorical_features=categorical_features, label=label, training=True
    )

    # Check the shape of the output arrays
    # Same number of rows
    assert X.shape[0] == data.shape[0]
    # Number of unique values in 'feature1'
    num_categorical_cols = len(encoder.categories_[0])
    # One-hot columns + 1 numeric column
    assert X.shape[1] == num_categorical_cols + 1

    # Check if label binarizer works
    assert set(y) == {0, 1}    # Labels should be binary (0 or 1)

    # Check encoder and lb are not None
    assert encoder is not None
    assert lb is not None
