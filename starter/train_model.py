# Script to train machine learning model.

import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

# Adjust the path for importing from ml subfolder
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'ml')))

from data import process_data
from model import train_model

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters
    ----------
    file_path : str
        The path to the CSV file.

    Returns
    -------
    data : DataFrame
        The loaded data.
    """
    data = pd.read_csv(file_path)
    return data

def main():
    # Path to the cleaned census data file
    data_file = os.path.join(os.getcwd(), '..', 'data', 'cleaned_census.csv')
    
    # Load the data
    data = load_data(data_file)

    # Split the data into training and test sets
    train, test = train_test_split(data, test_size=0.20)

    # Define categorical features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Process the training data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Train and save the model
    model = train_model(X_train, y_train)
    print("Model training complete.")

if __name__ == "__main__":
    main()
