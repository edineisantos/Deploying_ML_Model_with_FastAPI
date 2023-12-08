import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Adjust the path for importing from ml subfolder
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'ml')))

from data import process_data
from model import compute_model_metrics, inference

def compute_metrics_by_category(X, y, model, feature_name, encoder, lb, 
cat_features, file):
    """
    Computes performance metrics for each data slice based on unique values 
    of a given categorical feature.
    """
    unique_values = X[feature_name].unique()
    for value in unique_values:
        X_slice = X[X[feature_name] == value]
        y_slice = y[X[feature_name] == value]

        # Process the sliced data with all categorical features
        X_processed, _, _, _ = process_data(
            X_slice.drop("salary", axis=1), categorical_features=cat_features, 
            label=None, training=False, encoder=encoder, lb=lb
        )

        # Binarize y_slice to match the format of y_pred
        y_slice = lb.transform(y_slice)
        
        # Compute model metrics
        preds = inference(model, X_processed)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)

        print(f"Metrics for {feature_name} = {value}:", file=file)
        print(f"  Precision: {precision}", file=file)
        print(f"  Recall: {recall}", file=file)
        print(f"  F1-Score: {fbeta}\n", file=file)

def main():
    # Define categorical features
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]

    # Define paths
    model_path = os.path.join(os.getcwd(), '..', 'model', 'trained_model.joblib')
    encoder_path = os.path.join(os.getcwd(), '..', 'model', 'encoder.joblib')
    lb_path = os.path.join(os.getcwd(), '..', 'model', 'label_binarizer.joblib')
    output_path = os.path.join(os.getcwd(), '..', 'model', 'slice_output.txt')

    # Load model, encoder, and label binarizer
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)

    # Load test data
    data_file = os.path.join(os.getcwd(), '..', 'data', 'cleaned_census.csv')
    data = pd.read_csv(data_file)
    _, test = train_test_split(data, test_size=0.20)

# Compute and save metrics by category
    with open(output_path, 'w') as file:
        for feature in cat_features:
            print(f"=== Metrics for {feature} ===", file=file)
            compute_metrics_by_category(
                test, test["salary"], model, feature, encoder, lb, 
                cat_features, file
            )

if __name__ == "__main__":
    main()