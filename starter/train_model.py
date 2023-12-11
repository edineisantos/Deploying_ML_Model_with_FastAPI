# Script to train machine learning model.
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference


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

    # Define the path for saving the model, encoder, and label binarizer
    # Assuming the script is executed from the 'starter' directory
    model_directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'model'))
    os.makedirs(model_directory, exist_ok=True)

    # Paths for the encoder and label binarizer
    encoder_path = os.path.join(model_directory, 'encoder.joblib')
    lb_path = os.path.join(model_directory, 'label_binarizer.joblib')

    # Save the model, encoder, and label binarizer
    joblib.dump(encoder, encoder_path)
    joblib.dump(lb, lb_path)

    # Train and save the model
    model = train_model(X_train, y_train)
    print("Model training complete.")

    # Process the test data
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary",
        training=False, encoder=encoder, lb=lb
    )

    # Make predictions on the test data
    preds = inference(model, X_test)

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    # Output path for the metrics
    metrics_output_path = os.path.join(model_directory, 'metrics_output.txt')

    # Write the metrics to the file
    with open(metrics_output_path, 'w') as file:
        print(" === Metrics for trained model === ", file=file)
        file.write("Precision: {precision}\n")
        file.write("Recall: {recall}\n")
        file.write("F1-Score: {fbeta}\n")

    print("Model metrics saved to 'metrics_output.txt'.")


if __name__ == "__main__":
    main()
