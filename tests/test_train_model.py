import os
import joblib
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from starter.ml.model import train_model


def test_train_model():
    # Create a dummy dataset
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    # Define a custom model name
    custom_model_name = "test_model"

    # Train the model with the custom name
    model = train_model(X, y, model_name=custom_model_name)

    # Check if the model is an instance of LogisticRegression
    assert isinstance(model, LogisticRegression)

    # Check if the custom model file was saved
    model_directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'model'))
    custom_model_path = os.path.join(model_directory,
                                     f'{custom_model_name}.joblib')
    assert os.path.isfile(custom_model_path)

    # Load the saved model and make a prediction to ensure it's working
    loaded_model = joblib.load(custom_model_path)
    preds = loaded_model.predict(X)
    assert len(preds) == 100

    # Clean up: Remove the saved custom model file after the test
    os.remove(custom_model_path)
