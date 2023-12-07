import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Create and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

     # Define the path for saving the model
    # Assuming the script is executed from the 'starter' directory
    model_directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'model'))
    os.makedirs(model_directory, exist_ok=True)
    model_path = os.path.join(model_directory, 'trained_model.joblib')

    # Save the model
    joblib.dump(model, model_path)
    print(f"Model trained and saved at '{model_path}'")

    # Return the trained model
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall,
    and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)