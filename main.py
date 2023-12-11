import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference

# Load the model, encoder, and label binarizer
model_path = os.path.join('model', 'trained_model.joblib')
encoder_path = os.path.join('model', 'encoder.joblib')
lb_path = os.path.join('model', 'label_binarizer.joblib')

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)
lb = joblib.load(lb_path)

# Instantiate the app.
app = FastAPI()

# Define Pydantic model for request body
# Include examples with the `Field` function


class CensusData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status",
                                example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country",
                                example="United-States")


@app.get("/")
async def greeting():
    return {"greeting": "Welcome to the Census Income Prediction API!"}


@app.post("/predict")
async def predict(data: CensusData):
    try:
        # Convert Pydantic model to dict and then to dataframe
        data_dict = data.dict(by_alias=True)
        data_df = pd.DataFrame([data_dict])

        # Process the data
        cat_features = [
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country",
        ]
        X, _, _, _ = process_data(
            data_df,
            categorical_features=cat_features,
            encoder=encoder,
            lb=lb,
            training=False
        )

        # Make prediction
        prediction = inference(model, X)
        pred_label = lb.inverse_transform(prediction)[0]

        return {"prediction": pred_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
