# Deploying ML Model with FastAPI 

## Project Overview

The Deploying ML Model with FastAPI is a RESTful service developed as part of the ML DevOps Engineer Nanodegree at Udacity. This project utilizes a machine learning model to predict income levels based on census data. The API, built with FastAPI, serves as an interface for model inference, allowing users to submit demographic data and receive income predictions.This project is intended for educational purposes and is not recommended for production use. It serves as a demonstration of integrating machine learning models with web APIs and deploying them in a cloud environment.

## Objectives

The key goals of this project include:

- Developing a RESTful API using FastAPI for model inference.
- Adhering to best practices in Python programming and API development.
- Implementing comprehensive testing for both the machine learning model and the API.
- Demonstrating continuous integration and continuous deployment (CI/CD) with GitHub Actions and Google Cloud Platform (GCP).
- Showcasing skills in handling and deploying machine learning models in a production environment.

## File Structure Overview

The project is structured as follows:

- `main.py`: The FastAPI application code.
- `requirements.txt`: Lists the dependencies required to run the project.
- `app.yaml`: Configuration file for deploying the application on GCP's App Engine.
- `Dockerfile`: Instructions for building a Docker image of the project.
- `tests/`: Contains test scripts for the API.
- `starter/`: A Python package with utility functions for data processing and model inference.
- `model/`: Directory containing the trained model, encoder, and label binarizer.
- `data/`: Contains the dataset used in the project, providing census information for model training and testing.
- `screenshots/`: Includes images showcasing the application running in production and the successful execution of the CI/CD pipeline on GitHub Actions.
- `model_card.md`: A detailed document explaining the data used, model performance, ethical considerations, and limitations.
- `api_post_request.py`: A script for sending POST requests to the deployed API to test model predictions in a production environment.
- `environment.yml`: A Conda environment file for setting up a development environment with all required dependencies.


## Running the Project

### Local Setup

1. Clone the repository to your local machine.
2. Navigate to the project directory and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the FastAPI application locally:
   ```bash
   uvicorn main:app --reload
   ```

### Using Docker

To run the "Deploying ML Model with FastAPI" project using Docker, follow these steps:

1. Build the Docker image:
   ```bash
   docker build -t ml_model_fast_api_image .
   ```

2. Run the Docker container, ensuring the correct port is exposed. The default port in the Dockerfile is set to 8890, but you can modify it as needed:
   ```bash
   docker run -p 8890:8890 --name ml_model_fast_api ml_model_fast_api_image
   ```

3. Inside the Docker container, start the FastAPI application with the following command. This command uses Uvicorn to serve the app on the specified host and port. The `--reload` flag is optional and enables auto-reloading of the server on code changes:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8890
   ```

This setup allows you to run the FastAPI application in a Dockerized environment, ensuring consistency across different machines and ease of deployment. Remember to adjust the port numbers in both the Docker run command and the Uvicorn command if you change the exposed port in the Dockerfile.

### API Endpoints

- `GET /`: Returns a welcome message.
- `POST /predict`: Accepts JSON data for making income predictions.

### Example JSON for POST /predict

```json
{
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}
```

## Continuous Deployment

The project employs GitHub Actions for CI/CD, automatically deploying the application to Google Cloud's App Engine upon successful merging of changes into the master branch.