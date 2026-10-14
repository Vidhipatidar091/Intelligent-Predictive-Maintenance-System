# Intelligent Predictive Maintenance System

## 1. Project Overview
This project uses Machine Learning to predict potential failures in industrial equipment using sensor readings.

## 2. Objectives
- Analyze industrial machine sensor data.
- Perform exploratory data analysis.
- Train a machine learning classification model.
- Predict potential machine failures.
- Provide maintenance recommendations through a REST API.
- Deploy the API using Docker.

## 3. Dataset
AI4I 2020 Predictive Maintenance Dataset.

The dataset contains machine operating conditions, sensor readings, and machine failure labels.

## 4. Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- FastAPI
- Docker
- Git and GitHub

## 5. Machine Learning Model
A Random Forest Classifier is used to predict machine failure from machine type, air temperature, process temperature, rotational speed, torque, and tool wear.

## 6. API Features
The prediction API accepts machine sensor readings and returns:
- Machine failure prediction
- Probability of no failure
- Probability of failure
- Maintenance recommendation

## 7. Run Locally

Install the dependencies:

    pip install -r requirements.txt

Start the API:

    python -m uvicorn apps.main:app --reload

Open the API documentation:

    http://127.0.0.1:8000/docs

## 8. Run with Docker

Build the image:

    docker build -t predictive-maintenance .

Start the container:

    docker run -d --name predictive-maintenance-api -p 8000:8000 predictive-maintenance

Open:

    http://localhost:8000/docs

## 9. Conclusion
This project demonstrates how machine learning and API deployment can support predictive maintenance by identifying potential equipment failures from sensor data.