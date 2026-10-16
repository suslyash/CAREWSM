#!/bin/bash
source ".venv/bin/activate"

mlflow ui --backend-store-uri sqlite:///logs/mlflow.db

# mlflow gc --tracking-uri sqlite:///logs/mlflow.db --backend-store-uri sqlite:///logs/mlflow.db