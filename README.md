# ensoML Backend

**ensoML** (“Enso” is a Zen symbol for completeness and harmony) is a beginner-friendly, no-code AutoML platform. It empowers users to upload tabular datasets, generate rich EDA reports, and train optimized machine learning models—all without writing code. The backend is built with FastAPI, PyCaret, YData Profiling, and MongoDB, and is designed for robust, multi-user, production deployment.

## Features

- **No-code ML**: Train, evaluate, and export models with zero code.
- **EDA Reports**: Generate interactive, shareable EDA reports using YData Profiling.
- **Model Training**: Supports classification and regression with PyCaret, including XGBoost, LightGBM, and traditional models.
- **Evaluation Plots**: Auto-generates confusion matrices, ROC curves, feature importances, and more.
- **Per-user Management**: Each user’s models, reports, and predictions are isolated and managed.
- **Cleanup & Maintenance**: Automated and manual cleanup endpoints for user and system data.
- **Modern API**: Built with FastAPI, async MongoDB, and Pydantic schemas for robust validation and serialization.
- **Production-ready**: Deployable via Docker on Render or any cloud provider.

---

## API Overview

### User Management

- `POST /user/init`  
  Initialize a new user.  
  **Request:**  
  ```json
  {
    "user_id": "string",
    "email": "user@example.com",
    "name": "Optional Name"
  }
  ```
  **Response:** User details and creation timestamp.

---

### EDA (Exploratory Data Analysis)

- `POST /eda/generate`  
  Upload a CSV and generate an EDA report.  
  **Form Data:**  
  - `file`: CSV file  
  - `user_id`: string  
  - `dataset_name`: string (optional)  
  **Response:**  
  - Report filename, URL, dataset info, file size

- `GET /eda/view/{filename}`  
  View EDA report as HTML.

- `GET /eda/download/{filename}`  
  Download EDA report.

- `GET /eda/list/{user_id}`  
  List all EDA reports for a user.

- `GET /eda/history/{user_id}`  
  Get EDA generation history for a user.

- `DELETE /eda/delete/{filename}`  
  Delete a specific EDA report.

---

### Model Training & Management

- `POST /model/train`  
  Train a model on an uploaded CSV.  
  **Form Data:**  
  - `file`: CSV file  
  - `user_id`: string  
  - `target_column`: string  
  - `dataset_name`: string (optional)  
  - `model_types`: comma-separated string (optional, e.g. `"knn,rf,xgboost"`)  
  **Response:**  
  - Model filename, download URL, metrics, plot URLs, training time

- `GET /model/list/{user_id}`  
  List all models for a user.

- `GET /model/download/{filename}`  
  Download a trained model file.

- `GET /model/metrics/{filename}`  
  Get detailed metrics for a model.

- `GET /model/plots/{filename}`  
  Get all plot URLs for a model.

- `POST /model/compare`  
  Compare multiple models for a user.  
  **Request:**  
  ```json
  {
    "user_id": "string",
    "model_filenames": ["model1.pkl", "model2.pkl"]
  }
  ```
  **Response:**  
  - Comparison statistics, best model, per-model details

- `GET /model/validate/{filename}`  
  Validate that a model file is accessible and loadable.

- `DELETE /model/delete/{filename}`  
  Delete a model and its metadata.

---

### Prediction

- `POST /model/predict`  
  Make predictions using a trained model.  
  **Request:**  
  ```json
  {
    "user_id": "string",
    "model_filename": "model.pkl",
    "input_data": {
      "feature1": value,
      "feature2": value
    }
  }
  ```
  **Response:**  
  - Predictions, probabilities (if applicable), model used, input features

---

### Dataset Validation

- `POST /model/validate-dataset`  
  Validate a dataset and target column before training.  
  **Form Data:**  
  - `file`: CSV file  
  - `target_column`: string  
  **Response:**  
  - Dataset info, warnings, errors

---

### Plots

- `GET /plots/list/{user_id}`  
  List all plots for a user.

- `GET /plots/{filename}`  
  View/download a plot image.

- `GET /plots/by-model/{model_filename}`  
  Get all plots for a specific model.

- `GET /plots/by-type/{user_id}/{plot_type}`  
  Get all plots of a specific type for a user.

- `DELETE /plots/delete/{filename}`  
  Delete a specific plot.

- `DELETE /plots/delete-user/{user_id}`  
  Delete all plots for a user.

---

### Cleanup & Maintenance

- `POST /cleanup/user/{user_id}`  
  Delete all files and records for a user (requires confirmation).

- `POST /cleanup/system`  
  Delete files older than a specified number of hours.

- `POST /cleanup/orphaned`  
  Remove database records without corresponding files.

- `GET /cleanup/status`  
  Get cleanup statistics.

- `GET /cleanup/logs`  
  Get recent cleanup logs.

---

### Health & Root

- `GET /`  
  API root info.

- `GET /health`  
  Health check for database and storage.

---

## Schemas

- All endpoints use strict Pydantic schemas for request and response validation.
- See request_schemas.py and response_schemas.py for details.

---

## Tech Stack

- **FastAPI**: High-performance async API framework
- **PyCaret**: Automated machine learning (AutoML)
- **YData Profiling**: EDA report generation
- **MongoDB**: Async document database
- **Docker**: Containerized deployment
- **Render**: Cloud hosting (or any Docker-compatible provider)

---

## Getting Started

1. Clone the repo and install dependencies.
2. Set up your .env or config.py for MongoDB and storage paths.
3. Run with Docker or `uvicorn main:app --reload`.
4. Access the API docs at `/docs`.

---

## Philosophy

**ensoML** is inspired by the Zen concept of the Enso: a circle of togetherness, completeness, and simplicity. Our goal is to make machine learning accessible, transparent, and harmonious for everyone—no code required.

