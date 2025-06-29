"""
Model training service using PyCaret for AutoML operations.
"""

import time
import pickle
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from fastapi import UploadFile, HTTPException
import pycaret.classification as pc_clf
import pycaret.regression as pc_reg

from config import settings
from db.mongodb import mongodb
from db.models import ModelJob, Prediction
from schemas.request_schemas import ModelTrainRequest, PredictionRequest
from utils.file_utils import FileManager
from utils.naming import NamingUtils
from services.plot_service import PlotService

logger = logging.getLogger(__name__)


class TrainService:
    """Service for handling model training operations with PyCaret."""
    def __init__(self):
        self.db = mongodb
        self.plot_service = PlotService()
    
    async def train_model(self, file: UploadFile, request: ModelTrainRequest) -> Dict[str, Any]:
        """Train ML model using PyCaret and save results."""
        start_time = time.time()
        
        try:
            # Save uploaded file temporarily
            temp_filename = NamingUtils.generate_temp_filename(request.user_id, file.filename)
            temp_filepath = settings.storage_dir / "temp" / temp_filename
            temp_filepath.parent.mkdir(parents=True, exist_ok=True)
            
            await FileManager.save_uploaded_file(file, temp_filepath)
            
            # Read and validate dataset
            df = await FileManager.read_csv_file(temp_filepath)
            
            # Validate target column
            if request.target_column not in df.columns:
                raise ValueError(f"Target column '{request.target_column}' not found in dataset")
            
            # Determine problem type
            problem_type = self._determine_problem_type(df, request.target_column)
            
            # Clean dataset
            df_clean = self._preprocess_dataset(df, request.target_column)
            df_features = df_clean.drop(request.target_column, axis='columns')
            
            # Setup PyCaret environment
            if problem_type == "classification":
                pc_clf.setup(
                    df_clean,
                    target=request.target_column,
                    session_id=123,
                    train_size=0.8,
                    verbose=False,
                    use_gpu=False
                )
                pycaret_module = pc_clf
            else:
                pc_reg.setup(
                    df_clean,
                    target=request.target_column,
                    session_id=123,
                    train_size=0.8,
                    verbose=False,
                    use_gpu=False
                )
                pycaret_module = pc_reg
            
            # Train models with limited selection for performance
            model_types = request.model_types or settings.PYCARET_LIGHTWEIGHT_MODELS
            models = pycaret_module.compare_models(
                include=model_types,
                turbo=settings.PYCARET_TURBO_MODE,
                sort='Accuracy' if problem_type == "classification" else 'MAE',
                n_select=1,  # Only select best model
                verbose=False
            )
            
            # Get best model (if multiple returned, take first)
            best_model = models if not isinstance(models, list) else models[0]
            
            # Finalize model
            final_model = pycaret_module.finalize_model(best_model)
            
            # Generate model filename
            model_filename = NamingUtils.generate_model_filename(
                request.user_id, 
                request.dataset_name
            )
            model_filepath = settings.models_dir / model_filename
            # Ensure the models directory exists
            model_filepath.parent.mkdir(parents=True, exist_ok=True)
            # Save model
            with open(model_filepath, 'wb') as f:
                pickle.dump(final_model, f)
            
            # Get model metrics
            metrics = self._extract_model_metrics(pycaret_module, problem_type)
            
            # Generate evaluation plots
            plot_urls = await self._generate_evaluation_plots(
                pycaret_module, 
                request.user_id, 
                model_filename.split('.')[0],
                problem_type
            )
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Store job in database
            model_job = ModelJob(
                user_id=request.user_id,
                filename=model_filename,
                dataset_name=request.dataset_name,
                target_column=request.target_column,
                model_type=problem_type,
                best_model=str(type(best_model).__name__),
                best_model_score=metrics.get('best_score', 0.0),
                metrics=metrics,
                plot_filenames=[url.split('/')[-1] for url in plot_urls],
                feature_names=(df_features.columns),
                dataset_rows=len(df_clean),
                dataset_columns=len(df_clean.columns),
                training_time=training_time
            )
            model_jobs_collection = self.db.get_collection("model_jobs")
            await model_jobs_collection.insert_one(model_job.model_dump(by_alias=True))
            
            # Clean up temporary file
            if temp_filepath.exists():
                temp_filepath.unlink()
            
            # Generate download URL
            download_url = f"/api/v1/models/download/{model_filename}"
            
            return {
                "filename": model_filename,
                "download_url": download_url,
                "dataset_name": request.dataset_name,
                "target_column": request.target_column,
                "model_type": problem_type,
                "best_model": str(type(best_model).__name__),
                "best_model_score": metrics.get('best_score', 0.0),
                "metrics": metrics,
                "plot_urls": plot_urls,
                "training_time": training_time
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            # Clean up temporary file on error
            if 'temp_filepath' in locals() and temp_filepath.exists():
                temp_filepath.unlink()
            raise e
    
    async def predict(self, request: PredictionRequest) -> Dict[str, Any]:
        """Make predictions using a trained model."""
        try:
            # Find model file
            model_filepath = settings.models_dir / request.model_filename
            if not model_filepath.exists():
                raise HTTPException(status_code=404, detail="Model file not found")
            
            # Load model
            with open(model_filepath, 'rb') as f:
                model = pickle.load(f)
            
            # Convert input data to DataFrame
            input_df = pd.DataFrame([request.input_data])
            
            # Make predictions
            predictions = model.predict(input_df).tolist()
            
            # Try to get prediction probabilities (classification only)
            prediction_probabilities = None
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_df)
                    prediction_probabilities = proba.tolist()
            except Exception:
                pass  # Probabilities not available
            
            # Store prediction in database
            prediction_record = Prediction(
                user_id=request.user_id,
                model_filename=request.model_filename,
                input_data=request.input_data,
                predictions=predictions,
                prediction_probabilities=prediction_probabilities
            )
            predictions_collection = self.db.get_collection("predictions")
            await predictions_collection.insert_one(prediction_record.model_dump(by_alias=True))
            
            return {
                "predictions": predictions,
                "prediction_probabilities": prediction_probabilities,
                "model_filename": request.model_filename,
                "input_data": request.input_data
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise e
    
    def _determine_problem_type(self, df: pd.DataFrame, target_column: str) -> str:
        """Determine if problem is classification or regression."""
        target_series = df[target_column]
        
        # Check if target is numeric
        if pd.api.types.is_numeric_dtype(target_series):
            # Check number of unique values
            unique_count = target_series.nunique()
            total_count = len(target_series)
            
            # If unique values are less than 10% of total or less than 20, treat as classification
            if unique_count <= 20 or (unique_count / total_count) < 0.1:
                return "classification"
            else:
                return "regression"
        else:
            # Non-numeric targets are classification
            return "classification"
    
    def _preprocess_dataset(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Basic preprocessing of the dataset."""
        df_clean = df.copy()
        
        # Remove rows with missing target values
        df_clean = df_clean.dropna(subset=[target_column])
        
        # Limit dataset size if too large
        if len(df_clean) > settings.MAX_DATASET_ROWS:
            df_clean = df_clean.sample(n=settings.MAX_DATASET_ROWS, random_state=42)
        
        return df_clean
    
    def _extract_model_metrics(self, pycaret_module, problem_type: str) -> Dict[str, Any]:
        """Extract metrics from PyCaret model."""
        try:
            # Pull metrics from PyCaret
            metrics_df = pycaret_module.pull()
            
            if problem_type == "classification":
                best_score = float(metrics_df.iloc[0]['Accuracy'])
                metrics = {
                    'best_score': best_score,
                    'accuracy': float(metrics_df.iloc[0]['Accuracy']),
                    'precision': float(metrics_df.iloc[0]['Prec.']),
                    'recall': float(metrics_df.iloc[0]['Recall']),
                    'f1': float(metrics_df.iloc[0]['F1']),
                    'auc': float(metrics_df.iloc[0]['AUC'])
                }
            else:
                best_score = float(metrics_df.iloc[0]['MAE'])
                metrics = {
                    'best_score': best_score,
                    'mae': float(metrics_df.iloc[0]['MAE']),
                    'mse': float(metrics_df.iloc[0]['MSE']),
                    'rmse': float(metrics_df.iloc[0]['RMSE']),
                    'r2': float(metrics_df.iloc[0]['R2'])
                }
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to extract metrics: {e}")
            return {"best_score": 0.0}
    
    async def _generate_evaluation_plots(
        self, 
        pycaret_module, 
        user_id: str, 
        model_name: str, 
        problem_type: str
    ) -> List[str]:
        """Generate evaluation plots using PlotService."""
        try:
            plot_types = []
            
            if problem_type == "classification":
                plot_types = ['confusion_matrix', 'class_report', 'roc']
            else:
                plot_types = ['residuals', 'prediction_error']
            
            plot_urls = []
            
            for plot_type in plot_types:
                try:
                    # Generate plot using PyCaret
                    pycaret_module.plot_model(
                        pycaret_module.pull().index[0],  # Best model
                        plot=plot_type,
                        save=True,
                        verbose=False
                    )
                    
                    # Move and rename the saved plot
                    plot_filename = NamingUtils.generate_plot_filename(
                        user_id, model_name, plot_type
                    )
                    
                    # PyCaret saves plots as PNG files in current directory
                    source_plot = Path(f"{plot_type}.png")
                    target_plot = settings.plots_dir / plot_filename
                    
                    if source_plot.exists():
                        source_plot.rename(target_plot)
                        plot_url = f"/api/v1/plots/view/{plot_filename}"
                        plot_urls.append(plot_url)
                    
                except Exception as plot_error:
                    logger.warning(f"Failed to generate {plot_type} plot: {plot_error}")
            
            return plot_urls
            
        except Exception as e:
            logger.warning(f"Failed to generate evaluation plots: {e}")
            return []