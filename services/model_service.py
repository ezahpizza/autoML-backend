"""
Model management service for AutoML platform.
"""

import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import settings
from db.mongodb import MongoDB
from utils.naming import NamingUtils
from utils.file_utils import FileManager
from schemas.request_schemas import CompareModelsRequest

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing trained models and their metadata."""
    
    def __init__(self):
        self.db = MongoDB()
    
    async def async_init(self):
        await self.db.connect()

    async def list_user_models(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """List all models for a specific user."""
        try:            
            # Query models collection
            model_jobs_collection = self.db.get_collection("model_jobs")
            cursor = model_jobs_collection.find(
                {"user_id": user_id},
                sort=[("created_at", -1)],
                limit=limit
            )
            
            models = []
            async for model_doc in cursor:
                # Convert ObjectId to string for JSON serialization
                model_doc["_id"] = str(model_doc["_id"])
                models.append(model_doc)
            
            logger.info(f"Found {len(models)} models for user {user_id}")
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models for user {user_id}: {e}")
            raise
    
    async def get_model_path(self, filename: str) -> Optional[Path]:
        """Get the file path for a model."""
        try:
            model_path = settings.models_dir / filename
            
            if model_path.exists():
                return model_path
            
            logger.warning(f"Model file not found: {filename}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get model path for {filename}: {e}")
            return None
    
    async def delete_model(self, filename: str) -> bool:
        """Delete a model file and its metadata."""
        try:
            # Delete from database
            model_jobs_collection = self.db.get_collection("model_jobs")
            result = await model_jobs_collection.delete_one({"filename": filename})
            
            # Delete model file
            model_path = settings.models_dir / filename
            file_deleted = FileManager.delete_file(model_path)
            
            # Delete associated plot files
            await self._delete_model_plots(filename)
            
            if result.deleted_count > 0 or file_deleted:
                logger.info(f"Deleted model: {filename}")
                return True
            
            logger.warning(f"Model not found for deletion: {filename}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete model {filename}: {e}")
            raise
    
    async def _delete_model_plots(self, model_filename: str) -> int:
        """Delete all plots associated with a model."""
        try:            
            # Get model document to find plot filenames
            model_jobs_collection = self.db.get_collection("model_jobs")
            model_doc = await model_jobs_collection.find_one({"filename": model_filename})
            
            if not model_doc or not model_doc.get("plot_filenames"):
                return 0
            
            deleted_count = 0
            for plot_filename in model_doc["plot_filenames"]:
                plot_path = settings.plots_dir / plot_filename
                if FileManager.delete_file(plot_path):
                    deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} plots for model {model_filename}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete plots for model {model_filename}: {e}")
            return 0
    
    async def get_model_metrics(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for a specific model."""
        try:
            model_jobs_collection = self.db.get_collection("model_jobs")
            model_doc = await model_jobs_collection.find_one({"filename": filename})
            
            if not model_doc:
                logger.warning(f"Model metadata not found: {filename}")
                return None
            
            # Return comprehensive model information
            metrics = {
                "filename": model_doc.get("filename"),
                "dataset_name": model_doc.get("dataset_name"),
                "target_column": model_doc.get("target_column"),
                "model_type": model_doc.get("model_type"),
                "best_model": model_doc.get("best_model"),
                "best_model_score": model_doc.get("best_model_score"),
                "metrics": model_doc.get("metrics", {}),
                "dataset_rows": model_doc.get("dataset_rows"),
                "dataset_columns": model_doc.get("dataset_columns"),
                "training_time": model_doc.get("training_time"),
                "created_at": model_doc.get("created_at"),
                "status": model_doc.get("status", "completed")
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get model metrics for {filename}: {e}")
            raise
    
    async def get_model_plots(self, filename: str) -> Optional[List[Dict[str, str]]]:
        """Get all plot URLs associated with a specific model."""
        try:
            model_jobs_collection = self.db.get_collection("model_jobs")
            model_doc = await model_jobs_collection.find_one({"filename": filename})
            
            if not model_doc:
                logger.warning(f"Model metadata not found: {filename}")
                return None
            
            plot_filenames = model_doc.get("plot_filenames", [])
            
            plots = []
            for plot_filename in plot_filenames:
                # Parse plot filename to get plot type
                plot_info = NamingUtils.parse_plot_filename(plot_filename)
                plot_type = plot_info.get("plot_type", "unknown")
                
                # Check if plot file exists
                plot_path = settings.plots_dir / plot_filename
                if plot_path.exists():
                    plots.append({
                        "plot_type": plot_type,
                        "filename": plot_filename,
                        "url": f"/api/plot/view/{plot_filename}"
                    })
            
            return plots
            
        except Exception as e:
            logger.error(f"Failed to get model plots for {filename}: {e}")
            raise
    
    async def compare_models(self, request: CompareModelsRequest) -> Dict[str, Any]:
        """Compare multiple models for a user."""
        try:
            user_id = request.user_id
            model_filenames = request.model_filenames

            # Build query
            query = {"user_id": user_id}
            if model_filenames:
                query["filename"] = {"$in": model_filenames}
            
            # Get models to compare
            model_jobs_collection = self.db.get_collection("model_jobs")
            cursor = model_jobs_collection.find(query, sort=[("best_model_score", -1)])
            models = []
            async for model_doc in cursor:
                models.append(model_doc)
            
            if len(models) < 2:
                return {
                    "error": "Need at least 2 models to compare",
                    "available_models": len(models)
                }
            
            # Create comparison data
            comparison = {
                "total_models": len(models),
                "best_model": {
                    "filename": models[0]["filename"],
                    "model_name": models[0]["best_model"],
                    "score": models[0]["best_model_score"],
                    "dataset": models[0]["dataset_name"]
                },
                "models": []
            }
            
            # Add model comparison details
            for model in models:
                comparison["models"].append({
                    "filename": model["filename"],
                    "dataset_name": model["dataset_name"],
                    "target_column": model["target_column"],
                    "best_model": model["best_model"],
                    "score": model["best_model_score"],
                    "model_type": model["model_type"],
                    "training_time": model.get("training_time"),
                    "dataset_size": f"{model['dataset_rows']} rows x {model['dataset_columns']} cols",
                    "created_at": model["created_at"]
                })
            
            # Calculate statistics
            scores = [m["best_model_score"] for m in models]
            comparison["statistics"] = {
                "average_score": sum(scores) / len(scores),
                "score_range": max(scores) - min(scores),
                "model_types": list(set(m["model_type"] for m in models))
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models for user {user_id}: {e}")
            raise
    
    async def validate_model(self, filename: str) -> bool:
        """Validate that a model file is accessible and loadable."""
        try:
            model_path = settings.models_dir / filename
            
            # Check if file exists
            if not model_path.exists():
                logger.warning(f"Model file does not exist: {filename}")
                return False
            
            # Try to load the pickle file
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # Basic validation - check if it's a PyCaret model
                if hasattr(model, 'predict'):
                    logger.info(f"Model validation successful: {filename}")
                    return True
                else:
                    logger.warning(f"Model file is not a valid ML model: {filename}")
                    return False
                    
            except (pickle.PickleError, EOFError, AttributeError) as e:
                logger.warning(f"Failed to load model file {filename}: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to validate model {filename}: {e}")
            return False

