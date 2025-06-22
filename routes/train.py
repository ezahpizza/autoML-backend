"""
Model training API routes.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.encoders import jsonable_encoder

from services.train_service import TrainService
from schemas.request_schemas import ModelTrainRequest, PredictionRequest
from schemas.response_schemas import ModelTrainResponse, PredictionResponse
from utils.file_utils import FileManager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/train", response_model=ModelTrainResponse)
async def train_model(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV dataset file"),
    user_id: str = Form(..., description="User identifier"),
    target_column: str = Form(..., description="Target column name"),
    dataset_name: str = Form(None, description="Optional dataset name"),
    model_types: str = Form(None, description="Comma-separated model types")
):
    """
    Train ML model on uploaded dataset.
    
    - **file**: CSV file containing the dataset
    - **user_id**: Unique user identifier
    - **target_column**: Name of the target column for training
    - **dataset_name**: Optional custom name for the dataset
    - **model_types**: Optional comma-separated list of specific models to train
    """
    try:
        # Validate file
        is_valid, error_msg = await FileManager.validate_csv_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Parse model types if provided
        parsed_model_types = None
        if model_types:
            parsed_model_types = [mt.strip() for mt in model_types.split(',') if mt.strip()]
        
        # Create request object
        train_request = ModelTrainRequest(
            user_id=user_id,
            target_column=target_column,
            dataset_name=dataset_name or file.filename,
            model_types=parsed_model_types
        )
        
        # Initialize training service
        train_service = TrainService()
        await train_service.async_init()
        
        # Start training process
        result = await train_service.train_model(file, train_request)
        
        logger.info(f"Model training completed for user {user_id}")
        
        # Use jsonable_encoder to ensure datetime serialization
        return jsonable_encoder(ModelTrainResponse(
            success=True,
            message="Model training completed successfully",
            filename=result["filename"],
            download_url=result["download_url"],
            dataset_name=result["dataset_name"],
            target_column=result["target_column"],
            model_type=result["model_type"],
            best_model=result["best_model"],
            best_model_score=result["best_model_score"],
            metrics=result["metrics"],
            plot_urls=result["plot_urls"],
            training_time=result["training_time"]
        ))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/history/{user_id}")
async def get_model_history(user_id: str, limit: int = 50):
    """
    Get model training history for a user.
    
    - **user_id**: User identifier
    - **limit**: Maximum number of records to return (default: 50)
    """
    try:
        train_service = TrainService()
        await train_service.async_init()
        history = await train_service.get_training_history(user_id, limit)
        
        return jsonable_encoder({
            "success": True,
            "message": f"Retrieved {len(history)} training records",
            "history": history,
            "total_count": len(history)
        })
        
    except Exception as e:
        logger.error(f"Failed to get training history for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.post("/predict")
async def make_prediction(request: Dict[str, Any]):
    """
    Make predictions using a trained model.
    
    - **user_id**: User identifier
    - **model_filename**: Filename of the trained model
    - **input_data**: Dictionary of input features for prediction
    """
    try:

        # Validate request
        prediction_request = PredictionRequest(**request)
        
        # Initialize training service
        train_service = TrainService()
        await train_service.async_init()
        
        # Make prediction
        result = await train_service.predict(prediction_request)
        
        return jsonable_encoder(PredictionResponse(
            success=True,
            message="Prediction completed successfully",
            predictions=result["predictions"],
            prediction_probabilities=result.get("prediction_probabilities"),
            model_used=result["model_filename"],
            input_features=result["input_data"]
        ))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/validate-dataset")
async def validate_dataset(
    file: UploadFile = File(..., description="CSV dataset file"),
    target_column: str = Form(..., description="Target column name")
):
    """
    Validate dataset for training without actually training.
    
    - **file**: CSV file to validate
    - **target_column**: Target column name to check
    """
    try:
        # Validate file format
        is_valid, error_msg = await FileManager.validate_csv_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Initialize training service
        train_service = TrainService()
        await train_service.async_init()
        
        # Validate dataset
        validation_result = await train_service.validate_dataset(file, target_column)
        
        return jsonable_encoder({
            "success": True,
            "message": "Dataset validation completed",
            "validation_result": validation_result
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")


