"""
Model management API routes.
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder
from typing import List

from services.model_service import ModelService
from schemas.request_schemas import CompareModelsRequest
from schemas.response_schemas import ModelListResponse, ModelListItem, BaseResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/list/{user_id}", response_model=ModelListResponse)
async def list_user_models(user_id: str, limit: int = 50):
    """
    List all models for a specific user.
    
    - **user_id**: User identifier to list models for
    - **limit**: Maximum number of records to return (default: 50)
    """
    try:
        model_service = ModelService()

        models = await model_service.list_user_models(user_id, limit)
        
        model_items = []
        for model in models:
            model_items.append(ModelListItem(
                filename=model["filename"],
                dataset_name=model["dataset_name"],
                target_column=model["target_column"],
                best_model=model["best_model"],
                best_model_score=model["best_model_score"],
                created_at=model["created_at"],
                download_url=f"/api/model/download/{model['filename']}"
            ))
        
        return ModelListResponse(
            success=True,
            message=f"Found {len(model_items)} models",
            models=model_items,
            total_count=len(model_items)
        )
        
    except Exception as e:
        logger.error(f"Failed to list models for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")
    
    
@router.get("/download/{filename}")
async def download_model(filename: str):
    """
    Download a trained model file (.pkl).
    
    - **filename**: Model filename to download
    """
    try:
        model_service = ModelService()      

        file_path = await model_service.get_model_path(filename)
        
        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="Model file not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download model {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")


@router.delete("/delete/{filename}")
async def delete_model(filename: str):
    """
    Delete a specific model file and its metadata.
    
    - **filename**: Model filename to delete
    """
    try:
        model_service = ModelService()

        deleted = await model_service.delete_model(filename)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Model file not found")
        
        return BaseResponse(
            success=True,
            message=f"Model {filename} deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@router.get("/metrics/{filename}")
async def get_model_metrics(filename: str):
    """
    Get detailed metrics for a specific model.
    
    - **filename**: Model filename
    """
    try:
        model_service = ModelService()

        metrics = await model_service.get_model_metrics(filename)
        
        if not metrics:
            raise HTTPException(status_code=404, detail="Model metrics not found")
        
        return jsonable_encoder({
            "success": True,
            "message": "Model metrics retrieved",
            "metrics": metrics
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model metrics for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/plots/{filename}")
async def get_model_plots(filename: str):
    """
    Get all plot URLs associated with a specific model.
    
    - **filename**: Model filename
    """
    try:
        model_service = ModelService()

        plots = await model_service.get_model_plots(filename)
        
        if not plots:
            raise HTTPException(status_code=404, detail="Model plots not found")
        
        return jsonable_encoder({
            "success": True,
            "message": "Model plots retrieved",
            "plots": plots
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model plots for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get plots: {str(e)}")


@router.post("/compare", response_model=None)
async def compare_user_models(request: CompareModelsRequest):
    """
    Compare multiple models for a user.
    
    - **user_id**: User identifier
    - **model_filenames**: Optional list of specific models to compare
    """
    try:
        model_service = ModelService()

        comparison = await model_service.compare_models(request)
        
        return jsonable_encoder({
            "success": True,
            "message": "Model comparison completed",
            "comparison": comparison
        })
        
    except Exception as e:
        logger.error(f"Failed to compare models for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compare models: {str(e)}")

