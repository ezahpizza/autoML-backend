"""
Plot management API routes.
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from services.plot_service import PlotService
from schemas.response_schemas import PlotListResponse, PlotListItem, BaseResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/list/{user_id}", response_model=PlotListResponse)
async def list_user_plots(user_id: str, limit: int = 50):
    """
    List all plots for a specific user.
    
    - **user_id**: User identifier to list plots for
    - **limit**: Maximum number of plots to return (default: 50)
    """
    try:
        plot_service = PlotService()
        plots = await plot_service.list_user_plots(user_id, limit)
        
        # Convert to response format
        plot_items = []
        for plot in plots:
            plot_items.append(PlotListItem(
                filename=plot["filename"],
                plot_type=plot["plot_type"],
                model_name=plot["model_name"],
                created_at=plot["created_at"],
                view_url=plot["view_url"]
            ))
        
        return PlotListResponse(
            success=True,
            message=f"Found {len(plots)} plots",
            plots=plot_items,
            total_count=len(plots)
        )
        
    except Exception as e:
        logger.error(f"Failed to list plots for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list plots: {str(e)}")


@router.get("/{filename}")
async def view_plot(filename: str):
    """
    View/download a plot image file.
    
    - **filename**: Plot filename to view
    """
    try:
        plot_service = PlotService()
        file_path = await plot_service.get_plot_path(filename)
        
        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="Plot not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='image/png'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to view plot {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to view plot: {str(e)}")


@router.delete("/delete/{filename}")
async def delete_plot(filename: str):
    """
    Delete a specific plot file.
    
    - **filename**: Plot filename to delete
    """
    try:
        plot_service = PlotService()
        deleted = await plot_service.delete_plot(filename)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Plot not found")
        
        return BaseResponse(
            success=True,
            message=f"Plot {filename} deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete plot {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete plot: {str(e)}")

@router.get("/by-model/{model_filename}")
async def get_plots_by_model(model_filename: str):
    """
    Get all plots associated with a specific model.
    
    - **model_filename**: Model filename to get plots for
    """
    try:
        plot_service = PlotService()
        plots = await plot_service.find_plots_by_model(model_filename)
        
        # Get detailed info for each plot
        plot_details = []
        for plot_filename in plots:
            info = await plot_service.get_plot_info(plot_filename)
            if info:
                plot_details.append(info)
        
        return {
            "success": True,
            "message": f"Found {len(plot_details)} plots for model {model_filename}",
            "plots": plot_details,
            "total_count": len(plot_details)
        }
        
    except Exception as e:
        logger.error(f"Failed to get plots for model {model_filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get plots: {str(e)}")


@router.get("/by-type/{user_id}/{plot_type}")
async def get_plots_by_type(user_id: str, plot_type: str):
    """
    Get all plots of a specific type for a user.
    
    - **user_id**: User identifier
    - **plot_type**: Type of plot (e.g., 'confusion_matrix', 'roc_curve', 'feature_importance')
    """
    try:
        plot_service = PlotService()
        plots = await plot_service.get_plots_by_type(user_id, plot_type)
        
        return {
            "success": True,
            "message": f"Found {len(plots)} plots of type '{plot_type}'",
            "plots": plots,
            "total_count": len(plots)
        }
        
    except Exception as e:
        logger.error(f"Failed to get plots by type {plot_type} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get plots: {str(e)}")


@router.delete("/delete-user/{user_id}")
async def delete_user_plots(user_id: str):
    """
    Delete all plots for a specific user.
    
    - **user_id**: User identifier
    """
    try:
        plot_service = PlotService()
        result = await plot_service.delete_user_plots(user_id)
        
        return {
            "success": True,
            "message": f"Deleted {result['deleted']} plots for user {user_id}",
            "deleted": result["deleted"],
            "failed": result["failed"]
        }
        
    except Exception as e:
        logger.error(f"Failed to delete plots for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete plots: {str(e)}")


@router.post("/cleanup-orphaned")
async def cleanup_orphaned_plots():
    """
    Remove plot files that are not referenced in any model job.
    
    This endpoint helps clean up orphaned plot files that may exist
    due to failed operations or database inconsistencies.
    """
    try:
        plot_service = PlotService()
        result = await plot_service.cleanup_orphaned_plots()
        
        return {
            "success": True,
            "message": f"Cleaned up {result['deleted']} orphaned plots",
            "deleted": result["deleted"],
            "failed": result["failed"]
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup orphaned plots: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup: {str(e)}")

@router.get("/health")
async def plots_health_check():
    """
    Health check endpoint for plot service.
    """
    try:
        plot_service = PlotService()
        
        # Check if plots directory exists and is writable
        plots_dir = plot_service.plots_dir
        plots_accessible = plots_dir.exists() and plots_dir.is_dir()
        
        # Count total plots
        total_plots = len(list(plots_dir.glob("*.png"))) if plots_accessible else 0
        
        return {
            "success": True,
            "message": "Plot service is healthy",
            "plots_directory_accessible": plots_accessible,
            "total_plots": total_plots
        }
        
    except Exception as e:
        logger.error(f"Plot service health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")