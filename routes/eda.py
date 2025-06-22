"""
EDA report generation API routes.
"""

import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.encoders import jsonable_encoder

from services.eda_service import EDAService
from schemas.request_schemas import EDAGenerateRequest
from schemas.response_schemas import EDAResponse
from utils.file_utils import FileManager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate", response_model=EDAResponse)
async def generate_eda_report(
    file: UploadFile = File(..., description="CSV dataset file"),
    user_id: str = Form(..., description="User identifier"),
    dataset_name: str = Form(None, description="Optional dataset name")
):
    """
    Generate EDA report from uploaded CSV dataset.
    
    - **file**: CSV file containing the dataset
    - **user_id**: Unique user identifier
    - **dataset_name**: Optional custom name for the dataset
    """
    try:
        # Validate file
        is_valid, error_msg = await FileManager.validate_csv_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Create request object
        eda_request = EDAGenerateRequest(
            user_id=user_id,
            dataset_name=dataset_name or file.filename
        )
        
        # Initialize EDA service
        eda_service = EDAService()
        await eda_service.async_init()
        
        # Generate EDA report
        result = await eda_service.generate_report(file, eda_request)
        
        logger.info(f"EDA report generated for user {user_id}")
        
        return jsonable_encoder(EDAResponse(
            success=True,
            message="EDA report generated successfully",
            filename=result["filename"],
            report_url=result["report_url"],
            dataset_name=result["dataset_name"],
            dataset_rows=result["dataset_rows"],
            dataset_columns=result["dataset_columns"],
            file_size=result["file_size"]
        ))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"EDA report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"EDA generation failed: {str(e)}")


@router.get("/view/{filename}")
async def view_eda_report(filename: str):
    """
    View EDA report HTML file.
    
    - **filename**: EDA report filename to view
    """
    try:
        eda_service = EDAService()
        await eda_service.async_init()

        file_path = await eda_service.get_report_path(filename)
        
        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="EDA report not found")
        
        # Return HTML content directly
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return HTMLResponse(content=html_content)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to view EDA report {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to view report: {str(e)}")


@router.get("/download/{filename}")
async def download_eda_report(filename: str):
    """
    Download EDA report HTML file.
    
    - **filename**: EDA report filename to download
    """
    try:
        eda_service = EDAService()
        await eda_service.async_init()

        file_path = await eda_service.get_report_path(filename)
        
        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="EDA report not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='text/html'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download EDA report {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download report: {str(e)}")


@router.get("/list/{user_id}")
async def list_user_eda_reports(user_id: str, limit: int = 50):
    """
    List all EDA reports for a specific user.
    
    - **user_id**: User identifier to list reports for
    - **limit**: Maximum number of records to return (default: 50)
    """
    try:
        eda_service = EDAService()
        await eda_service.async_init()
        
        reports = await eda_service.list_user_reports(user_id, limit)
        
        return jsonable_encoder({
            "success": True,
            "message": f"Found {len(reports)} EDA reports",
            "reports": reports,
            "total_count": len(reports)
        })
        
    except Exception as e:
        logger.error(f"Failed to list EDA reports for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")


@router.get("/history/{user_id}")
async def get_eda_history(user_id: str, limit: int = 50):
    """
    Get EDA generation history for a user.
    
    - **user_id**: User identifier
    - **limit**: Maximum number of records to return (default: 50)
    """
    try:
        eda_service = EDAService()
        await eda_service.async_init()

        history = await eda_service.get_eda_history(user_id, limit)
        
        return jsonable_encoder({
            "success": True,
            "message": f"Retrieved {len(history)} EDA records",
            "history": history,
            "total_count": len(history)
        })
        
    except Exception as e:
        logger.error(f"Failed to get EDA history for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.delete("/delete/{filename}")
async def delete_eda_report(filename: str):
    """
    Delete a specific EDA report.
    
    - **filename**: EDA report filename to delete
    """
    try:
        eda_service = EDAService()
        await eda_service.async_init()
        
        deleted = await eda_service.delete_report(filename)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="EDA report not found")
        
        return jsonable_encoder({
            "success": True,
            "message": f"EDA report {filename} deleted successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete EDA report {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete report: {str(e)}")
