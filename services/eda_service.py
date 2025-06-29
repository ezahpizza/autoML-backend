"""
EDA service for generating and managing EDA reports using Pandas Profiling.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from pandas_profiling import ProfileReport
from fastapi import UploadFile, HTTPException

from config import settings
from db.mongodb import mongodb
from db.models import EDAJob
from schemas.request_schemas import EDAGenerateRequest
from utils.file_utils import FileManager
from utils.naming import NamingUtils

logger = logging.getLogger(__name__)


class EDAService:
    """Service for EDA report generation and management."""
    
    def __init__(self):
        self.db = mongodb
        self.eda_collection = self.db.get_collection("eda_jobs")
        self.reports_dir = settings.eda_reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_report(self, file: UploadFile, request: EDAGenerateRequest) -> Dict[str, Any]:
        """Generate EDA report from uploaded CSV file."""
        try:
            # Generate unique filename for the report
            report_filename = NamingUtils.generate_eda_filename(
                request.user_id, 
                request.dataset_name or file.filename
            )
            report_path = self.reports_dir / report_filename
            
            # Create temporary file path for uploaded data
            temp_filename = NamingUtils.generate_temp_filename(request.user_id, file.filename)
            temp_path = self.reports_dir / temp_filename
            
            try:
                # Save uploaded file temporarily
                await FileManager.save_uploaded_file(file, temp_path)
                
                # Read CSV data
                df = await FileManager.read_csv_file(temp_path)
                
                # Generate profile report
                profile = ProfileReport(
                    df,
                    title=f"EDA Report - {request.dataset_name or file.filename}",
                    explorative=True,
                    minimal=False
                )
                
                # Save report as HTML
                profile.to_file(report_path)
                
                # Get file size
                file_size = report_path.stat().st_size
                
                # Create EDA job record
                eda_job = EDAJob(
                    user_id=request.user_id,
                    filename=report_filename,
                    dataset_name=request.dataset_name or file.filename,
                    dataset_rows=len(df),
                    dataset_columns=len(df.columns),
                    file_size=file_size,
                    status="completed"
                )
                
                # Store in database
                await self.eda_collection.insert_one(eda_job.model_dump(by_alias=True))
                
                logger.info(f"EDA report generated: {report_filename}")
                
                return {
                    "filename": report_filename,
                    "report_url": f"/eda/view/{report_filename}",
                    "dataset_name": request.dataset_name or file.filename,
                    "dataset_rows": len(df),
                    "dataset_columns": len(df.columns),
                    "file_size": file_size
                }
                
            finally:
                # Clean up temporary file
                if temp_path.exists():
                    FileManager.delete_file(temp_path)
                    
        except Exception as e:
            logger.error(f"Failed to generate EDA report: {e}")
            raise HTTPException(status_code=500, detail=f"EDA generation failed: {str(e)}")
    
    async def get_report_path(self, filename: str) -> Optional[Path]:
        """Get path to EDA report file."""
        try:
            report_path = self.reports_dir / filename
            
            if report_path.exists():
                return report_path
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get report path for {filename}: {e}")
            return None
    
    async def list_user_reports(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """List all EDA reports for a specific user."""
        try:
            # Query database for user's EDA jobs
            cursor = self.eda_collection.find(
                {"user_id": user_id}
            ).sort("created_at", -1).limit(limit)
            
            reports = []
            async for doc in cursor:
                # Check if file still exists
                report_path = self.reports_dir / doc["filename"]
                if report_path.exists():
                    reports.append({
                        "filename": doc["filename"],
                        "dataset_name": doc["dataset_name"],
                        "dataset_rows": doc["dataset_rows"],
                        "dataset_columns": doc["dataset_columns"],
                        "created_at": doc["created_at"],
                        "view_url": f"/eda/view/{doc['filename']}",
                        "download_url": f"/eda/download/{doc['filename']}",
                        "file_size": doc.get("file_size", 0)
                    })
            
            return reports
            
        except Exception as e:
            logger.error(f"Failed to list EDA reports for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")
    
    async def get_eda_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get EDA generation history for a user."""
        try:
            # Query database for user's EDA history
            cursor = self.eda_collection.find(
                {"user_id": user_id}
            ).sort("created_at", -1).limit(limit)
            
            history = []
            async for doc in cursor:
                # Check if file still exists
                report_path = self.reports_dir / doc["filename"]
                file_exists = report_path.exists()
                
                history.append({
                    "id": str(doc["_id"]),
                    "filename": doc["filename"],
                    "dataset_name": doc["dataset_name"],
                    "dataset_rows": doc["dataset_rows"],
                    "dataset_columns": doc["dataset_columns"],
                    "file_size": doc.get("file_size", 0),
                    "status": doc.get("status", "completed"),
                    "created_at": doc["created_at"],
                    "file_exists": file_exists,
                    "view_url": f"/eda/view/{doc['filename']}" if file_exists else None
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get EDA history for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")
    
    async def delete_report(self, filename: str) -> bool:
        """Delete a specific EDA report and its database record."""
        try:
            # Delete file
            report_path = self.reports_dir / filename
            file_deleted = FileManager.delete_file(report_path)
            
            # Delete database record
            result = await self.eda_collection.delete_one({"filename": filename})
            record_deleted = result.deleted_count > 0
            
            if file_deleted or record_deleted:
                logger.info(f"Deleted EDA report: {filename}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete EDA report {filename}: {e}")
            return False
    
    