"""
Cleanup service for managing file and database cleanup operations.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import settings
from db.mongodb import mongodb
from utils.file_utils import FileManager
from schemas.request_schemas import CleanupUserRequest
from db.models import CleanupLog

logger = logging.getLogger(__name__)


class CleanupService:
    """Service for handling cleanup operations on files and database records."""
    
    def __init__(self):
        self.db = mongodb
    
    async def cleanup_user_data(self, request: CleanupUserRequest) -> Dict[str, Any]:
        """
        Clean up all files and database records for a specific user.
        
        Args:
            request: User cleanup request with user_id and confirmation
            
        Returns:
            Dictionary containing cleanup results
        """
        try:
            user_id = request.user_id
            files_deleted = []
            records_deleted = {}
            
            # Find and delete user files from all directories
            directories = [
                settings.models_dir,
                settings.eda_reports_dir
            ]
            
            for directory in directories:
                user_files = FileManager.find_user_files(directory, user_id)
                
                for file_path in user_files:
                    if FileManager.delete_file(file_path):
                        files_deleted.append(str(file_path))
                        logger.info(f"Deleted user file: {file_path}")

            db = mongodb
            await db.connect()
            eda_collection = db.get_collection("eda_jobs")
            model_collection = db.get_collection("model_jobs")
            pred_collection = db.get_collection("predictions")

            eda_result = await eda_collection.delete_many({"user_id": user_id})
            records_deleted["eda_jobs"] = eda_result.deleted_count
            model_result = await model_collection.delete_many({"user_id": user_id})
            records_deleted["model_jobs"] = model_result.deleted_count
            pred_result = await pred_collection.delete_many({"user_id": user_id})
            records_deleted["predictions"] = pred_result.deleted_count
            
            # Log cleanup operation
            await self._log_cleanup_operation(
                operation_type="user_cleanup",
                user_id=user_id,
                files_deleted=files_deleted,
                records_deleted=records_deleted
            )
                        
            total_files = len(files_deleted)
            total_records = sum(records_deleted.values())
            
            logger.info(f"User cleanup completed for {user_id}: {total_files} files, {total_records} records")
            
            return {
                "files_deleted": files_deleted,
                "records_deleted": records_deleted,
                "total_files_deleted": total_files,
                "total_records_deleted": total_records
            }
            
        except Exception as e:
            logger.error(f"User cleanup failed for {request.user_id}: {e}")
            raise
    
    async def cleanup_old_files(self, hours: int = 24, dry_run: bool = False) -> Dict[str, Any]:
        """
        Clean up files older than specified hours.
        
        Args:
            hours: Delete files older than this many hours
            dry_run: If True, only return what would be deleted
            
        Returns:
            Dictionary containing cleanup results
        """
        try:
            files_deleted = []
            records_deleted = {}
            
            # Find old files in all directories
            directories = [
                settings.models_dir,
                settings.eda_reports_dir
            ]
            
            for directory in directories:
                old_files = FileManager.find_old_files(directory, hours)
                
                for file_path in old_files:
                    if not dry_run:
                        if FileManager.delete_file(file_path):
                            files_deleted.append(str(file_path))
                            logger.info(f"Deleted old file: {file_path}")
                    else:
                        files_deleted.append(str(file_path))
            
            if not dry_run:
                db = mongodb
                await db.connect()
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

                eda_collection = db.get_collection("eda_jobs")
                model_collection = db.get_collection("model_jobs")
                pred_collection = db.get_collection("predictions")

                eda_result = await eda_collection.delete_many({"created_at": {"$lt": cutoff_time}})
                records_deleted["eda_jobs"] = eda_result.deleted_count
                model_result = await model_collection.delete_many({"created_at": {"$lt": cutoff_time}})
                records_deleted["model_jobs"] = model_result.deleted_count
                pred_result = await pred_collection.delete_many({"created_at": {"$lt": cutoff_time}})
                records_deleted["predictions"] = pred_result.deleted_count
                
                await self._log_cleanup_operation(
                    operation_type="system_cleanup",
                    files_deleted=files_deleted,
                    records_deleted=records_deleted
                )
            
            total_files = len(files_deleted)
            total_records = sum(records_deleted.values()) if not dry_run else 0
            
            action = "Would delete" if dry_run else "Deleted"
            logger.info(f"System cleanup: {action} {total_files} files, {total_records} records")
            
            return {
                "files_deleted": files_deleted,
                "records_deleted": records_deleted,
                "total_files_deleted": total_files,
                "total_records_deleted": total_records
            }
            
        except Exception as e:
            logger.error(f"System cleanup failed: {e}")
            raise
    
    async def cleanup_orphaned_records(self) -> Dict[str, Any]:
        """
        Remove database records that don't have corresponding files.
        
        Returns:
            Dictionary containing cleanup results
        """
        try:
            records_deleted = {}
            db = mongodb
            await db.connect()
            eda_collection = db.get_collection("eda_jobs")
            model_collection = db.get_collection("model_jobs")
            eda_jobs = await eda_collection.find({}).to_list(None)
            orphaned_eda_jobs = []
            
            for job in eda_jobs:
                if "report_path" in job and job["report_path"]:
                    report_path = Path(job["report_path"])
                    if not report_path.exists():
                        orphaned_eda_jobs.append(job["_id"])
            
            if orphaned_eda_jobs:
                result = await eda_collection.delete_many({"_id": {"$in": orphaned_eda_jobs}})
                records_deleted["eda_jobs"] = result.deleted_count
            else:
                records_deleted["eda_jobs"] = 0
            
            model_jobs = await model_collection.find({}).to_list(None)
            orphaned_model_jobs = []
            
            for job in model_jobs:
                if "model_path" in job and job["model_path"]:
                    model_path = Path(job["model_path"])
                    if not model_path.exists():
                        orphaned_model_jobs.append(job["_id"])
            
            if orphaned_model_jobs:
                result = await model_collection.delete_many({"_id": {"$in": orphaned_model_jobs}})
                records_deleted["model_jobs"] = result.deleted_count
            else:
                records_deleted["model_jobs"] = 0
            
            # Log cleanup operation
            await self._log_cleanup_operation(
                operation_type="orphaned_cleanup",
                records_deleted=records_deleted
            )
            
            total_records = sum(records_deleted.values())
            logger.info(f"Orphaned cleanup: removed {total_records} records")
            
            return {
                "records_deleted": records_deleted,
                "total_records_deleted": total_records
            }
            
        except Exception as e:
            logger.error(f"Orphaned cleanup failed: {e}")
            raise
    
    async def get_cleanup_statistics(self) -> Dict[str, Any]:
        """
        Get cleanup statistics and current storage status.
        
        Returns:
            Dictionary containing cleanup statistics
        """
        try:
            stats = {}
            
            # File statistics
            directories = {
                "models": settings.models_dir,
                "eda_reports": settings.eda_reports_dir
            }
            
            file_stats = {}
            total_files = 0
            total_size = 0
            
            for name, directory in directories.items():
                if directory.exists():
                    files = list(directory.rglob('*'))
                    files = [f for f in files if f.is_file()]
                    
                    size = FileManager.get_directory_size(directory)
                    file_stats[name] = {
                        "count": len(files),
                        "size_bytes": size,
                        "size_mb": round(size / (1024 * 1024), 2)
                    }
                    
                    total_files += len(files)
                    total_size += size
                else:
                    file_stats[name] = {"count": 0, "size_bytes": 0, "size_mb": 0}
            
            stats["files"] = file_stats
            stats["total_files"] = total_files
            stats["total_size_bytes"] = total_size
            stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)
            
            # Database statistics
            db = mongodb
            await db.connect()
            
            db_stats = {}
            collections = ["eda_jobs", "model_jobs", "predictions", "cleanup_logs"]
            
            for collection in collections:
                collection_obj = db.get_collection(collection)
                count = await collection_obj.count_documents({})
                db_stats[collection] = count
            
            stats["database"] = db_stats
            
            # Old files count (older than retention period)
            old_files_count = 0
            for directory in directories.values():
                old_files = FileManager.find_old_files(directory, settings.FILE_RETENTION_HOURS)
                old_files_count += len(old_files)
            
            stats["old_files_count"] = old_files_count
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cleanup statistics: {e}")
            raise
    
    async def get_cleanup_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent cleanup operation logs.
        
        Args:
            limit: Maximum number of log entries to return
            
        Returns:
            List of cleanup log entries
        """
        try:
            db = mongodb
            await db.connect()
            logs_collection = db.get_collection("cleanup_logs")
            cursor = logs_collection.find({}).sort("created_at", -1).limit(limit)
            logs = await cursor.to_list(None)
            
            # Convert ObjectId to string for JSON serialization
            for log in logs:
                log["_id"] = str(log["_id"])
                if "created_at" in log:
                    log["created_at"] = log["created_at"].isoformat()
            
            return logs
            
        except Exception as e:
            logger.error(f"Failed to get cleanup logs: {e}")
            raise
    
    async def startup_cleanup(self) -> Dict[str, Any]:
        """
        Perform cleanup operations on application startup.
        
        Returns:
            Dictionary containing startup cleanup results
        """
        try:
            logger.info("Starting application startup cleanup...")
            
            # Clean up old files
            result = await self.cleanup_old_files(settings.FILE_RETENTION_HOURS, dry_run=False)
            
            # Clean up orphaned records
            orphaned_result = await self.cleanup_orphaned_records()
            
            # Combine results
            startup_result = {
                "files_deleted": result["files_deleted"],
                "records_deleted": {
                    **result["records_deleted"],
                    **{f"orphaned_{k}": v for k, v in orphaned_result["records_deleted"].items()}
                },
                "total_files_deleted": result["total_files_deleted"],
                "total_records_deleted": result["total_records_deleted"] + orphaned_result["total_records_deleted"]
            }
            
            logger.info(
                f"Startup cleanup completed: {startup_result['total_files_deleted']} files, "
                f"{startup_result['total_records_deleted']} records"
            )
            
            return startup_result
            
        except Exception as e:
            logger.error(f"Startup cleanup failed: {e}")
            raise
    
    async def _log_cleanup_operation(
        self,
        operation_type: str,
        user_id: Optional[str] = None,
        files_deleted: Optional[List[str]] = None,
        records_deleted: Optional[Dict[str, int]] = None
    ) -> None:
        """
        Log cleanup operation to database.
        
        Args:
            operation_type: Type of cleanup operation
            user_id: User ID if user-specific cleanup
            files_deleted: List of deleted files
            records_deleted: Count of deleted records per collection
        """
        try:
            db = mongodb
            await db.connect()
            logs_collection = db.get_collection("cleanup_logs")
            log_entry = CleanupLog(
                operation_type=operation_type,
                user_id=user_id,
                files_deleted=files_deleted or [],
                records_deleted=records_deleted or {},
                created_at=datetime.now(timezone.utc)
            )
            
            await logs_collection.insert_one(log_entry.model_dump(by_alias=True))
            
        except Exception as e:
            logger.error(f"Failed to log cleanup operation: {e}")
            # Don't raise exception here as it's just logging