"""
Cleanup API routes for file and database maintenance.
"""

import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks

from services.cleanup_service import CleanupService
from schemas.request_schemas import CleanupUserRequest
from schemas.response_schemas import CleanupResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/user/{user_id}", response_model=CleanupResponse)
async def cleanup_user_files(
    user_id: str,
    background_tasks: BackgroundTasks,
    confirm: bool = False
):
    """
    Delete all files and records belonging to a specific user.
    
    - **user_id**: User identifier to cleanup
    - **confirm**: Must be True to proceed with cleanup
    """
    try:
        # Validate confirmation
        if not confirm:
            raise HTTPException(
                status_code=400, 
                detail="Confirmation required. Set confirm=true to proceed"
            )
        
        # Create request object
        cleanup_request = CleanupUserRequest(
            user_id=user_id,
            confirm=confirm
        )
        
        # Initialize cleanup service
        cleanup_service = CleanupService()
        
        # Perform cleanup
        result = await cleanup_service.cleanup_user_data(cleanup_request)
        
        logger.info(f"User cleanup completed for user {user_id}")
        
        return CleanupResponse(
            success=True,
            message=f"Successfully cleaned up data for user {user_id}",
            files_deleted=result["files_deleted"],
            records_deleted=result["records_deleted"],
            total_files_deleted=result["total_files_deleted"],
            total_records_deleted=result["total_records_deleted"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User cleanup failed for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.post("/system", response_model=CleanupResponse)
async def cleanup_old_files(
    background_tasks: BackgroundTasks,
    hours: int = 24,
    dry_run: bool = False
):
    """
    Delete files older than specified hours across all users.
    
    - **hours**: Delete files older than this many hours (default: 24)
    - **dry_run**: If True, only return what would be deleted without deleting
    """
    try:
        # Initialize cleanup service
        cleanup_service = CleanupService()
        
        # Perform system cleanup
        result = await cleanup_service.cleanup_old_files(hours, dry_run)
        
        action = "Would delete" if dry_run else "Deleted"
        logger.info(f"System cleanup: {action} {result['total_files_deleted']} files")
        
        return CleanupResponse(
            success=True,
            message=f"{action} {result['total_files_deleted']} old files and {result['total_records_deleted']} records",
            files_deleted=result["files_deleted"],
            records_deleted=result["records_deleted"],
            total_files_deleted=result["total_files_deleted"],
            total_records_deleted=result["total_records_deleted"]
        )
        
    except Exception as e:
        logger.error(f"System cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"System cleanup failed: {str(e)}")


@router.post("/orphaned")
async def cleanup_orphaned_records():
    """
    Remove database records that don't have corresponding files.
    """
    try:
        # Initialize cleanup service
        cleanup_service = CleanupService()
        
        # Clean orphaned records
        result = await cleanup_service.cleanup_orphaned_records()
        
        logger.info(f"Orphaned cleanup: removed {result['total_records_deleted']} records")
        
        return {
            "success": True,
            "message": f"Removed {result['total_records_deleted']} orphaned records",
            "records_deleted": result["records_deleted"],
            "total_records_deleted": result["total_records_deleted"]
        }
        
    except Exception as e:
        logger.error(f"Orphaned cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Orphaned cleanup failed: {str(e)}")


@router.get("/status")
async def get_cleanup_status():
    """
    Get current cleanup status and statistics.
    """
    try:
        # Initialize cleanup service
        cleanup_service = CleanupService()
        
        # Get cleanup statistics
        stats = await cleanup_service.get_cleanup_statistics()
        
        return {
            "success": True,
            "message": "Cleanup status retrieved",
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get cleanup status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/logs")
async def get_cleanup_logs(limit: int = 50):
    """
    Get recent cleanup operation logs.
    
    - **limit**: Maximum number of log entries to return (default: 50)
    """
    try:
        # Initialize cleanup service
        cleanup_service = CleanupService()
        
        # Get cleanup logs
        logs = await cleanup_service.get_cleanup_logs(limit)
        
        return {
            "success": True,
            "message": f"Retrieved {len(logs)} cleanup log entries",
            "logs": logs,
            "total_count": len(logs)
        }
        
    except Exception as e:
        logger.error(f"Failed to get cleanup logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")


@router.post("/schedule")
async def schedule_cleanup(
    background_tasks: BackgroundTasks,
    hours: int = 24,
    user_id: str = None
):
    """
    Schedule a cleanup operation to run in the background.
    
    - **hours**: Delete files older than this many hours
    - **user_id**: Optional user ID for user-specific cleanup
    """
    try:
        # Initialize cleanup service
        cleanup_service = CleanupService()
        
        # Schedule cleanup in background
        if user_id:
            cleanup_request = CleanupUserRequest(user_id=user_id, confirm=True)
            background_tasks.add_task(cleanup_service.cleanup_user_data, cleanup_request)
            message = f"Scheduled cleanup for user {user_id}"
        else:
            background_tasks.add_task(cleanup_service.cleanup_old_files, hours, False)
            message = f"Scheduled cleanup of files older than {hours} hours"
        
        logger.info(f"Cleanup scheduled: {message}")
        
        return {
            "success": True,
            "message": message,
            "scheduled": True
        }
        
    except Exception as e:
        logger.error(f"Failed to schedule cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule cleanup: {str(e)}")