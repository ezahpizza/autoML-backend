"""
File operation utilities for AutoML platform.
"""

import os
import aiofiles
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import pandas as pd
from fastapi import UploadFile, HTTPException

from config import settings

logger = logging.getLogger(__name__)


class FileManager:
    """File management utilities for the AutoML platform."""
    
    @staticmethod
    async def save_uploaded_file(file: UploadFile, filepath: Path) -> Dict[str, Any]:
        """Save uploaded file to disk and return metadata."""
        try:
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save file
            async with aiofiles.open(filepath, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Get file stats
            stat = filepath.stat()
            
            return {
                "filename": filepath.name,
                "filepath": str(filepath),
                "size": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_ctime)
            }
            
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    @staticmethod
    async def read_csv_file(filepath: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Read CSV file with validation and size limits."""
        try:
            # Check file exists
            if not filepath.exists():
                raise HTTPException(status_code=404, detail="File not found")
            
            # Check file size
            file_size = filepath.stat().st_size
            if file_size > settings.max_file_size_bytes:
                raise HTTPException(
                    status_code=413, 
                    detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
                )
            
            # Read CSV with limits
            df = pd.read_csv(
                filepath,
                nrows=max_rows or settings.MAX_DATASET_ROWS
            )
            
            # Validate dataset dimensions
            if len(df) > settings.MAX_DATASET_ROWS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Dataset has too many rows. Max: {settings.MAX_DATASET_ROWS}"
                )
            
            if len(df.columns) > settings.MAX_DATASET_COLUMNS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Dataset has too many columns. Max: {settings.MAX_DATASET_COLUMNS}"
                )
            
            return df
            
        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        except pd.errors.ParserError as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to read CSV: {str(e)}")
    
    @staticmethod
    async def validate_csv_file(file: UploadFile) -> Tuple[bool, str]:
        """Validate uploaded CSV file."""
        # Check file extension
        if not file.filename.lower().endswith('.csv'):
            return False, "File must be a CSV file"
        
        # Check file size
        if file.size and file.size > settings.max_file_size_bytes:
            return False, f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
        
        # Reset file pointer
        await file.seek(0)
        
        # Try to read first few lines
        try:
            content = await file.read(1024)  # Read first 1KB
            await file.seek(0)  # Reset pointer
            
            # Check if it looks like CSV
            lines = content.decode('utf-8').split('\n')
            if len(lines) < 2:
                return False, "CSV must have at least a header and one data row"
            
            # Basic CSV format check
            header_cols = len(lines[0].split(','))
            if header_cols < 2:
                return False, "CSV must have at least 2 columns"
            
            return True, "Valid CSV file"
            
        except UnicodeDecodeError:
            return False, "File encoding not supported. Please use UTF-8"
        except Exception as e:
            return False, f"Invalid CSV file: {str(e)}"
    
    @staticmethod
    def delete_file(filepath: Path) -> bool:
        """Delete a file safely."""
        try:
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Deleted file: {filepath}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file {filepath}: {e}")
            return False
    
    @staticmethod
    def delete_files(filepaths: List[Path]) -> Dict[str, int]:
        """Delete multiple files and return statistics."""
        deleted = 0
        failed = 0
        
        for filepath in filepaths:
            if FileManager.delete_file(filepath):
                deleted += 1
            else:
                failed += 1
        
        return {"deleted": deleted, "failed": failed}
    
    @staticmethod
    def find_old_files(directory: Path, hours: int = 24) -> List[Path]:
        """Find files older than specified hours."""
        if not directory.exists():
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        old_files = []
        
        try:
            for filepath in directory.rglob('*'):
                if filepath.is_file():
                    # Check file modification time
                    file_time = datetime.fromtimestamp(filepath.stat().st_mtime)
                    if file_time < cutoff_time:
                        old_files.append(filepath)
        except Exception as e:
            logger.error(f"Error finding old files in {directory}: {e}")
        
        return old_files
    
    @staticmethod
    def find_user_files(directory: Path, user_id: str) -> List[Path]:
        """Find all files belonging to a specific user."""
        if not directory.exists():
            return []
        
        user_files = []
        
        try:
            for filepath in directory.rglob('*'):
                if filepath.is_file() and user_id in filepath.name:
                    user_files.append(filepath)
        except Exception as e:
            logger.error(f"Error finding user files in {directory}: {e}")
        
        return user_files
    
    @staticmethod
    def get_file_info(filepath: Path) -> Optional[Dict[str, Any]]:
        """Get file information."""
        try:
            if not filepath.exists():
                return None
            
            stat = filepath.stat()
            
            return {
                "filename": filepath.name,
                "size": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_ctime),
                "modified_at": datetime.fromtimestamp(stat.st_mtime),
                "extension": filepath.suffix
            }
        except Exception as e:
            logger.error(f"Error getting file info for {filepath}: {e}")
            return None
    
    @staticmethod
    def get_directory_size(directory: Path) -> int:
        """Get total size of directory in bytes."""
        if not directory.exists():
            return 0
        
        total_size = 0
        try:
            for filepath in directory.rglob('*'):
                if filepath.is_file():
                    total_size += filepath.stat().st_size
        except Exception as e:
            logger.error(f"Error calculating directory size for {directory}: {e}")
        
        return total_size
    
    @staticmethod
    def cleanup_empty_directories(base_directory: Path) -> int:
        """Remove empty directories and return count."""
        removed_count = 0
        
        try:
            # Walk through directories bottom-up
            for directory in sorted(base_directory.rglob('*'), key=lambda p: len(p.parts), reverse=True):
                if directory.is_dir() and directory != base_directory:
                    try:
                        # Try to remove if empty
                        directory.rmdir()
                        removed_count += 1
                        logger.info(f"Removed empty directory: {directory}")
                    except OSError:
                        # Directory not empty, skip
                        pass
        except Exception as e:
            logger.error(f"Error cleaning up empty directories: {e}")
        
        return removed_count
    
    @staticmethod
    async def create_backup(source_path: Path, backup_dir: Path) -> Optional[Path]:
        """Create backup of a file."""
        try:
            if not source_path.exists():
                return None
            
            backup_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{source_path.stem}_{timestamp}{source_path.suffix}"
            backup_path = backup_dir / backup_filename
            
            shutil.copy2(source_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None