"""
Naming utilities for generating unique filenames and identifiers.
"""

import uuid
import re
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path


class NamingUtils:
    """Utilities for generating consistent, unique filenames."""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename by removing invalid characters."""
        # Remove file extension
        name = Path(filename).stem
        
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[<>:"/\\|?*\s]', '_', name)
        
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure filename is not empty
        if not sanitized:
            sanitized = "dataset"
        
        # Limit length
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
        
        return sanitized
    
    @staticmethod
    def generate_model_filename(user_id: str, dataset_name: str) -> str:
        """Generate unique filename for trained models."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        sanitized_dataset = NamingUtils.sanitize_filename(dataset_name)
        
        return f"{user_id}_{sanitized_dataset}_{timestamp}_{unique_id}.pkl"
    
    @staticmethod
    def generate_plot_filename(user_id: str, model_name: str, plot_type: str) -> str:
        """Generate unique filename for evaluation plots."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        sanitized_model = NamingUtils.sanitize_filename(model_name)
        sanitized_plot_type = NamingUtils.sanitize_filename(plot_type)
        
        return f"{user_id}_{sanitized_model}_{sanitized_plot_type}_{timestamp}_{unique_id}.png"
    
    @staticmethod
    def generate_eda_filename(user_id: str, dataset_name: str) -> str:
        """Generate unique filename for EDA reports."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        sanitized_dataset = NamingUtils.sanitize_filename(dataset_name)
        
        return f"{user_id}_{sanitized_dataset}_{timestamp}_{unique_id}.html"
    
    @staticmethod
    def generate_temp_filename(user_id: str, original_filename: str) -> str:
        """Generate temporary filename for uploaded files."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        extension = Path(original_filename).suffix
        sanitized_name = NamingUtils.sanitize_filename(original_filename)
        
        return f"temp_{user_id}_{sanitized_name}_{timestamp}_{unique_id}{extension}"
    
    @staticmethod
    def extract_user_id_from_filename(filename: str) -> Optional[str]:
        """Extract user ID from filename if it follows naming convention."""
        try:
            # Assuming format: {user_id}_{other_parts}
            parts = filename.split('_')
            if len(parts) >= 2:
                return parts[0]
            return None
        except Exception:
            return None
    
    @staticmethod
    def extract_timestamp_from_filename(filename: str) -> Optional[datetime]:
        """Extract timestamp from filename if it follows naming convention."""
        try:
            # Look for timestamp pattern: YYYYMMDD_HHMMSS
            timestamp_pattern = r'(\d{8}_\d{6})'
            match = re.search(timestamp_pattern, filename)
            
            if match:
                timestamp_str = match.group(1)
                return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            
            return None
        except Exception:
            return None
    
    @staticmethod
    def parse_model_filename(filename: str) -> dict:
        """Parse model filename to extract components."""
        try:
            # Expected format: {user_id}_{dataset}_{timestamp}_{unique_id}.pkl
            name_without_ext = Path(filename).stem
            parts = name_without_ext.split('_')
            
            if len(parts) >= 4:
                user_id = parts[0]
                # Dataset name might contain underscores, so join middle parts
                dataset_name = '_'.join(parts[1:-2])
                timestamp_str = f"{parts[-2]}_{parts[-1][:6]}"  # Last part contains unique_id
                unique_id = parts[-1][6:] if len(parts[-1]) > 6 else parts[-1]
                
                return {
                    "user_id": user_id,
                    "dataset_name": dataset_name,
                    "timestamp": timestamp_str,
                    "unique_id": unique_id,
                    "filename": filename
                }
            
            return {"filename": filename}
            
        except Exception:
            return {"filename": filename}
    
    @staticmethod
    def parse_plot_filename(filename: str) -> dict:
        """Parse plot filename to extract components."""
        try:
            # Expected format: {user_id}_{model}_{plot_type}_{timestamp}_{unique_id}.png
            name_without_ext = Path(filename).stem
            parts = name_without_ext.split('_')
            
            if len(parts) >= 5:
                user_id = parts[0]
                # Model and plot type might contain underscores
                model_name = parts[1]
                plot_type = parts[2]
                timestamp_str = f"{parts[-2]}_{parts[-1][:6]}"
                unique_id = parts[-1][6:] if len(parts[-1]) > 6 else parts[-1]
                
                return {
                    "user_id": user_id,
                    "model_name": model_name,
                    "plot_type": plot_type,
                    "timestamp": timestamp_str,
                    "unique_id": unique_id,
                    "filename": filename
                }
            
            return {"filename": filename}
            
        except Exception:
            return {"filename": filename}
    
    @staticmethod
    def parse_eda_filename(filename: str) -> dict:
        """Parse EDA filename to extract components."""
        try:
            # Expected format: {user_id}_{dataset}_{timestamp}_{unique_id}.html
            name_without_ext = Path(filename).stem
            parts = name_without_ext.split('_')
            
            if len(parts) >= 4:
                user_id = parts[0]
                # Dataset name might contain underscores
                dataset_name = '_'.join(parts[1:-2])
                timestamp_str = f"{parts[-2]}_{parts[-1][:6]}"
                unique_id = parts[-1][6:] if len(parts[-1]) > 6 else parts[-1]
                
                return {
                    "user_id": user_id,
                    "dataset_name": dataset_name,
                    "timestamp": timestamp_str,
                    "unique_id": unique_id,
                    "filename": filename
                }
            
            return {"filename": filename}
            
        except Exception:
            return {"filename": filename}
    
    @staticmethod
    def generate_unique_id() -> str:
        """Generate a short unique identifier."""
        return str(uuid.uuid4())[:8]
    
    @staticmethod
    def is_valid_filename(filename: str) -> bool:
        """Check if filename is valid for the platform."""
        if not filename:
            return False
        
        # Check for invalid characters
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        if any(char in filename for char in invalid_chars):
            return False
        
        # Check length
        if len(filename) > 255:
            return False
        
        return True
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate API key for future authentication needs."""
        return f"aml_{uuid.uuid4().hex}"