"""
Pydantic request schemas for API endpoints.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class UserInitRequest(BaseModel):
    """Request schema for user initialization."""
    
    user_id: str = Field(..., description="Unique user identifier from frontend auth")
    email: str = Field(..., description="User email address")
    name: Optional[str] = Field(None, description="User display name")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('user_id cannot be empty')
        return v.strip()
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if not v or '@' not in v:
            raise ValueError('Invalid email address')
        return v.lower().strip()


class EDAGenerateRequest(BaseModel):
    """Request schema for EDA report generation."""
    
    user_id: str = Field(..., description="User identifier")
    dataset_name: Optional[str] = Field(None, description="Optional dataset name")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('user_id cannot be empty')
        return v.strip()


class ModelTrainRequest(BaseModel):
    """Request schema for model training."""
    
    user_id: str = Field(..., description="User identifier")
    target_column: str = Field(..., description="Target column name for training")
    dataset_name: Optional[str] = Field(None, description="Optional dataset name")
    model_types: Optional[List[str]] = Field(None, description="Specific models to train")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('user_id cannot be empty')
        return v.strip()
    
    @field_validator('target_column')
    @classmethod
    def validate_target_column(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('target_column cannot be empty')
        return v.strip()
    
    @field_validator('model_types')
    @classmethod
    def validate_model_types(cls, v):
        if v is not None:
            allowed_models = ['lr', 'knn', 'nb', 'dt', 'rf', 'xgboost', 'lightgbm']
            for model in v:
                if model not in allowed_models:
                    raise ValueError(f'Invalid model type: {model}. Allowed: {allowed_models}')
        return v


class PredictionRequest(BaseModel):
    """Request schema for making predictions."""
    
    user_id: str = Field(..., description="User identifier")
    model_filename: str = Field(..., description="Model filename to use for prediction")
    input_data: Dict[str, Any] = Field(..., description="Input data for prediction")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('user_id cannot be empty')
        return v.strip()
    
    @field_validator('model_filename')
    @classmethod
    def validate_model_filename(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('model_filename cannot be empty')
        if not v.endswith('.pkl'):
            raise ValueError('model_filename must end with .pkl')
        return v.strip()
    
    @field_validator('input_data')
    @classmethod
    def validate_input_data(cls, v):
        if not v or len(v) == 0:
            raise ValueError('input_data cannot be empty')
        return v


class CleanupUserRequest(BaseModel):
    """Request schema for user cleanup operations."""
    
    user_id: str = Field(..., description="User identifier")
    confirm: bool = Field(..., description="Confirmation flag for cleanup")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('user_id cannot be empty')
        return v.strip()
    
    @field_validator('confirm')
    @classmethod
    def validate_confirm(cls, v):
        if not v:
            raise ValueError('Confirmation required for cleanup operation')
        return v
    
    
class CompareModelsRequest(BaseModel):
    """Request schema for user model comparison operations."""

    user_id: str
    model_filenames: Optional[List[str]] = None