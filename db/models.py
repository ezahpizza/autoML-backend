"""
Pydantic models for MongoDB document schemas.
"""

from bson import ObjectId
from datetime import datetime, UTC
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue

class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic v2."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> ObjectId:
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: Any,
        handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Generate base schema and update type
        schema = handler(core_schema)
        schema.update(type="string", examples=["507f1f77bcf86cd799439011"])
        return schema

class User(BaseModel):
    """User document model."""
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str = Field(..., description="Unique user identifier from frontend auth")
    email: str = Field(..., description="User email address")
    name: Optional[str] = Field(None, description="User display name")
    created_at: datetime = Field(default_factory=datetime.now(UTC))
    updated_at: datetime = Field(default_factory=datetime.now(UTC))
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class EDAJob(BaseModel):
    """EDA job document model."""
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str = Field(..., description="User who created this EDA job")
    filename: str = Field(..., description="Generated HTML report filename")
    dataset_name: str = Field(..., description="Original dataset name")
    dataset_rows: int = Field(..., description="Number of rows in dataset")
    dataset_columns: int = Field(..., description="Number of columns in dataset")
    file_size: int = Field(..., description="File size in bytes")
    status: str = Field(default="completed", description="Job status")
    created_at: datetime = Field(default_factory=datetime.now(UTC))
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ModelJob(BaseModel):
    """Model training job document model."""
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str = Field(..., description="User who created this model job")
    filename: str = Field(..., description="Generated model pickle filename")
    dataset_name: str = Field(..., description="Original dataset name")
    target_column: str = Field(..., description="Target column for training")
    model_type: str = Field(..., description="Type of ML problem (classification/regression)")
    best_model: str = Field(..., description="Best performing model name")
    best_model_score: float = Field(..., description="Best model performance score")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Model evaluation metrics")
    plot_filenames: List[str] = Field(default_factory=list, description="Generated plot filenames")
    dataset_rows: int = Field(..., description="Number of rows in dataset")
    dataset_columns: int = Field(..., description="Number of columns in dataset")
    training_time: Optional[float] = Field(None, description="Training time in seconds")
    status: str = Field(default="completed", description="Job status")
    created_at: datetime = Field(default_factory=datetime.now(UTC))
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class Prediction(BaseModel):
    """Prediction request document model."""
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str = Field(..., description="User who made the prediction")
    model_filename: str = Field(..., description="Model used for prediction")
    input_data: Dict[str, Any] = Field(..., description="Input data for prediction")
    predictions: List[Any] = Field(..., description="Model predictions")
    prediction_probabilities: Optional[List[List[float]]] = Field(None, description="Prediction probabilities")
    created_at: datetime = Field(default_factory=datetime.now(UTC))
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class CleanupLog(BaseModel):
    """Cleanup operation log document model."""
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    operation_type: str = Field(..., description="Type of cleanup operation")
    user_id: Optional[str] = Field(None, description="User ID if user-specific cleanup")
    files_deleted: List[str] = Field(default_factory=list, description="List of deleted files")
    records_deleted: Dict[str, int] = Field(default_factory=dict, description="Count of deleted records per collection")
    created_at: datetime = Field(default_factory=datetime.now(UTC))
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}