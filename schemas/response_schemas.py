"""
Pydantic response schemas for API endpoints.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """Base response schema with common fields."""
    
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Response timestamp")


class EDAResponse(BaseResponse):
    """Response schema for EDA report generation."""
    
    filename: str = Field(..., description="Generated report filename")
    report_url: str = Field(..., description="URL to view the report")
    dataset_name: str = Field(..., description="Original dataset name")
    dataset_rows: int = Field(..., description="Number of rows processed")
    dataset_columns: int = Field(..., description="Number of columns processed")
    file_size: int = Field(..., description="Report file size in bytes")


class ModelTrainResponse(BaseResponse):
    """Response schema for model training."""
    
    filename: str = Field(..., description="Generated model filename")
    download_url: str = Field(..., description="URL to download the model")
    dataset_name: str = Field(..., description="Original dataset name")
    target_column: str = Field(..., description="Target column used")
    model_type: str = Field(..., description="ML problem type (classification/regression)")
    best_model: str = Field(..., description="Best performing model name")
    best_model_score: float = Field(..., description="Best model performance score")
    metrics: Dict[str, Any] = Field(..., description="Detailed model metrics")
    plot_urls: List[str] = Field(..., description="URLs to evaluation plots")
    training_time: float = Field(..., description="Training time in seconds")


class PredictionResponse(BaseResponse):
    """Response schema for predictions."""
    
    predictions: List[Any] = Field(..., description="Model predictions")
    prediction_probabilities: Optional[List[List[float]]] = Field(None, description="Prediction probabilities")
    model_used: str = Field(..., description="Model filename used")
    input_features: Dict[str, Any] = Field(..., description="Input features used")


class ModelListItem(BaseModel):
    """Model list item schema."""
    
    filename: str = Field(..., description="Model filename")
    dataset_name: str = Field(..., description="Original dataset name")
    target_column: str = Field(..., description="Target column")
    best_model: str = Field(..., description="Best model name")
    best_model_score: float = Field(..., description="Best model score")
    created_at: datetime = Field(..., description="Creation timestamp")
    download_url: str = Field(..., description="Download URL")


class ModelListResponse(BaseResponse):
    """Response schema for model listing."""
    
    models: List[ModelListItem] = Field(..., description="List of user models")
    total_count: int = Field(..., description="Total number of models")


class PlotListItem(BaseModel):
    """Plot list item schema."""
    
    filename: str = Field(..., description="Plot filename")
    plot_type: str = Field(..., description="Type of plot")
    model_name: str = Field(..., description="Associated model name")
    created_at: datetime = Field(..., description="Creation timestamp")
    view_url: str = Field(..., description="View URL")


class PlotListResponse(BaseResponse):
    """Response schema for plot listing."""
    
    plots: List[PlotListItem] = Field(..., description="List of plots")
    total_count: int = Field(..., description="Total number of plots")


class CleanupResponse(BaseResponse):
    """Response schema for cleanup operations."""
    
    files_deleted: List[str] = Field(..., description="List of deleted files")
    records_deleted: Dict[str, int] = Field(..., description="Count of deleted records per collection")
    total_files_deleted: int = Field(..., description="Total number of files deleted")
    total_records_deleted: int = Field(..., description="Total number of records deleted")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    database_connected: bool = Field(..., description="Database connection status")
    storage_accessible: bool = Field(..., description="Storage accessibility status")
    version: str = Field(..., description="API version")
