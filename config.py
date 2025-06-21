"""
Application configuration settings for AutoML platform.
"""

from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # MongoDB Configuration
    MONGODB_URL: str = Field(env="MONGODB_URL")
    MONGODB_DB_NAME: str = Field(env="MONGODB_DB_NAME")
    
    # File Storage Configuration - computed properties
    @property
    def base_dir(self) -> Path:
        return Path(__file__).parent
    
    @property
    def storage_dir(self) -> Path:
        return self.base_dir / "storage"
    
    @property
    def models_dir(self) -> Path:
        return self.storage_dir / "models"
    
    @property
    def plots_dir(self) -> Path:
        return self.storage_dir / "plots"
    
    @property
    def eda_reports_dir(self) -> Path:
        return self.storage_dir / "eda_reports"
    
    # Dataset Limits (Render Free Tier Safe)
    MAX_DATASET_ROWS: int = Field(5000)
    MAX_DATASET_COLUMNS: int = Field(50)
    MAX_FILE_SIZE_MB: int = Field(20)
    
    # PyCaret Configuration
    PYCARET_TURBO_MODE: bool = Field(True)
    PYCARET_LIGHTWEIGHT_MODELS: List[str] = Field(
        default=[
            'lr',    # Logistic Regression
            'knn',   # K-Nearest Neighbors
            'nb',    # Naive Bayes
            'dt',    # Decision Tree
            'rf',    # Random Forest (limited)
            'xgboost',  # XGBoost
            'lightgbm'  # LightGBM
        ]
    )
    
    # File Cleanup Configuration
    FILE_RETENTION_HOURS: int = Field(24)
    
    # API Configuration
    API_TITLE: str = Field("AutoML Platform API")
    API_VERSION: str = Field("1.0.0")
    API_DESCRIPTION: str = Field("Backend for AutoML platform with PyCaret and EDA")
    CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:8080",
            "http://localhost:5173",
        ],
        env="CORS_ORIGINS"
    )
    
    @property
    def mongodb_url(self) -> str:
        """Get MongoDB connection URL."""
        return self.MONGODB_URL
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024
    
    def create_directories(self) -> None:
        """Create storage directories if they don't exist."""
        for directory in [self.storage_dir, self.models_dir, self.plots_dir, self.eda_reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

# Create directories on initialization
settings.create_directories()