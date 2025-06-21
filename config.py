"""
Application configuration settings for AutoML platform.
"""

import os
from pathlib import Path
from typing import Optional


class Settings:
    """Application settings and configuration."""
    
    # MongoDB Configuration
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "automl_platform")
    
    # File Storage Configuration
    BASE_DIR: Path = Path(__file__).parent
    STORAGE_DIR: Path = BASE_DIR / "storage"
    MODELS_DIR: Path = STORAGE_DIR / "models"
    PLOTS_DIR: Path = STORAGE_DIR / "plots"
    EDA_REPORTS_DIR: Path = STORAGE_DIR / "eda_reports"
    
    # Dataset Limits (Render Free Tier Safe)
    MAX_DATASET_ROWS: int = int(os.getenv("MAX_DATASET_ROWS", "5000"))
    MAX_DATASET_COLUMNS: int = int(os.getenv("MAX_DATASET_COLUMNS", "50"))
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    
    # PyCaret Configuration
    PYCARET_TURBO_MODE: bool = True
    PYCARET_LIGHTWEIGHT_MODELS: list = [
        'lr',    # Logistic Regression
        'knn',   # K-Nearest Neighbors
        'nb',    # Naive Bayes
        'dt',    # Decision Tree
        'rf',    # Random Forest (limited)
        'xgboost',  # XGBoost
        'lightgbm'  # LightGBM
    ]
    
    # File Cleanup Configuration
    FILE_RETENTION_HOURS: int = int(os.getenv("FILE_RETENTION_HOURS", "24"))
    
    # API Configuration
    API_TITLE: str = "AutoML Platform API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Backend for AutoML platform with PyCaret and EDA"
    CORS_ORIGINS: list = [
        "http://localhost:8080",
        "http://localhost:5173",
    ]
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    RELOAD: bool = os.getenv("ENVIRONMENT", "development") == "development"
    
    def __init__(self):
        """Initialize settings and create necessary directories."""
        self.create_directories()
    
    def create_directories(self) -> None:
        """Create storage directories if they don't exist."""
        for directory in [self.STORAGE_DIR, self.MODELS_DIR, self.PLOTS_DIR, self.EDA_REPORTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def mongodb_url(self) -> str:
        """Get MongoDB connection URL."""
        return self.MONGODB_URL
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


# Global settings instance
settings = Settings()