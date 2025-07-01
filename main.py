"""
FastAPI entrypoint for AutoML platform.
"""

import logging
from contextlib import asynccontextmanager

from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from db.mongodb import mongodb
from routes import train, eda, models, cleanup
from services.cleanup_service import CleanupService
from schemas.response_schemas import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting AutoML Platform API...")
    
    try:
        # Connect to MongoDB
        await mongodb.connect()
        logger.info("MongoDB connection established")
        
        # Run startup cleanup (files older than 24 hours)
        cleanup_service = CleanupService()
        cleanup_result = await cleanup_service.cleanup_old_files()
        logger.info(f"Startup cleanup completed: {cleanup_result}")
        
        # Ensure storage directories exist
        logger.info("Storage directories initialized")
        
        logger.info("AutoML Platform API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AutoML Platform API...")
    
    try:
        # Disconnect from MongoDB
        await mongodb.disconnect()
        logger.info("MongoDB connection closed")
        
        logger.info("AutoML Platform API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["settings.CORS_ORIGINS"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving plots and EDA reports
app.mount("/static/plots", StaticFiles(directory=str(settings.plots_dir)), name="plots")
app.mount("/static/eda", StaticFiles(directory=str(settings.eda_reports_dir)), name="eda_reports")

# Include route modules
app.include_router(train.router, prefix="/model", tags=["Model Training"])
app.include_router(eda.router, prefix="/eda", tags=["EDA Reports"])
app.include_router(models.router, prefix="/model", tags=["Model Management"])
app.include_router(cleanup.router, prefix="/cleanup", tags=["Cleanup"])


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AutoML Platform API",
        "version": settings.API_VERSION,
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        database_connected = False
        try:
            await mongodb.client.admin.command('ping')
            database_connected = True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
        
        # Test storage accessibility
        storage_accessible = all([
            settings.models_dir.exists(),
            settings.plots_dir.exists(),
            settings.eda_reports_dir.exists()
        ])
                
        return HealthResponse(
            status="healthy" if database_connected and storage_accessible else "degraded",
            database_connected=database_connected,
            storage_accessible=storage_accessible,
            version=settings.API_VERSION
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")
