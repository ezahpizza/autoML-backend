"""
MongoDB connection and database operations for AutoML platform.
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from pymongo import AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase
import logging

from config import settings

logger = logging.getLogger(__name__)


class MongoDB:
    """MongoDB client wrapper with async operations."""
    
    def __init__(self):
        self.client: Optional[AsyncMongoClient] = None
        self.database: Optional[AsyncDatabase] = None
        
    async def connect(self) -> None:
        """Establish connection to MongoDB."""
        try:
            self.client = AsyncMongoClient(settings.mongodb_url)
            self.database = self.client[settings.MONGODB_DB_NAME]
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {settings.MONGODB_DB_NAME}")
            
            # Create indexes for better performance
            await self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def _create_indexes(self) -> None:
        """Create database indexes for better query performance."""
        try:
            # Users collection indexes
            await self.database.users.create_index("user_id", unique=True)
            
            # EDA jobs collection indexes
            await self.database.eda_jobs.create_index([("user_id", 1), ("created_at", -1)])
            await self.database.eda_jobs.create_index("filename", unique=True)
            
            # Model jobs collection indexes
            await self.database.model_jobs.create_index([("user_id", 1), ("created_at", -1)])
            await self.database.model_jobs.create_index("filename", unique=True)
            
            # Predictions collection indexes
            await self.database.predictions.create_index([("user_id", 1), ("created_at", -1)])
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")
    
    def get_collection(self, collection_name: str) -> AsyncCollection:
        """Get a collection from the database."""
        if self.database is None:
            raise RuntimeError("Database not connected")
        return self.database[collection_name]
    
    # User Operations
    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user record."""
        user_data["created_at"] = datetime.now(timezone.utc)
        user_data["updated_at"] = datetime.now(timezone.utc)
        
        collection = self.get_collection("users")
        result = await collection.insert_one(user_data)
        
        return {"_id": str(result.inserted_id), **user_data}
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by user_id."""
        collection = self.get_collection("users")
        user = await collection.find_one({"user_id": user_id})
        
        if user:
            user["_id"] = str(user["_id"])
        
        return user
    
    async def update_user(self, user_id: str, update_data: Dict[str, Any]) -> bool:
        """Update user data."""
        update_data["updated_at"] = datetime.now(timezone.utc)
        
        collection = self.get_collection("users")
        result = await collection.update_one(
            {"user_id": user_id},
            {"$set": update_data}
        )
        
        return result.modified_count > 0
    
    # EDA Jobs Operations
    async def create_eda_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new EDA job record."""
        job_data["created_at"] = datetime.now(timezone.utc)
        
        collection = self.get_collection("eda_jobs")
        result = await collection.insert_one(job_data)
        
        return {"_id": str(result.inserted_id), **job_data}
    
    async def get_eda_jobs(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get EDA jobs for a user."""
        collection = self.get_collection("eda_jobs")
        cursor = collection.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
        
        jobs = []
        async for job in cursor:
            job["_id"] = str(job["_id"])
            jobs.append(job)
        
        return jobs
    
    async def delete_eda_job(self, filename: str) -> bool:
        """Delete EDA job by filename."""
        collection = self.get_collection("eda_jobs")
        result = await collection.delete_one({"filename": filename})
        
        return result.deleted_count > 0
    
    # Model Jobs Operations
    async def create_model_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new model job record."""
        job_data["created_at"] = datetime.now(timezone.utc)
        
        collection = self.get_collection("model_jobs")
        result = await collection.insert_one(job_data)
        
        return {"_id": str(result.inserted_id), **job_data}
    
    async def get_model_jobs(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get model jobs for a user."""
        collection = self.get_collection("model_jobs")
        cursor = collection.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
        
        jobs = []
        async for job in cursor:
            job["_id"] = str(job["_id"])
            jobs.append(job)
        
        return jobs
    
    async def get_model_job(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get model job by filename."""
        collection = self.get_collection("model_jobs")
        job = await collection.find_one({"filename": filename})
        
        if job:
            job["_id"] = str(job["_id"])
        
        return job
    
    async def delete_model_job(self, filename: str) -> bool:
        """Delete model job by filename."""
        collection = self.get_collection("model_jobs")
        result = await collection.delete_one({"filename": filename})
        
        return result.deleted_count > 0
    
    # Predictions Operations
    async def create_prediction(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new prediction record."""
        prediction_data["created_at"] = datetime.now(timezone.utc)
        
        collection = self.get_collection("predictions")
        result = await collection.insert_one(prediction_data)
        
        return {"_id": str(result.inserted_id), **prediction_data}
    
    async def get_predictions(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get predictions for a user."""
        collection = self.get_collection("predictions")
        cursor = collection.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
        
        predictions = []
        async for prediction in cursor:
            prediction["_id"] = str(prediction["_id"])
            predictions.append(prediction)
        
        return predictions
    
    # Cleanup Operations
    async def delete_user_data(self, user_id: str) -> Dict[str, int]:
        """Delete all data for a user."""
        deleted_counts = {}
        
        # Delete from all collections
        collections = ["eda_jobs", "model_jobs", "predictions"]
        
        for collection_name in collections:
            collection = self.get_collection(collection_name)
            result = await collection.delete_many({"user_id": user_id})
            deleted_counts[collection_name] = result.deleted_count
        
        # Delete user record
        user_collection = self.get_collection("users")
        user_result = await user_collection.delete_one({"user_id": user_id})
        deleted_counts["users"] = user_result.deleted_count
        
        return deleted_counts
    
    async def delete_old_records(self, hours: int = 24) -> Dict[str, int]:
        """Delete records older than specified hours."""
        cutoff_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_time = cutoff_time.replace(hour=cutoff_time.hour - hours)
        
        deleted_counts = {}
        
        # Delete old EDA jobs
        eda_collection = self.get_collection("eda_jobs")
        eda_result = await eda_collection.delete_many({"created_at": {"$lt": cutoff_time}})
        deleted_counts["eda_jobs"] = eda_result.deleted_count
        
        # Delete old model jobs
        model_collection = self.get_collection("model_jobs")
        model_result = await model_collection.delete_many({"created_at": {"$lt": cutoff_time}})
        deleted_counts["model_jobs"] = model_result.deleted_count
        
        # Delete old predictions
        pred_collection = self.get_collection("predictions")
        pred_result = await pred_collection.delete_many({"created_at": {"$lt": cutoff_time}})
        deleted_counts["predictions"] = pred_result.deleted_count
        
        return deleted_counts
    
    async def find(self, collection_name: str, filter: dict, limit: int = None, sort: list = None, one: bool = False):
        """Find one or many documents in a collection."""
        collection = self.get_collection(collection_name)
        if one:
            doc = await collection.find_one(filter)
            if doc and "_id" in doc:
                doc["_id"] = str(doc["_id"])
            return doc
        else:
            cursor = collection.find(filter)
            if sort:
                cursor = cursor.sort(sort)
            if limit:
                cursor = cursor.limit(limit)
            results = []
            async for doc in cursor:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                results.append(doc)
            return results

    async def update_document(self, collection_name: str, filter: dict, update: dict):
        """Update a document in a collection."""
        collection = self.get_collection(collection_name)
        result = await collection.update_one(filter, update)
        return result.modified_count > 0


# Global MongoDB instance
mongodb = MongoDB()