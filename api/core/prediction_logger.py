"""
Prediction logging module for storing model inputs and outputs.

This module provides async logging functionality to store prediction
requests and responses in a structured SQLite database.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import aiosqlite

from .config import settings


class PredictionLogger:
    """
    Async logger for storing prediction inputs and outputs.

    Uses SQLite database with structured schema to store all prediction
    data for analysis and monitoring purposes.
    """

    def __init__(self, db_path: str = "/app/logs/predictions.db", enabled: bool = True):
        """
        Initialize the prediction logger.

        Args:
            db_path: Path to SQLite database file
            enabled: Whether logging is enabled
        """
        self.db_path = db_path
        self.enabled = enabled
        self._initialized = False

        if self.enabled:
            # Ensure directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """Verify database connection and mark as initialized."""
        if not self.enabled or self._initialized:
            return

        try:
            # Just verify the database exists and is accessible
            async with aiosqlite.connect(self.db_path) as db:
                # Simple query to verify database is accessible
                await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_logs'"
                )

            self._initialized = True
            print("Prediction logging database connection verified")

        except Exception as e:
            print(f"Failed to connect to prediction logging database: {e}")
            self.enabled = False

    async def log_prediction(
        self,
        input_data: Dict[str, Any],
        prediction_result: Dict[str, Any],
        endpoint: str,
        processing_time_ms: Optional[float] = None,
    ):
        """
        Log a prediction request and response.

        Args:
            input_data: Original input features from request
            prediction_result: Prediction response data
            endpoint: API endpoint used (e.g., '/v1/predict')
            processing_time_ms: Processing time in milliseconds
        """
        if not self.enabled:
            return

        try:
            await self.initialize()

            # Extract data with safe defaults
            timestamp = datetime.utcnow().isoformat() + "Z"

            # Input features (handle both full and minimal requests)
            bedrooms = input_data.get("bedrooms")
            bathrooms = input_data.get("bathrooms")
            sqft_living = input_data.get("sqft_living")
            sqft_lot = input_data.get("sqft_lot")
            floors = input_data.get("floors")
            waterfront = input_data.get("waterfront")
            view = input_data.get("view")
            condition = input_data.get("condition")
            grade = input_data.get("grade")
            sqft_above = input_data.get("sqft_above")
            sqft_basement = input_data.get("sqft_basement")
            yr_built = input_data.get("yr_built")
            yr_renovated = input_data.get("yr_renovated")
            zipcode = input_data.get("zipcode")
            lat = input_data.get("lat")
            long = input_data.get("long")
            sqft_living15 = input_data.get("sqft_living15")
            sqft_lot15 = input_data.get("sqft_lot15")

            # Prediction results
            predicted_price = prediction_result.get("prediction")
            model_version = prediction_result.get("model_version")
            model_type = prediction_result.get("model_type")
            features_used = prediction_result.get("features_used")
            prediction_type = prediction_result.get("prediction_type", "full")

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO prediction_logs (
                        timestamp, endpoint, model_version, model_type, processing_time_ms,
                        bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                        waterfront, view, condition, grade, sqft_above, sqft_basement,
                        yr_built, yr_renovated, zipcode, lat, long,
                        sqft_living15, sqft_lot15,
                        predicted_price, features_used, prediction_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        timestamp,
                        endpoint,
                        model_version,
                        model_type,
                        processing_time_ms,
                        bedrooms,
                        bathrooms,
                        sqft_living,
                        sqft_lot,
                        floors,
                        waterfront,
                        view,
                        condition,
                        grade,
                        sqft_above,
                        sqft_basement,
                        yr_built,
                        yr_renovated,
                        zipcode,
                        lat,
                        long,
                        sqft_living15,
                        sqft_lot15,
                        predicted_price,
                        features_used,
                        prediction_type,
                    ),
                )
                await db.commit()

        except Exception as e:
            # Log error but don't fail the prediction
            print(f"Failed to log prediction: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics about logged predictions.

        Returns:
            Dictionary with logging statistics
        """
        if not self.enabled:
            return {"enabled": False}

        try:
            await self.initialize()

            async with aiosqlite.connect(self.db_path) as db:
                # Total predictions
                cursor = await db.execute("SELECT COUNT(*) FROM prediction_logs")
                total_predictions = (await cursor.fetchone())[0]

                # Predictions by endpoint
                cursor = await db.execute("""
                    SELECT endpoint, COUNT(*)
                    FROM prediction_logs
                    GROUP BY endpoint
                """)
                by_endpoint = dict(await cursor.fetchall())

                # Recent predictions (last 24 hours)
                cursor = await db.execute("""
                    SELECT COUNT(*)
                    FROM prediction_logs
                    WHERE datetime(timestamp) > datetime('now', '-1 day')
                """)
                recent_predictions = (await cursor.fetchone())[0]

                return {
                    "enabled": True,
                    "total_predictions": total_predictions,
                    "predictions_by_endpoint": by_endpoint,
                    "recent_predictions_24h": recent_predictions,
                }

        except Exception as e:
            return {"enabled": True, "error": str(e)}


# Global logger instance
prediction_logger = PredictionLogger(
    enabled=getattr(settings, "enable_prediction_logging", True)
)
