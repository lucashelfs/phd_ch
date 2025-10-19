#!/usr/bin/env python3
"""
Database initialization script for prediction logging.

This script creates the SQLite database and schema for storing
prediction inputs and outputs. It's designed to be run once
during container initialization.
"""

import sqlite3
import sys
from pathlib import Path


def initialize_prediction_database():
    """Initialize the prediction logging database with schema and indexes."""
    db_path = "/app/logs/predictions.db"

    print(f"Initializing prediction database at {db_path}")

    try:
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Connect to database (creates file if it doesn't exist)
        conn = sqlite3.connect(db_path)

        # Create main table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                -- Primary key and metadata
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_type TEXT NOT NULL,
                processing_time_ms REAL,

                -- Input features (structured)
                bedrooms INTEGER,
                bathrooms REAL,
                sqft_living INTEGER,
                sqft_lot INTEGER,
                floors REAL,
                waterfront INTEGER,
                view INTEGER,
                condition INTEGER,
                grade INTEGER,
                sqft_above INTEGER,
                sqft_basement INTEGER,
                yr_built INTEGER,
                yr_renovated INTEGER,
                zipcode TEXT,
                lat REAL,
                long REAL,
                sqft_living15 INTEGER,
                sqft_lot15 INTEGER,

                -- Prediction output
                predicted_price REAL,
                features_used INTEGER,
                prediction_type TEXT
            )
        """)

        # Create indexes for common queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON prediction_logs(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_zipcode ON prediction_logs(zipcode)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_endpoint ON prediction_logs(endpoint)"
        )

        # Commit changes
        conn.commit()
        conn.close()

        print("✅ Prediction logging database initialized successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to initialize prediction database: {e}")
        return False


if __name__ == "__main__":
    success = initialize_prediction_database()
    sys.exit(0 if success else 1)
