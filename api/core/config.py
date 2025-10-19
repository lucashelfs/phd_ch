"""
Configuration management for the Real Estate Price Prediction API.

This module handles environment-based configuration with sensible defaults
and validation for production deployment.
"""

from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Configuration
    api_title: str = "Real Estate Price Prediction API"
    api_description: str = "Production-ready REST API for real estate price prediction"
    api_version: str = "1.0.0"
    debug: bool = False

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Model Configuration
    model_path: str = "model/model.pkl"
    model_features_path: str = "model/model_features.json"
    demographics_data_path: str = "data/zipcode_demographics.csv"

    # MLflow Configuration
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_id: str = "1"
    champion_metric: str = "test_r2"
    champion_model_mlflow_uri: Optional[str] = None

    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"

    # Performance Configuration
    max_request_size: int = 1024 * 1024  # 1MB
    request_timeout: int = 30  # seconds

    # Prediction Logging Configuration
    enable_prediction_logging: bool = True

    @field_validator("demographics_data_path")
    @classmethod
    def validate_file_paths(cls, v):
        """Validate that required files exist."""
        if not Path(v).exists():
            raise ValueError(f"Required file not found: {v}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()

    model_config = SettingsConfigDict(
        env_prefix="", case_sensitive=False, protected_namespaces=("settings_",)
    )


# Global settings instance
settings = Settings()
