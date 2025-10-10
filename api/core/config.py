"""
Configuration management for the Real Estate Price Prediction API.

This module handles environment-based configuration with sensible defaults
and validation for production deployment.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import validator
from pydantic_settings import BaseSettings


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
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Performance Configuration
    max_request_size: int = 1024 * 1024  # 1MB
    request_timeout: int = 30  # seconds
    
    @validator("model_path", "model_features_path", "demographics_data_path")
    def validate_file_paths(cls, v):
        """Validate that required files exist."""
        if not Path(v).exists():
            raise ValueError(f"Required file not found: {v}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()
    
    class Config:
        env_prefix = "API_"
        case_sensitive = False
        protected_namespaces = ('settings_',)


# Global settings instance
settings = Settings()
