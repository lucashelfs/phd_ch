"""
Custom exception handling for the Real Estate Price Prediction API.

This module defines custom exceptions and error handling patterns
for consistent error responses across the API.
"""

from typing import Any, Dict, Optional


class APIException(Exception):
    """Base exception class for API-specific errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ModelLoadError(APIException):
    """Raised when model loading fails."""
    
    def __init__(self, message: str = "Failed to load prediction model"):
        super().__init__(
            message=message,
            status_code=503,
            error_code="MODEL_LOAD_ERROR"
        )


class PredictionError(APIException):
    """Raised when model prediction fails."""
    
    def __init__(self, message: str = "Prediction failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="PREDICTION_ERROR",
            details=details
        )


class DataValidationError(APIException):
    """Raised when input data validation fails."""
    
    def __init__(self, message: str = "Data validation failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=422,
            error_code="VALIDATION_ERROR",
            details=details
        )


class DemographicDataError(APIException):
    """Raised when demographic data processing fails."""
    
    def __init__(self, message: str = "Demographic data processing failed", zipcode: Optional[str] = None):
        details = {"zipcode": zipcode} if zipcode else {}
        super().__init__(
            message=message,
            status_code=400,
            error_code="DEMOGRAPHIC_DATA_ERROR",
            details=details
        )


class ConfigurationError(APIException):
    """Raised when configuration validation fails."""
    
    def __init__(self, message: str = "Configuration error"):
        super().__init__(
            message=message,
            status_code=500,
            error_code="CONFIGURATION_ERROR"
        )
