"""
Pydantic models for API v1 request and response validation.

This module defines the data models used for request validation
and response formatting in API version 1.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class HouseFeaturesRequest(BaseModel):
    """Request model for house features used in prediction."""
    
    # Core house features
    bedrooms: int = Field(..., ge=0, le=20, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, le=10, description="Number of bathrooms")
    sqft_living: int = Field(..., ge=100, le=20000, description="Square feet of living space")
    sqft_lot: int = Field(..., ge=500, le=1000000, description="Square feet of lot")
    floors: float = Field(..., ge=1, le=5, description="Number of floors")
    
    # Additional house features
    waterfront: Optional[int] = Field(0, ge=0, le=1, description="Waterfront property (0 or 1)")
    view: Optional[int] = Field(0, ge=0, le=4, description="View rating (0-4)")
    condition: Optional[int] = Field(3, ge=1, le=5, description="Condition rating (1-5)")
    grade: Optional[int] = Field(7, ge=1, le=13, description="Grade rating (1-13)")
    sqft_above: int = Field(..., ge=100, le=20000, description="Square feet above ground")
    sqft_basement: Optional[int] = Field(0, ge=0, le=10000, description="Square feet of basement")
    yr_built: int = Field(..., ge=1900, le=2025, description="Year built")
    yr_renovated: Optional[int] = Field(0, ge=0, le=2025, description="Year renovated (0 if never)")
    
    # Location features
    zipcode: str = Field(..., min_length=5, max_length=5, description="5-digit zipcode")
    lat: float = Field(..., ge=47.0, le=48.0, description="Latitude")
    long: float = Field(..., ge=-123.0, le=-121.0, description="Longitude")
    sqft_living15: Optional[int] = Field(None, ge=100, le=20000, description="Living space of 15 nearest neighbors")
    sqft_lot15: Optional[int] = Field(None, ge=500, le=1000000, description="Lot size of 15 nearest neighbors")
    
    @validator('zipcode')
    def validate_zipcode(cls, v):
        """Validate zipcode format."""
        if not v.isdigit():
            raise ValueError('Zipcode must contain only digits')
        return v
    
    @validator('yr_renovated')
    def validate_renovation_year(cls, v, values):
        """Validate renovation year is after build year."""
        if v and v > 0:
            yr_built = values.get('yr_built')
            if yr_built and v < yr_built:
                raise ValueError('Renovation year cannot be before build year')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "bedrooms": 4,
                "bathrooms": 2.5,
                "sqft_living": 2630,
                "sqft_lot": 4501,
                "floors": 2.0,
                "waterfront": 0,
                "view": 0,
                "condition": 3,
                "grade": 8,
                "sqft_above": 2630,
                "sqft_basement": 0,
                "yr_built": 2015,
                "yr_renovated": 0,
                "zipcode": "98028",
                "lat": 47.7748,
                "long": -122.244,
                "sqft_living15": 2380,
                "sqft_lot15": 4599
            }
        }


class MinimalHouseFeaturesRequest(BaseModel):
    """Request model for minimal house features (bonus endpoint)."""
    
    # Core features only
    bedrooms: int = Field(..., ge=0, le=20, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, le=10, description="Number of bathrooms")
    sqft_living: int = Field(..., ge=100, le=20000, description="Square feet of living space")
    sqft_lot: int = Field(..., ge=500, le=1000000, description="Square feet of lot")
    floors: float = Field(..., ge=1, le=5, description="Number of floors")
    sqft_above: int = Field(..., ge=100, le=20000, description="Square feet above ground")
    sqft_basement: Optional[int] = Field(0, ge=0, le=10000, description="Square feet of basement")
    
    # Still need zipcode for demographic enrichment
    zipcode: str = Field(..., min_length=5, max_length=5, description="5-digit zipcode")
    
    @validator('zipcode')
    def validate_zipcode(cls, v):
        """Validate zipcode format."""
        if not v.isdigit():
            raise ValueError('Zipcode must contain only digits')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "bedrooms": 4,
                "bathrooms": 2.5,
                "sqft_living": 2630,
                "sqft_lot": 4501,
                "floors": 2.0,
                "sqft_above": 2630,
                "sqft_basement": 0,
                "zipcode": "98028"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for house price predictions."""
    
    prediction: float = Field(..., description="Predicted house price in USD")
    model_version: str = Field(..., description="API/Model version")
    model_type: str = Field(..., description="Type of machine learning model used")
    features_used: int = Field(..., description="Number of features used in prediction")
    zipcode: str = Field(..., description="Zipcode used for prediction")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z", description="Prediction timestamp")
    prediction_type: Optional[str] = Field(None, description="Type of prediction (e.g., minimal_features)")
    core_features_used: Optional[int] = Field(None, description="Number of core features used (minimal endpoint)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 450000.0,
                "model_version": "1.0.0",
                "model_type": "KNeighborsRegressor",
                "features_used": 33,
                "zipcode": "98028",
                "timestamp": "2025-01-06T23:26:57Z"
            }
        }


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    model_type: str = Field(..., description="Type of machine learning model")
    model_version: str = Field(..., description="API/Model version")
    total_features: int = Field(..., description="Total number of features the model uses")
    features: List[str] = Field(..., description="List of feature names")
    demographics_zipcodes: int = Field(..., description="Number of zipcodes with demographic data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "KNeighborsRegressor",
                "model_version": "1.0.0",
                "total_features": 33,
                "features": ["bedrooms", "bathrooms", "sqft_living", "..."],
                "demographics_zipcodes": 70
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z", description="Health check timestamp")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether the model is loaded successfully")
    demographics_loaded: bool = Field(..., description="Whether demographic data is loaded")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-06T23:26:57Z",
                "version": "1.0.0",
                "model_loaded": True,
                "demographics_loaded": True
            }
        }


class ErrorResponse(BaseModel):
    """Response model for API errors."""
    
    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z", description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "VALIDATION_ERROR",
                "message": "Invalid input data",
                "details": {"field": "zipcode", "issue": "must be 5 digits"},
                "timestamp": "2025-01-06T23:26:57Z"
            }
        }
