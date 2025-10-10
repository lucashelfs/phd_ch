"""
Model prediction service for the Real Estate Price Prediction API.

This module handles model loading, demographic data integration,
and prediction logic with comprehensive error handling.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

from .config import settings
from .exceptions import ModelLoadError, PredictionError, DemographicDataError


class HousePricePredictor:
    """
    House price prediction service with demographic data integration.
    
    This class handles model loading, feature engineering, and prediction
    with comprehensive error handling and validation.
    """
    
    def __init__(self):
        """Initialize the predictor with model and demographic data."""
        self.model: Optional[BaseEstimator] = None
        self.model_features: Optional[List[str]] = None
        self.demographics_data: Optional[pd.DataFrame] = None
        self._load_model()
        self._load_demographics_data()
    
    def _load_model(self) -> None:
        """Load the trained model and feature list."""
        try:
            # Load the pickled model
            model_path = Path(settings.model_path)
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load the model features
            features_path = Path(settings.model_features_path)
            with open(features_path, 'r') as f:
                self.model_features = json.load(f)
            
            if not self.model or not self.model_features:
                raise ModelLoadError("Model or features list is empty")
                
        except FileNotFoundError as e:
            raise ModelLoadError(f"Model file not found: {e}")
        except json.JSONDecodeError as e:
            raise ModelLoadError(f"Invalid model features JSON: {e}")
        except Exception as e:
            raise ModelLoadError(f"Unexpected error loading model: {e}")
    
    def _load_demographics_data(self) -> None:
        """Load demographic data for zipcode enrichment."""
        try:
            demographics_path = Path(settings.demographics_data_path)
            self.demographics_data = pd.read_csv(
                demographics_path,
                dtype={'zipcode': str}
            )
            
            if self.demographics_data.empty:
                raise DemographicDataError("Demographics data is empty")
                
        except FileNotFoundError as e:
            raise DemographicDataError(f"Demographics file not found: {e}")
        except Exception as e:
            raise DemographicDataError(f"Error loading demographics data: {e}")
    
    def _enrich_with_demographics(self, house_data: Dict[str, Union[str, float, int]]) -> Dict[str, Union[str, float, int]]:
        """
        Enrich house data with demographic information based on zipcode.
        
        Args:
            house_data: Dictionary containing house features including zipcode
            
        Returns:
            Dictionary with demographic features added
            
        Raises:
            DemographicDataError: If zipcode is missing or demographic data unavailable
        """
        zipcode = house_data.get('zipcode')
        if not zipcode:
            raise DemographicDataError("Zipcode is required for prediction")
        
        # Convert zipcode to string for consistent matching
        zipcode = str(zipcode)
        
        # Find demographic data for the zipcode
        demographic_row = self.demographics_data[
            self.demographics_data['zipcode'] == zipcode
        ]
        
        if demographic_row.empty:
            raise DemographicDataError(
                f"No demographic data found for zipcode: {zipcode}",
                zipcode=zipcode
            )
        
        # Convert demographic data to dictionary and merge with house data
        demographic_dict = demographic_row.iloc[0].to_dict()
        
        # Remove zipcode from demographic data to avoid duplication
        demographic_dict.pop('zipcode', None)
        
        # Merge house data with demographic data
        enriched_data = {**house_data, **demographic_dict}
        
        return enriched_data
    
    def _prepare_features(self, enriched_data: Dict[str, Union[str, float, int]]) -> pd.DataFrame:
        """
        Prepare features for model prediction.
        
        Args:
            enriched_data: Dictionary containing all features including demographics
            
        Returns:
            DataFrame with features in the correct order for model prediction
            
        Raises:
            PredictionError: If required features are missing
        """
        try:
            # Create DataFrame with model features in correct order
            feature_data = {}
            missing_features = []
            
            for feature in self.model_features:
                if feature in enriched_data:
                    feature_data[feature] = [enriched_data[feature]]
                else:
                    missing_features.append(feature)
            
            if missing_features:
                raise PredictionError(
                    f"Missing required features: {missing_features}",
                    details={"missing_features": missing_features}
                )
            
            # Create DataFrame
            features_df = pd.DataFrame(feature_data)
            
            # Ensure numeric types for all features
            for col in features_df.columns:
                try:
                    features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
                except Exception:
                    pass  # Keep original value if conversion fails
            
            # Check for NaN values
            if features_df.isnull().any().any():
                nan_columns = features_df.columns[features_df.isnull().any()].tolist()
                raise PredictionError(
                    f"Invalid numeric values in features: {nan_columns}",
                    details={"invalid_features": nan_columns}
                )
            
            return features_df
            
        except Exception as e:
            if isinstance(e, PredictionError):
                raise
            raise PredictionError(f"Error preparing features: {e}")
    
    def predict(self, house_data: Dict[str, Union[str, float, int]]) -> Dict[str, Union[float, str, int]]:
        """
        Make a house price prediction.
        
        Args:
            house_data: Dictionary containing house features
            
        Returns:
            Dictionary containing prediction and metadata
            
        Raises:
            PredictionError: If prediction fails
            DemographicDataError: If demographic data processing fails
        """
        try:
            # Enrich data with demographics
            enriched_data = self._enrich_with_demographics(house_data)
            
            # Prepare features for prediction
            features_df = self._prepare_features(enriched_data)
            
            # Make prediction
            prediction = self.model.predict(features_df)[0]
            
            # Ensure prediction is a valid number
            if not np.isfinite(prediction):
                raise PredictionError("Model returned invalid prediction value")
            
            # Return prediction with metadata
            return {
                "prediction": float(prediction),
                "model_version": settings.api_version,
                "model_type": type(self.model).__name__,
                "features_used": len(self.model_features),
                "zipcode": str(house_data.get('zipcode', ''))
            }
            
        except (DemographicDataError, PredictionError):
            raise
        except Exception as e:
            raise PredictionError(f"Unexpected error during prediction: {e}")
    
    def predict_minimal(self, house_data: Dict[str, Union[str, float, int]]) -> Dict[str, Union[float, str, int]]:
        """
        Make a prediction using only core house features (bonus endpoint).
        
        This method uses only the features that were originally selected
        from the house sales data, excluding demographic features.
        
        Args:
            house_data: Dictionary containing house features
            
        Returns:
            Dictionary containing prediction and metadata
        """
        # Core features from the original model training
        core_features = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'sqft_above', 'sqft_basement'
        ]
        
        # Filter to only core features
        core_data = {k: v for k, v in house_data.items() if k in core_features}
        
        # Still need zipcode for demographic enrichment
        if 'zipcode' in house_data:
            core_data['zipcode'] = house_data['zipcode']
        
        # Use regular predict method with filtered data
        result = self.predict(core_data)
        result["prediction_type"] = "minimal_features"
        result["core_features_used"] = len(core_features)
        
        return result
    
    def get_model_info(self) -> Dict[str, Union[str, int, List[str]]]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            "model_type": type(self.model).__name__ if self.model else "Unknown",
            "model_version": settings.api_version,
            "total_features": len(self.model_features) if self.model_features else 0,
            "features": self.model_features or [],
            "demographics_zipcodes": len(self.demographics_data) if self.demographics_data is not None else 0
        }


# Global predictor instance
predictor = HousePricePredictor()
