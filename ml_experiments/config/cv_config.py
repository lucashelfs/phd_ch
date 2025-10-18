"""
Simple cross-validation configuration.

This module provides a minimal configuration class for cross-validation experiments,
following the KISS principle and scikit-learn patterns.
"""

from typing import List, Optional


class CrossValidationConfig:
    """Simple cross-validation configuration class."""

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
        scoring_metrics: Optional[List[str]] = None,
    ):
        """Initialize cross-validation configuration.

        Args:
            n_splits: Number of folds for K-Fold CV
            shuffle: Whether to shuffle data before splitting
            random_state: Random state for reproducibility
            scoring_metrics: List of scoring metrics to evaluate
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.scoring_metrics = scoring_metrics or [
            "neg_mean_absolute_error",
            "neg_mean_squared_error",
            "r2",
        ]

        # Basic validation
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    def to_dict(self):
        """Convert to dictionary for logging."""
        return {
            "n_splits": self.n_splits,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "scoring_metrics": self.scoring_metrics,
        }
