import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    median_absolute_error, explained_variance_score
)
from sklearn.preprocessing import LabelEncoder

from model.delay_duration.config import XGBOOST_PARAMS, XGBOOST_VERBOSE


class DelayDurationModel:
    """
    XGBoost-based model for predicting flight delay duration.

    This model predicts the duration (in minutes) of a flight delay
    for flights that are delayed by more than 15 minutes (regression).
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the delay duration model.

        Args:
            params: XGBoost hyperparameters. If None, uses default from config.
        """
        self.params = params if params is not None else XGBOOST_PARAMS.copy()
        self.model = None
        self.feature_names = None
        self.is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True
    ) -> 'DelayDurationModel':
        """
        Train the model on training data.

        Args:
            X_train: Training features
            y_train: Training target (delay duration in minutes)
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            verbose: Whether to print training progress

        Returns:
            Self (fitted model)
        """
        if verbose:
            print("\n" + "=" * 70)
            print("TRAINING XGBOOST REGRESSOR")
            print("=" * 70)

        if verbose:
            print("\nModel parameters:")
            for key, val in self.params.items():
                print(f"  {key}: {val}")

        # Create model
        self.model = xgb.XGBRegressor(**self.params)

        # Store feature names
        self.feature_names = list(X_train.columns)

        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Train model
        if verbose:
            print("\nTraining model...")

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=XGBOOST_VERBOSE if verbose else 0
        )

        self.is_fitted = True

        if verbose:
            print("\nTraining complete!")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict delay duration for input samples.

        Args:
            X: Features DataFrame

        Returns:
            Array of predicted delay durations (in minutes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Returns:
            DataFrame with features and their importance scores, sorted descending
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")

        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return feature_importance

    def save(self, model_path: Path, encoders: Optional[Dict[str, LabelEncoder]] = None) -> None:
        """
        Save the trained model and optionally the label encoders.

        Args:
            model_path: Path to save the model file (.json)
            encoders: Dictionary of label encoders to save (optional)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        # Ensure output directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_model(str(model_path))
        print(f"Model saved to: {model_path}")

        # Save encoders if provided
        if encoders is not None:
            encoders_path = model_path.parent / 'label_encoders.pkl'
            with open(encoders_path, 'wb') as f:
                pickle.dump(encoders, f)
            print(f"Encoders saved to: {encoders_path}")

    def load(self, model_path: Path) -> 'DelayDurationModel':
        """
        Load a trained model from file.

        Args:
            model_path: Path to the saved model file (.json)

        Returns:
            Self (loaded model)
        """
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(model_path))
        self.is_fitted = True
        print(f"Model loaded from: {model_path}")
        return self

    @staticmethod
    def load_encoders(encoders_path: Path) -> Dict[str, LabelEncoder]:
        """
        Load label encoders from file.

        Args:
            encoders_path: Path to the encoders pickle file

        Returns:
            Dictionary of label encoders
        """
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        print(f"Encoders loaded from: {encoders_path}")
        return encoders


def evaluate_model(
    model: DelayDurationModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = True
) -> Dict:
    """
    Evaluate model performance on test data.

    Args:
        model: Trained DelayDurationModel
        X_test: Test features
        y_test: Test target (delay duration in minutes)
        verbose: Whether to print evaluation results

    Returns:
        Dictionary containing all evaluation metrics
    """
    if verbose:
        print("\n" + "=" * 70)
        print("GENERATING PREDICTIONS")
        print("=" * 70)

    # Generate predictions
    y_pred = model.predict(X_test)

    if verbose:
        print("Predictions generated")

        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    median_ae = median_absolute_error(y_test, y_pred)
    explained_var = explained_variance_score(y_test, y_pred)

    # Calculate residuals
    residuals = y_test - y_pred

    # Calculate percentage metrics
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Store all metrics
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2),
        'median_absolute_error': float(median_ae),
        'explained_variance': float(explained_var),
        'mape': float(mape),
        'mean_actual': float(y_test.mean()),
        'mean_predicted': float(y_pred.mean()),
        'std_actual': float(y_test.std()),
        'std_predicted': float(y_pred.std()),
        'min_actual': float(y_test.min()),
        'max_actual': float(y_test.max()),
        'min_predicted': float(y_pred.min()),
        'max_predicted': float(y_pred.max()),
        'predictions': y_pred.tolist(),
        'residuals': residuals.tolist()
    }

    if verbose:
        # Print metrics
        print(f"\nRoot Mean Squared Error (RMSE): {rmse:.2f} minutes")
        print(f"Mean Absolute Error (MAE): {mae:.2f} minutes")
        print(f"Median Absolute Error: {median_ae:.2f} minutes")
        print(f"R² Score: {r2:.4f}")
        print(f"Explained Variance: {explained_var:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

        print("\nTarget Statistics:")
        print(f"  Actual - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")
        print(f"  Predicted - Mean: {y_pred.mean():.2f}, Std: {y_pred.std():.2f}")

        print("\nValue Ranges:")
        print(f"  Actual: [{y_test.min():.2f}, {y_test.max():.2f}]")
        print(f"  Predicted: [{y_pred.min():.2f}, {y_pred.max():.2f}]")

    return metrics


def save_metrics(metrics: Dict, output_path: Path) -> None:
    """
    Save evaluation metrics to JSON file.

    Args:
        metrics: Dictionary of evaluation metrics
        output_path: Path to save the JSON file
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Don't save predictions and residuals to JSON (too large)
    metrics_to_save = {k: v for k, v in metrics.items()
                       if k not in ['predictions', 'residuals']}

    with open(output_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)

    print(f"Metrics saved to: {output_path}")
