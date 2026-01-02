"""
Visualization module for Flight Delay Duration Model.
Creates evaluation plots and dashboards.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from pathlib import Path
from typing import Dict

from model.delay_duration.config import (
    OUTPUT_DIR, FIGURE_SIZE, TOP_N_FEATURES,
    PLOT_STYLE, SCATTER_ALPHA, RESIDUAL_BINS,
    PLOT_ALL, PLOT_FEATURE_IMPORTANCE, PLOT_RESIDUALS, PLOT_PREDICTIONS
)


def create_evaluation_dashboard(
    feature_importance: pd.DataFrame,
    metrics: Dict,
    y_test: pd.Series,
    save_path: Path = None
) -> None:
    """
    Create a comprehensive evaluation dashboard with multiple plots.
    
    Args:
        feature_importance: DataFrame with feature importance scores
        metrics: Dictionary containing evaluation metrics
        y_test: Actual test values
        save_path: Path to save the combined figure
    """
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    # Set style
    try:
        plt.style.use(PLOT_STYLE)
    except:
        plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    
    # Get predictions and residuals from metrics
    y_pred = np.array(metrics.get('predictions', []))
    residuals = np.array(metrics.get('residuals', []))
    
    # 1. Feature Importance (top left)
    ax1 = axes[0, 0]
    top_features = feature_importance.head(TOP_N_FEATURES)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
    bars = ax1.barh(range(len(top_features)), top_features['importance'].values, color=colors)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'].values)
    ax1.invert_yaxis()
    ax1.set_xlabel('Importance Score')
    ax1.set_title(f'Top {TOP_N_FEATURES} Feature Importance', fontsize=12, fontweight='bold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'].values)):
        ax1.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=8)
    
    # 2. Predictions vs Actual (top right)
    ax2 = axes[0, 1]
    if len(y_pred) > 0 and len(y_test) > 0:
        # Sample for faster plotting
        sample_size = min(5000, len(y_pred))
        idx = np.random.choice(len(y_pred), sample_size, replace=False)
        
        ax2.scatter(y_test.values[idx], y_pred[idx], alpha=SCATTER_ALPHA, s=10, c='steelblue')
        
        # Perfect prediction line
        max_val = max(y_test.max(), y_pred.max())
        ax2.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax2.set_xlabel('Actual Delay (minutes)')
        ax2.set_ylabel('Predicted Delay (minutes)')
        ax2.set_title('Predicted vs Actual Delay Duration', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left')
        
        # Add R² annotation
        ax2.text(0.95, 0.05, f"R² = {metrics.get('r2_score', 0):.4f}",
                transform=ax2.transAxes, ha='right', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Residuals Plot (bottom left)
    ax3 = axes[1, 0]
    if len(y_pred) > 0 and len(residuals) > 0:
        sample_size = min(5000, len(y_pred))
        idx = np.random.choice(len(y_pred), sample_size, replace=False)
        
        ax3.scatter(y_pred[idx], residuals[idx], alpha=SCATTER_ALPHA, s=10, c='coral')
        ax3.axhline(y=0, color='black', linestyle='-', lw=1)
        ax3.set_xlabel('Predicted Delay (minutes)')
        ax3.set_ylabel('Residual (Actual - Predicted)')
        ax3.set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
    
    # 4. Residuals Distribution (bottom right)
    ax4 = axes[1, 1]
    if len(residuals) > 0:
        ax4.hist(residuals, bins=RESIDUAL_BINS, color='teal', edgecolor='white', alpha=0.7)
        ax4.axvline(x=0, color='red', linestyle='--', lw=2)
        ax4.axvline(x=np.mean(residuals), color='orange', linestyle='--', lw=2, label=f'Mean: {np.mean(residuals):.2f}')
        ax4.set_xlabel('Residual (minutes)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
        ax4.legend()
    
    # Add metrics summary
    metrics_text = (
        f"RMSE: {metrics.get('rmse', 0):.2f} min | "
        f"MAE: {metrics.get('mae', 0):.2f} min | "
        f"R²: {metrics.get('r2_score', 0):.4f} | "
        f"MAPE: {metrics.get('mape', 0):.2f}%"
    )
    fig.suptitle('Flight Delay Duration Model - Evaluation Dashboard\n' + metrics_text, 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = PLOT_ALL
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Dashboard saved to: {save_path}")
    
    # Save individual plots
    save_individual_plots(feature_importance, metrics, y_test, y_pred, residuals)
    
    plt.close(fig)


def save_individual_plots(
    feature_importance: pd.DataFrame,
    metrics: Dict,
    y_test: pd.Series,
    y_pred: np.ndarray,
    residuals: np.ndarray
) -> None:
    """Save individual plots for each visualization."""
    
    # Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = feature_importance.head(TOP_N_FEATURES)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
    ax.barh(range(len(top_features)), top_features['importance'].values, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance - Top 10 Features')
    plt.tight_layout()
    fig.savefig(PLOT_FEATURE_IMPORTANCE, dpi=150, bbox_inches='tight')
    print(f"Feature importance plot saved to: {PLOT_FEATURE_IMPORTANCE}")
    plt.close(fig)
    
    # Predictions scatter
    if len(y_pred) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        sample_size = min(5000, len(y_pred))
        idx = np.random.choice(len(y_pred), sample_size, replace=False)
        ax.scatter(y_test.values[idx], y_pred[idx], alpha=0.3, s=10)
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2)
        ax.set_xlabel('Actual Delay (minutes)')
        ax.set_ylabel('Predicted Delay (minutes)')
        ax.set_title(f'Predicted vs Actual (R² = {metrics.get("r2_score", 0):.4f})')
        plt.tight_layout()
        fig.savefig(PLOT_PREDICTIONS, dpi=150, bbox_inches='tight')
        print(f"Predictions scatter saved to: {PLOT_PREDICTIONS}")
        plt.close(fig)
    
    # Residuals plot
    if len(residuals) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(residuals, bins=50, edgecolor='white', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', lw=2)
        ax.set_xlabel('Residual (minutes)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Prediction Residuals')
        plt.tight_layout()
        fig.savefig(PLOT_RESIDUALS, dpi=150, bbox_inches='tight')
        print(f"Residuals plot saved to: {PLOT_RESIDUALS}")
        plt.close(fig)


def display_example_predictions(model, X_test: pd.DataFrame, y_test: pd.Series, n_examples: int = 5) -> None:
    """
    Display example predictions for visual inspection.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        n_examples: Number of examples to display
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)
    
    # Get random sample
    idx = np.random.choice(len(X_test), min(n_examples, len(X_test)), replace=False)
    
    X_sample = X_test.iloc[idx]
    y_actual = y_test.iloc[idx]
    y_pred = model.predict(X_sample)
    
    print(f"\n{'Actual':<12} {'Predicted':<12} {'Error':<12} {'% Error':<12}")
    print("-" * 50)
    
    for actual, pred in zip(y_actual, y_pred):
        error = actual - pred
        pct_error = abs(error / actual) * 100 if actual != 0 else 0
        print(f"{actual:<12.1f} {pred:<12.1f} {error:<12.1f} {pct_error:<12.1f}%")
