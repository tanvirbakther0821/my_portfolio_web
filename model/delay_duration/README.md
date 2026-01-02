# Flight Delay Duration Prediction Model

This model predicts the **duration** (in minutes) of flight delays for flights that are delayed by 15 minutes or more.

## Overview

- **Model Type**: XGBoost Regression
- **Target**: `arr_delay_minutes` (Arrival delay in minutes)
- **Data Filter**: Only flights where `arr_del15 = 1` (delayed ≥ 15 minutes)
- **Prediction**: Continuous value representing delay duration in minutes

## Relationship to Delay Probability Model

This model complements the delay probability model:

1. **Delay Probability Model** (`/model/delay_probability/`)
   - Predicts: Whether a flight will be delayed ≥ 15 minutes (binary: yes/no)
   - Target: `arr_del15` (0 or 1)
   - Model: XGBoost Classifier

2. **Delay Duration Model** (`/model/delay_duration/`) ← This model
   - Predicts: How long the delay will be (regression: minutes)
   - Target: `arr_delay_minutes` (continuous)
   - Model: XGBoost Regressor
   - Only trained on delayed flights (arr_del15 = 1)

## Directory Structure

```
delay_duration/
├── config.py           # Configuration and hyperparameters
├── model.py            # Model class and evaluation
├── utils.py            # Data loading and preprocessing
├── visualization.py    # Plotting and visualization
├── main.py             # Main training pipeline
├── README.md           # This file
└── output/             # Saved models and plots
    ├── delay_duration_model.json
    ├── label_encoders.pkl
    ├── metrics.json
    └── *.png
```

## Quick Start

### Training the Model

Run the full training pipeline:

```bash
cd model/delay_duration
python main.py
```

With custom options:

```bash
python main.py --test-size 0.3 --no-plots --quiet
```

### Command Line Arguments

- `--db-path`: Path to SQLite database (default: flights.db)
- `--output-dir`: Directory for outputs (default: output/)
- `--test-size`: Test set proportion (default: 0.2)
- `--no-save`: Skip saving model and metrics
- `--no-plots`: Skip creating visualizations
- `--quiet`: Reduce output verbosity

## Features

The model uses the following features:

**Temporal Features:**
- Month, Quarter, DayofMonth, DayOfWeek

**Route Features:**
- Reporting_Airline (encoded)
- Origin airport (encoded)
- Dest airport (encoded)
- Distance

**Time Features:**
- CRSElapsedTime (scheduled flight duration)
- dep_hour (departure hour 0-23)
- arr_hour (arrival hour 0-23)
- dep_time_category (time period 1-5)

## Model Performance Metrics

The model is evaluated using regression metrics:

- **RMSE** (Root Mean Squared Error): Average prediction error in minutes
- **MAE** (Mean Absolute Error): Average absolute error in minutes
- **R² Score**: Proportion of variance explained (0-1)
- **MAPE** (Mean Absolute Percentage Error): Average percentage error
- **Median Absolute Error**: Median of absolute errors

## Visualizations

The pipeline generates the following plots:

1. **Feature Importance**: Top features driving predictions
2. **Predictions Scatter**: Actual vs predicted delay duration
3. **Residuals Plot**: Residuals vs predicted values
4. **Residuals Distribution**: Histogram of prediction errors

## Usage Example

```python
from model.delay_duration import DelayDurationModel
from model.delay_duration.utils import load_data_from_db

# Load data (only delayed flights)
df = load_data_from_db('flights.db')

# Train model
model = DelayDurationModel()
model.fit(X_train, y_train)

# Predict delay duration
predictions = model.predict(X_test)  # Returns delay in minutes
```

## Data Filtering

**Important**: This model only trains on flights where `arr_del15 = 1`, meaning:
- Flights with arrival delay ≥ 15 minutes
- Excludes on-time and slightly delayed flights (< 15 min)
- Target values range from ~15 minutes to several hours

## Model Configuration

Key hyperparameters (in `config.py`):

```python
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': ['rmse', 'mae'],
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
```

## Output Files

After training, the following files are saved to `output/`:

- `delay_duration_model.json`: Trained XGBoost model
- `label_encoders.pkl`: Categorical feature encoders
- `metrics.json`: Evaluation metrics
- `model_evaluation.png`: Combined dashboard
- `feature_importance.png`: Feature importance plot
- `predictions_scatter.png`: Actual vs predicted scatter
- `residuals_plot.png`: Residuals analysis

## Next Steps

- Tune hyperparameters for better performance
- Add more features (weather, airline delays, etc.)
- Experiment with different regression models
- Implement prediction intervals/confidence bounds
- Deploy model for real-time predictions
