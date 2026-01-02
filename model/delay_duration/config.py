from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Get the directory where this config file is located
BASE_DIR = Path(__file__).parent

# Database path (relative to project root)
DB_PATH = BASE_DIR.parent.parent / 'flights.db'

# Output directory for saving models and plots
OUTPUT_DIR = BASE_DIR / 'output'

# Model output paths
MODEL_FILE = OUTPUT_DIR / 'delay_duration_model.json'
ENCODERS_FILE = OUTPUT_DIR / 'label_encoders.pkl'
METRICS_FILE = OUTPUT_DIR / 'metrics.json'

# Plot output paths
PLOT_ALL = OUTPUT_DIR / 'model_evaluation.png'
PLOT_FEATURE_IMPORTANCE = OUTPUT_DIR / 'feature_importance.png'
PLOT_RESIDUALS = OUTPUT_DIR / 'residuals_plot.png'
PLOT_PREDICTIONS = OUTPUT_DIR / 'predictions_scatter.png'

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Feature columns to use in the model
FEATURE_COLUMNS = [
    'Month',
    'Quarter',
    'DayofMonth',
    'DayOfWeek',
    'Reporting_Airline_encoded',
    'Origin_encoded',
    'Dest_encoded',
    'Distance',
    'CRSElapsedTime',
    'dep_hour',
    'arr_hour',
    'dep_time_category'
]

# Categorical columns to encode
CATEGORICAL_COLUMNS = ['Reporting_Airline', 'Origin', 'Dest']

# Train/test split ratio
TEST_SIZE = 0.2

# Random seed for reproducibility
RANDOM_STATE = 42

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# XGBoost hyperparameters for regression
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': ['rmse', 'mae'],
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'tree_method': 'hist',  # Faster for large datasets
    'early_stopping_rounds': 20,
}

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Figure size for combined plot
FIGURE_SIZE = (15, 10)

# Top N features to show in feature importance plot
TOP_N_FEATURES = 10

# Plot style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# Color palette
SCATTER_ALPHA = 0.3
RESIDUAL_BINS = 50

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Verbosity level
VERBOSE = True

# XGBoost training verbosity (print every N rounds)
XGBOOST_VERBOSE = 50
