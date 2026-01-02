import pandas as pd
import numpy as np
import sqlite3
from typing import Tuple, Dict, List
from sklearn.preprocessing import LabelEncoder


def parse_time_to_hour(time_val) -> float:
    """
    Convert TIME or HHMM format to hour (0-23).

    Args:
        time_val: Time value in HHMM integer format or HH:MM:SS string format

    Returns:
        Hour as integer (0-23), or np.nan if input is NaN

    Examples:
        >>> parse_time_to_hour(1530)
        15
        >>> parse_time_to_hour("15:30:00")
        15
    """
    if pd.isna(time_val):
        return np.nan

    # Handle if it's already a time string (HH:MM:SS)
    if isinstance(time_val, str) and ':' in time_val:
        return int(time_val.split(':')[0])

    # Handle HHMM integer format
    time_str = str(int(time_val)).zfill(4)
    hour = int(time_str[:2])
    return hour


def get_time_category(hour: float) -> float:
    """
    Categorize hour into time of day period.

    Args:
        hour: Hour of day (0-23)

    Returns:
        Time category:
            1 - Early morning (5-9)
            2 - Morning (9-12)
            3 - Afternoon (12-17)
            4 - Evening (17-21)
            5 - Night (other hours)
        Returns np.nan if input is NaN

    Examples:
        >>> get_time_category(8)
        1
        >>> get_time_category(15)
        3
    """
    if pd.isna(hour):
        return np.nan
    if 5 <= hour < 9:
        return 1  # Early morning
    elif 9 <= hour < 12:
        return 2  # Morning
    elif 12 <= hour < 17:
        return 3  # Afternoon
    elif 17 <= hour < 21:
        return 4  # Evening
    else:
        return 5  # Night


def load_data_from_db(db_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load flight data from SQLite database with filters for DELAYED flights only.

    Args:
        db_path: Path to SQLite database file
        verbose: Whether to print progress information

    Returns:
        DataFrame with flight data

    Filters applied:
        - Year > 2001
        - Cancelled = 0
        - Diverted = 0
        - arr_del15 = 1 (only flights delayed >= 15 minutes)
    """
    if verbose:
        print("=" * 70)
        print("LOADING DATA FROM DATABASE")
        print("=" * 70)
        print(f"Database: {db_path}")
        print("Filters: Year > 2001, Cancelled = 0, Diverted = 0, arr_del15 = 1")

    # Connect to database
    conn = sqlite3.connect(db_path)

    # Query with filters - ONLY delayed flights (arr_del15 = 1)
    query = """
    SELECT
        year AS Year,
        month AS Month,
        quarter AS Quarter,
        day_of_month AS DayofMonth,
        day_of_week AS DayOfWeek,
        airline_id AS Reporting_Airline,
        origin_airport_id AS Origin,
        dest_airport_id AS Dest,
        distance AS Distance,
        crs_dep_time AS CRSDepTime,
        crs_arr_time AS CRSArrTime,
        crs_elapsed_time AS CRSElapsedTime,
        arr_del15 AS ArrDel15,
        arr_delay_minutes AS ArrDelayMinutes
    FROM flights
    WHERE year > 2001
        AND cancelled = 0
        AND diverted = 0
        AND arr_del15 = 1
    ORDER BY year, month, day_of_month;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if verbose:
        print(f"\nData loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Target statistics (ArrDelayMinutes):")
        print(f"  Mean: {df['ArrDelayMinutes'].mean():.2f} minutes")
        print(f"  Median: {df['ArrDelayMinutes'].median():.2f} minutes")
        print(f"  Std: {df['ArrDelayMinutes'].std():.2f} minutes")
        print(f"  Min: {df['ArrDelayMinutes'].min():.2f} minutes")
        print(f"  Max: {df['ArrDelayMinutes'].max():.2f} minutes")

    return df


def engineer_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Engineer time-based features from raw data.

    Creates:
        - dep_hour: Departure hour (0-23)
        - arr_hour: Arrival hour (0-23)
        - dep_time_category: Departure time period (1-5)

    Args:
        df: Input DataFrame with CRSDepTime and CRSArrTime columns
        verbose: Whether to print progress information

    Returns:
        DataFrame with additional engineered features
    """
    if verbose:
        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING")
        print("=" * 70)

    df = df.copy()

    # Create hour features
    df['dep_hour'] = df['CRSDepTime'].apply(parse_time_to_hour)
    df['arr_hour'] = df['CRSArrTime'].apply(parse_time_to_hour)
    df['dep_time_category'] = df['dep_hour'].apply(get_time_category)

    if verbose:
        print("Created features:")
        print("  - dep_hour: Departure hour (0-23)")
        print("  - arr_hour: Arrival hour (0-23)")
        print("  - dep_time_category: Time period category (1-5)")

    return df


def handle_missing_values(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Strategy:
        - Drop rows with missing target (ArrDelayMinutes)
        - Fill missing hour features with mode
        - Fill missing numeric features (Distance, CRSElapsedTime) with median

    Args:
        df: Input DataFrame
        verbose: Whether to print progress information

    Returns:
        DataFrame with missing values handled
    """
    if verbose:
        print("\n" + "=" * 70)
        print("HANDLING MISSING VALUES")
        print("=" * 70)

    df = df.copy()
    initial_rows = len(df)

    # Drop rows with missing target
    df = df.dropna(subset=['ArrDelayMinutes'])
    if verbose and initial_rows - len(df) > 0:
        print(f"Dropped {initial_rows - len(df):,} rows with missing target")

    # Fill missing hour features with mode
    for col in ['dep_hour', 'arr_hour', 'dep_time_category']:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            missing_count = df[col].isnull().sum()
            df[col].fillna(mode_val, inplace=True)
            if verbose:
                print(f"Filled {missing_count:,} {col} missing values with mode: {mode_val}")

    # Fill missing numeric features with median
    for col in ['CRSElapsedTime', 'Distance']:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            missing_count = df[col].isnull().sum()
            df[col].fillna(median_val, inplace=True)
            if verbose:
                print(f"Filled {missing_count:,} {col} missing values with median: {median_val:.0f}")

    if verbose:
        print(f"\nFinal dataset: {df.shape[0]:,} rows")

    return df


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_cols: List[str] = None,
    encoders: Dict[str, LabelEncoder] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical features using LabelEncoder.

    Args:
        df: Input DataFrame
        categorical_cols: List of columns to encode.
                         Defaults to ['Reporting_Airline', 'Origin', 'Dest']
        encoders: Pre-fitted encoders (for inference). If None, fit new encoders.
        verbose: Whether to print progress information

    Returns:
        Tuple of (DataFrame with encoded features, dictionary of encoders)
    """
    if categorical_cols is None:
        categorical_cols = ['Reporting_Airline', 'Origin', 'Dest']

    if verbose:
        print("\n" + "=" * 70)
        print("LABEL ENCODING")
        print("=" * 70)

    df = df.copy()

    if encoders is None:
        # Fit new encoders (training mode)
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            if verbose:
                print(f"Encoded {col}: {len(le.classes_):,} unique values")
    else:
        # Use existing encoders (inference mode)
        for col in categorical_cols:
            le = encoders[col]
            df[f'{col}_encoded'] = le.transform(df[col].astype(str))
            if verbose:
                print(f"Encoded {col} using existing encoder")

    return df, encoders


def prepare_features(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target for modeling.

    Args:
        df: Input DataFrame with all features
        feature_cols: List of feature column names. If None, uses default set.
        verbose: Whether to print progress information

    Returns:
        Tuple of (X features DataFrame, y target Series)
    """
    if feature_cols is None:
        feature_cols = [
            'Month', 'Quarter', 'DayofMonth', 'DayOfWeek',
            'Reporting_Airline_encoded', 'Origin_encoded', 'Dest_encoded',
            'Distance', 'CRSElapsedTime',
            'dep_hour', 'arr_hour', 'dep_time_category'
        ]

    if verbose:
        print("\n" + "=" * 70)
        print("FEATURES FOR MODELING")
        print("=" * 70)
        print(f"Feature columns ({len(feature_cols)}):")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i:2d}. {col}")

    X = df[feature_cols].copy()
    y = df['ArrDelayMinutes'].copy()

    if verbose:
        print(f"\nX shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Target (ArrDelayMinutes) range: [{y.min():.2f}, {y.max():.2f}]")

    return X, y


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    test_size: float = 0.2,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform temporal train/test split.

    Uses chronological split (data must be pre-sorted by date) to prevent
    data leakage - earlier data for training, later data for testing.

    Args:
        X: Features DataFrame
        y: Target Series
        df: Original DataFrame (for getting year information)
        test_size: Proportion of data to use for testing (default: 0.2)
        verbose: Whether to print progress information

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if verbose:
        print("\n" + "=" * 70)
        print(f"TEMPORAL TRAIN/TEST SPLIT ({int((1-test_size)*100)}-{int(test_size*100)})")
        print("=" * 70)

    # Data should already be sorted by Year, Month, Day
    split_idx = int(len(X) * (1 - test_size))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if verbose:
        train_years = df['Year'].iloc[:split_idx]
        test_years = df['Year'].iloc[split_idx:]

        print(f"Train set: {X_train.shape[0]:,} samples (mean delay: {y_train.mean():.2f} min)")
        print(f"Test set:  {X_test.shape[0]:,} samples (mean delay: {y_test.mean():.2f} min)")
        print(f"Train years: {train_years.min()} - {train_years.max()}")
        print(f"Test years:  {test_years.min()} - {test_years.max()}")

    return X_train, X_test, y_train, y_test


def get_feature_columns() -> List[str]:
    """
    Get the default list of feature columns used in the model.

    Returns:
        List of feature column names
    """
    return [
        'Month', 'Quarter', 'DayofMonth', 'DayOfWeek',
        'Reporting_Airline_encoded', 'Origin_encoded', 'Dest_encoded',
        'Distance', 'CRSElapsedTime',
        'dep_hour', 'arr_hour', 'dep_time_category'
    ]
