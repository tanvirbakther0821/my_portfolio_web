import argparse
import sys
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import local modules
from model.delay_duration.config import (
    DB_PATH, OUTPUT_DIR, MODEL_FILE, ENCODERS_FILE, METRICS_FILE,
    FEATURE_COLUMNS, CATEGORICAL_COLUMNS, TEST_SIZE, VERBOSE
)
from model.delay_duration.utils import (
    load_data_from_db, engineer_features, handle_missing_values,
    encode_categorical_features, prepare_features, temporal_train_test_split
)
from model.delay_duration.model import DelayDurationModel, evaluate_model, save_metrics
from model.delay_duration.visualization import create_evaluation_dashboard, display_example_predictions


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train flight delay duration prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default=str(DB_PATH),
        help='Path to SQLite database file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(OUTPUT_DIR),
        help='Directory to save model and plots'
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=TEST_SIZE,
        help='Proportion of data to use for testing (0-1)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save model, encoders, or metrics'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Do not create or save plots'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Convert paths to Path objects
    db_path = Path(args.db_path)
    output_dir = Path(args.output_dir)

    # Set verbosity
    verbose = not args.quiet

    # Start timer
    start_time = time.time()

    if verbose:
        print("=" * 70)
        print(" FLIGHT DELAY DURATION PREDICTION MODEL")
        print("=" * 70)
        print(f"\nDatabase: {db_path}")
        print(f"Output directory: {output_dir}")
        print(f"Test size: {args.test_size * 100:.0f}%")
        print()

    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    try:
        df = load_data_from_db(str(db_path), verbose=verbose)
    except Exception as e:
        print(f"\nError loading data: {e}")
        print(f"Please ensure the database exists at: {db_path}")
        sys.exit(1)

    # ========================================================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================================================
    df = engineer_features(df, verbose=verbose)

    # ========================================================================
    # STEP 3: HANDLE MISSING VALUES
    # ========================================================================
    df = handle_missing_values(df, verbose=verbose)

    # ========================================================================
    # STEP 4: ENCODE CATEGORICAL FEATURES
    # ========================================================================
    df, label_encoders = encode_categorical_features(
        df,
        categorical_cols=CATEGORICAL_COLUMNS,
        verbose=verbose
    )

    # ========================================================================
    # STEP 5: PREPARE FEATURES
    # ========================================================================
    X, y = prepare_features(df, feature_cols=FEATURE_COLUMNS, verbose=verbose)

    # ========================================================================
    # STEP 6: TRAIN/TEST SPLIT
    # ========================================================================
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        X, y, df, test_size=args.test_size, verbose=verbose
    )

    # ========================================================================
    # STEP 7: TRAIN MODEL
    # ========================================================================
    model = DelayDurationModel()
    model.fit(X_train, y_train, X_test, y_test, verbose=verbose)

    # ========================================================================
    # STEP 8: EVALUATE MODEL
    # ========================================================================
    metrics = evaluate_model(model, X_test, y_test, verbose=verbose)

    # ========================================================================
    # STEP 9: FEATURE IMPORTANCE
    # ========================================================================
    feature_importance = model.get_feature_importance()

    if verbose:
        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE")
        print("=" * 70)
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))

    # ========================================================================
    # STEP 10: SAVE MODEL AND RESULTS
    # ========================================================================
    if not args.no_save:
        if verbose:
            print("\n" + "=" * 70)
            print("SAVING MODEL AND RESULTS")
            print("=" * 70)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model and encoders
        model.save(MODEL_FILE, encoders=label_encoders)

        # Save metrics
        save_metrics(metrics, METRICS_FILE)

        if verbose:
            print(f"\nAll outputs saved to: {output_dir}")

    # ========================================================================
    # STEP 11: CREATE VISUALIZATIONS
    # ========================================================================
    if not args.no_plots:
        create_evaluation_dashboard(feature_importance, metrics, y_test)

        # Display example predictions
        if verbose:
            display_example_predictions(model, X_test, y_test)

    # ========================================================================
    # COMPLETION
    # ========================================================================
    elapsed_time = time.time() - start_time

    if verbose:
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"\nTotal execution time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print("\nModel Summary:")
        print(f"  RMSE:     {metrics['rmse']:.2f} minutes")
        print(f"  MAE:      {metrics['mae']:.2f} minutes")
        print(f"  R² Score: {metrics['r2_score']:.4f}")
        print(f"  MAPE:     {metrics['mape']:.2f}%")

        if not args.no_save:
            print(f"\nOutputs saved to: {output_dir}")
            print("  - Model: delay_duration_model.json")
            print("  - Encoders: label_encoders.pkl")
            print("  - Metrics: metrics.json")

        if not args.no_plots:
            print("  - Plots: *.png files")

        print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
