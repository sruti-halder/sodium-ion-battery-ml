# MAIN PIPELINE

from src.data_preprocessing import clean_dataset
from src.data_analysis import show_statistics
from src.visualization import plot_distributions

from src.feature_engineering import (
    generate_magpie_features,
    generate_domain_features,
    generate_structural_features
)

from src.data_merging import create_hybrid_dataset

from src.model_training import (
    run_random_forest,
    run_gradient_boosting,
    run_xgboost
)

from src.external_validation import (
    validate_mukherjee,
    validate_joshi
)

from src.screening import (
    generate_candidates,
    generate_features,
    run_screening,
    classify_materials
)

import pandas as pd


def main():

    print("STARTING PROJECT")

    # DATA CLEANING
    
    print("Cleaning dataset")

    raw_data = "data/na_battery_dataset.csv"
    clean_data = "data/na_battery_cleaned_dataset.csv"

    clean_dataset(raw_data, clean_data)


    # BASIC ANALYSIS + PLOTS
  
    print("Data analysis")

    show_statistics(clean_data)
    plot_distributions(clean_data)


    # FEATURE ENGINEERING

    print("Feature engineering")

    generate_magpie_features(
        clean_data,
        "data/processed/magpie.csv"
    )

    generate_domain_features(
        clean_data,
        "data/processed/domain.csv"
    )

    generate_structural_features(
        clean_data,
        "data/processed/structural.csv",
        api_key="YOUR_API_KEY"   
    )


    # LOAD FEATURE DATA
    
    print("Loading feature datasets")

    df_magpie = pd.read_csv("data/processed/magpie.csv")
    df_domain = pd.read_csv("data/processed/domain.csv")
    df_struct = pd.read_csv("data/processed/structural.csv")


    # CREATE HYBRID DATASET
    
    print("Creating hybrid dataset")

    X, y_voltage, y_capacity, y_volume = create_hybrid_dataset(
        df_magpie,
        df_domain,
        df_struct
    )


    # MODEL TRAINING
    
    print("Training models")

    print("\n--- RANDOM FOREST ---")
    rf_v = run_random_forest(X, y_voltage, "Voltage")
    rf_c = run_random_forest(X, y_capacity, "Capacity")

    print("\n--- GRADIENT BOOSTING ---")
    gb_v = run_gradient_boosting(X, y_voltage, "Voltage")
    gb_c = run_gradient_boosting(X, y_capacity, "Capacity")

    print("\n--- XGBOOST ---")
    xgb_v = run_xgboost(X, y_voltage, "Voltage")


    # EXTERNAL VALIDATION
    
    print("External validation")

    validate_mukherjee(
        gb_c,
        X.columns,
        "data/raw/mukherjee.csv"
    )

    validate_joshi(
        gb_v,
        xgb_v,
        X.columns,
        "data/raw/joshi.csv"
    )


    # SCREENING
   
    print("Candidate screening")

    candidates = generate_candidates()

    candidates, X_candidate = generate_features(
        candidates,
        X.columns
    )

    candidates = run_screening(
        candidates,
        X_candidate,
        gb_v,
        gb_c
    )

    classify_materials(candidates)


    print("PROJECT COMPLETE")


# RUN

if __name__ == "__main__":
    main()