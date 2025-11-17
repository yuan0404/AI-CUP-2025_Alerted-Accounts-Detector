"""
Main pipeline for detecting alert accounts from transaction data.

Functions
---------
main()
    Executes the full pipeline including feature extraction, alert detection,
    visualization, and result saving.

Directory Structure
-------------------
Data/
    acct_transaction.csv
    acct_alert.csv
    acct_predict.csv
Preprocess/
    features.py
Model/
    detector.py
"""

import pandas as pd
from Preprocess.features import build_acct_features, plot_features
from Model.detector import alert_acct_detector, plot_distribution, save_result


def main():
    """
    Execute the main pipeline.

    Steps
    -----
    1. Load transaction, alert, and predict account data.
    2. Build features for alert and predicted accounts.
    3. Visualize feature distributions.
    4. Configure and run alert detection.
    5. Plot score distribution and save final results.
    """
    # Define directories
    PREPROCESS_DIR = "Preprocess/"
    MODEL_DIR = "Model/"
    DATA_DIR = "Data/"

    # Load transaction, alert, and predict account data
    df_txn = pd.read_csv(DATA_DIR + "acct_transaction.csv")
    df_alert = pd.read_csv(DATA_DIR + "acct_alert.csv")
    df_predict = pd.read_csv(DATA_DIR + "acct_predict.csv")

    # Extract unique account IDs
    alert_list = df_alert["acct"].unique()
    predict_list = df_predict["acct"].unique()

    # Build features for alert and predict accounts
    df_alert_features = build_acct_features(alert_list, df_txn, alert_list)
    df_predict_features = build_acct_features(predict_list, df_txn, alert_list)

    # Visualize features comparison
    plot_features(df_alert_features, df_predict_features, PREPROCESS_DIR + "features.jpg")

    # Configuration for alert detection
    config = {
        "n_runs": 10,
        "n_estimators": 200,
        "max_depth": None,
        "class_weight": "balanced",
        "threshold": 95,
    }

    # Detect alert accounts
    df_all_features = alert_acct_detector(config, df_alert_features, df_predict_features)

    # Visualize score distribution
    plot_distribution(config, df_all_features, MODEL_DIR + "detector.jpg")

    # Save detection results
    save_result(config, df_all_features, predict_list, "result.csv")


if __name__ == "__main__":
    main()
