"""
detector.py

Module for detecting alert accounts and visualizing results.

Functions
---------
alert_acct_detector(config, df_alert_features, df_predict_features)
    Train models to score accounts for alert likelihood.
plot_distribution(config, df_all_features, save_path)
    Plot score distributions for alert and predicted accounts.
save_result(config, df_all_features, predict_list, save_path)
    Save final detection results to CSV based on threshold.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def alert_acct_detector(config, df_alert_features, df_predict_features):
    """
    Detect alert accounts using ensemble RandomForest models with PU Bagging.

    Parameters
    ----------
    config : dict
        Configuration dictionary including 'n_runs', 'n_estimators', 'max_depth', 'class_weight'.
    df_alert_features : pandas.DataFrame
        Features of alert accounts.
    df_predict_features : pandas.DataFrame
        Features of accounts to predict.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing all accounts with calculated alert scores.
    """

    # Copy input DataFrames to avoid modification
    df_alert_copy = df_alert_features.copy()
    df_predict_copy = df_predict_features.copy()

    # Assign labels
    df_alert_copy["label"] = 1
    df_predict_copy["label"] = 0

    # Merge alert and predict accounts
    df_all_features = pd.concat([df_alert_copy, df_predict_copy], ignore_index=True)

    # Get list of feature columns
    feature_cols = df_all_features.drop(columns=["acct", "label"]).columns.tolist()

    # Accumulator for model scores
    score_sum = np.zeros(len(df_all_features))

    # PU Bagging loop
    for i in range(config["n_runs"]):

        # Random sample from predicted accounts
        df_sample = df_predict_copy.sample(len(df_alert_copy), random_state=i)

        # Combine positives and sampled negatives to form training data
        df_train = pd.concat([df_alert_copy, df_sample])
        X_train = df_train[feature_cols]
        y_train = df_train["label"]

        # Train RandomForest classifier
        clf = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=i,
            class_weight=config["class_weight"],
        )
        clf.fit(X_train, y_train)

        # Accumulate predicted probabilities
        score_sum += clf.predict_proba(df_all_features[feature_cols])[:, 1]

    # Average scores across runs
    df_all_features["score"] = score_sum / config["n_runs"]
    return df_all_features


def plot_distribution(config, df_all_features, save_path):
    """
    Plot the score distribution for alert and predicted accounts.

    Parameters
    ----------
    config : dict
        Configuration dictionary including 'threshold'.
    df_all_features : pandas.DataFrame
        DataFrame containing accounts with scores.
    save_path : str
        File path to save the plotted figure.
    """

    # Create figure
    plt.figure(figsize=(8, 6))

    # Plot histogram for accounts
    plt.hist(df_all_features.loc[df_all_features["label"] == 1, "score"], bins=50, alpha=0.5, color="red", label="Alert")
    plt.hist(df_all_features.loc[df_all_features["label"] == 0, "score"], bins=50, alpha=0.5, color="blue", label="Predict")

    # Add threshold line
    percent = np.percentile(df_all_features.loc[df_all_features["label"] == 0, "score"], config["threshold"])
    plt.axvline(percent, color="blue", linestyle="--", label=f"{config["threshold"]}%")

    # Set plot title, labels, and legend
    plt.title("Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.legend()

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_result(config, df_all_features, predict_list, save_path):
    """
    Save detection results of predicted accounts to CSV.

    Parameters
    ----------
    config : dict
        Configuration dictionary including 'threshold'.
    df_all_features : pandas.DataFrame
        DataFrame containing accounts with scores.
    predict_list : list
        List of accounts to predict.
    save_path : str
        CSV file path to save results.
    """

    # Filter predicted accounts
    df_result = df_all_features[df_all_features["acct"].isin(predict_list)].copy()

    # Apply threshold to determine alert labels
    df_result["label"] = (df_result["score"] >= np.percentile(df_result["score"], config["threshold"])).astype(int)

    # Save final results
    df_result[["acct", "label"]].to_csv(save_path, index=False)
    print(f"Save to {save_path}.")
