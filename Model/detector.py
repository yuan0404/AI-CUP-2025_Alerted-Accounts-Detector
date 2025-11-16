import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def alert_acct_detector(config, df_alert_features, df_predict_features):
    df_alert_copy = df_alert_features.copy()
    df_predict_copy = df_predict_features.copy()
    df_alert_copy["label"] = 1
    df_predict_copy["label"] = 0

    df_all_features = pd.concat([df_alert_copy, df_predict_copy], ignore_index=True)
    feature_cols = df_all_features.drop(columns=["acct", "label"]).columns.tolist()
    score_sum = np.zeros(len(df_all_features))

    for i in range(config["n_runs"]):
        df_sample = df_predict_copy.sample(len(df_alert_copy), random_state=i)
        df_train = pd.concat([df_alert_copy, df_sample])

        X_train = df_train[feature_cols]
        y_train = df_train["label"]

        clf = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=i,
            class_weight=config["class_weight"],
        )
        clf.fit(X_train, y_train)

        score_sum += clf.predict_proba(df_all_features[feature_cols])[:, 1]

    df_all_features["score"] = score_sum / config["n_runs"]
    return df_all_features


def plot_distribution(config, df_all_features, save_path):
    plt.figure(figsize=(8, 6))
    plt.hist(df_all_features.loc[df_all_features["label"] == 1, "score"], bins=50, alpha=0.5, color="red", label="Alert")
    plt.hist(df_all_features.loc[df_all_features["label"] == 0, "score"], bins=50, alpha=0.5, color="blue", label="Predict")

    percent = np.percentile(df_all_features.loc[df_all_features["label"] == 0, "score"], config["threshold"])
    plt.axvline(percent, color="blue", linestyle="--", label=f"{config["threshold"]}%")

    plt.title("Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_result(config, df_all_features, predict_list, save_path):
    df_result = df_all_features[df_all_features["acct"].isin(predict_list)].copy()
    df_result["label"] = (df_result["score"] >= np.percentile(df_result["score"], config["threshold"])).astype(int)
    df_result[["acct", "label"]].to_csv(save_path, index=False)
    print(f"Save to {save_path}.")
