import pandas as pd
from Preprocess.features import build_acct_features, plot_features
from Model.detector import alert_acct_detector, plot_distribution, save_result


def main():
    PREPROCESS_DIR = "Preprocess/"
    MODEL_DIR = "Model/"
    DATA_DIR = "Data/"

    df_txn = pd.read_csv(DATA_DIR + "acct_transaction.csv")
    df_alert = pd.read_csv(DATA_DIR + "acct_alert.csv")
    df_predict = pd.read_csv(DATA_DIR + "acct_predict.csv")

    alert_list = df_alert["acct"].unique()
    predict_list = df_predict["acct"].unique()

    df_alert_features = build_acct_features(alert_list, df_txn, alert_list)
    df_predict_features = build_acct_features(predict_list, df_txn, alert_list)

    plot_features(df_alert_features, df_predict_features, PREPROCESS_DIR + "features.jpg")

    config = {
        "n_runs": 10,
        "n_estimators": 200,
        "max_depth": None,
        "class_weight": "balanced",
        "threshold": 95,
    }

    df_all_features = alert_acct_detector(config, df_alert_features, df_predict_features)
    plot_distribution(config, df_all_features, MODEL_DIR + "detector.jpg")
    save_result(config, df_all_features, predict_list, "result.csv")


if __name__ == "__main__":
    main()
