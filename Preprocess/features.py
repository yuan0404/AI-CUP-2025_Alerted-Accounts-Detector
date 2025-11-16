import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def build_acct_features(acct_list, df_txn, alert_list):
    df_filtered = df_txn.query("from_acct in @acct_list or to_acct in @acct_list").copy()
    df_filtered = df_filtered.sort_values(by=["txn_date", "txn_time"])

    acct_txn_dict = {
        acct: df_filtered.query("from_acct == @acct or to_acct == @acct").copy()
        for acct in acct_list
    }

    features_dict = {}

    for acct, df_acct in acct_txn_dict.items():
        n = len(df_acct)
        send = df_acct[df_acct["from_acct"] == acct]
        recv = df_acct[df_acct["to_acct"] == acct]

        f = features_dict[acct] = {}

        f["net_txn_ratio"] = ((len(recv) - len(send)) / n + 1) / 2
        f["crossbank_ratio"] = ((recv["from_acct_type"].eq(2).sum() + send["to_acct_type"].eq(2).sum()) / n)
        f["counterparty_diversity"] = pd.concat([recv["from_acct"], send["to_acct"]]).nunique() / n

        valid_self = df_acct["is_self_txn"].isin(["Y", "N"])
        f["self_txn_ratio"] = (df_acct["is_self_txn"].eq("Y").sum() / valid_self.sum() if valid_self.sum() else 0.5)
        f["txn_with_alert"] = (recv["from_acct"].isin(alert_list).sum() + send["to_acct"].isin(alert_list).sum()) / n

        f["net_amt_ratio"] = ((recv["txn_amt"].sum() - send["txn_amt"].sum()) / df_acct["txn_amt"].sum() + 1) / 2
        f["top_amt_ratio"] = df_acct.groupby("txn_amt").size().nlargest(3).sum() / n

        df_acct['txn_hour'] = pd.to_datetime(df_acct['txn_time'], format='%H:%M:%S').dt.hour
        f["top_hour_ratio"] = df_acct.groupby(["txn_date", "txn_hour"]).size().nlargest(3).sum() / n
        f["top_date_ratio"] = df_acct.groupby("txn_date").size().nlargest(2).sum() / n

        f["channel_diversity"] = df_acct.loc[df_acct["channel_type"] != "UNK", "channel_type"].nunique() / 5
        f["unknown_channel_ratio"] = df_acct["channel_type"].eq("UNK").sum() / n

    df_features = (pd.DataFrame.from_dict(features_dict, orient="index").assign(acct=acct_list).reset_index(drop=True))
    df_features = df_features[["acct"] + [c for c in df_features.columns if c != "acct"]]
    df_features.iloc[:, 1:] = df_features.iloc[:, 1:].round(4)
    return df_features


def plot_features(df_alert_features, df_predict_features, save_path):
    feature_cols = df_alert_features.drop(columns=["acct"]).columns.tolist()
    n_cols = 4
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols
    figsize = (n_cols * 4, n_rows * 3)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)

    for i, col in enumerate(feature_cols):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df_alert_features[col], color="red", label="Alert", kde=False, bins=30, stat="percent", alpha=0.4, ax=ax)
        sns.histplot(df_predict_features[col], color="blue", label="Predict", kde=False, bins=30, stat="percent", alpha=0.4, ax=ax)

        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Percent")

        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
