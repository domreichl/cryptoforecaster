import seaborn as sns
import matplotlib.pyplot as plt

from utils.file_handling import ResultsHandler


def plot_tuning_results():
    df = ResultsHandler().load_csv_results("tuning")

    fig, axs = plt.subplots(3, 4, sharey=True)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    fig.suptitle("Hypertuning Results on Various Performance Metrics")

    for i, metric in enumerate(["Accuracy", "Precision", "F1", "PredictiveScore"]):
        axs[0, i].set_title(metric)
        axs[0, i].axhline(y=0.5, color="black", linestyle="-", linewidth=3)
        sns.boxplot(
            x="look_back_window", y=metric, hue="forecast_window", data=df, ax=axs[0, i]
        ).set_ylabel(None)
        axs[1, i].axhline(y=0.5, color="black", linestyle="-", linewidth=3)
        sns.boxplot(
            x="layers", y=metric, hue="units", data=df, ax=axs[1, i]
        ).set_ylabel(None)
        axs[2, i].axhline(y=0.5, color="black", linestyle="-", linewidth=3)
        sns.boxplot(x="dropout_rate", y=metric, data=df, ax=axs[2, i]).set_ylabel(None)

    plt.tight_layout()
    plt.show()


def plot_test_results():
    df = ResultsHandler().load_csv_results("test")
    df = df.iloc[-1]

    metrics = df[["Precision", "Recall", "Accuracy", "F1", "PredictiveScore"]]
    metrics = metrics.transpose().reset_index()
    metrics.columns = ["Metric", "Score"]
    print(f"Forecast Bias: {df['ForecastBias']*100}%")

    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(15)

    sns.barplot(x="Metric", y="Score", data=metrics, ax=ax0)
    ax0.axhline(y=0.5, color="black", linestyle="-", linewidth=3)
    ax0.set_title("Test Set Performance")

    sns.heatmap([[df.TN, df.FP], [df.FN, df.TP]], cmap="Blues", annot=True, ax=ax1)
    ax1.set_title("Confusion Matrix for Sign Prediction")
    ax1.set_xlabel("Prediction")
    ax1.set_ylabel("Ground Truth")

    plt.tight_layout()
    plt.show()
