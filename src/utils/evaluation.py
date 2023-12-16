import numpy as np


def compute_metrics(y_true: np.array, y_pred: np.array) -> dict:
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    assert y_true.shape == y_pred.shape
    P = np.count_nonzero(y_true)
    N = np.count_nonzero(y_true == 0)
    TP = np.count_nonzero(y_pred[np.where(y_true == 1)])
    FN = np.count_nonzero(y_pred[np.where(y_true == 1)] == 0)
    TN = np.count_nonzero(y_pred[np.where(y_true == 0)] == 0)
    FP = np.count_nonzero(y_pred[np.where(y_true == 0)])
    Ppr = TP + FP
    precision = TP / Ppr if Ppr != 0 else 0.0
    Npr = TN + FN
    NPV = TN / Npr if Npr != 0 else 0.0
    recall = TP / P if P != 0 else 0.0
    forecast_bias = min(1.0, (Ppr - P) / P) if P != 0 else max(-1.0, -(Npr - N) / N)
    return {
        "P": P,
        "N": N,
        "TP": TP,
        "FN": FN,
        "TN": TN,
        "FP": FP,
        "NPV": round(NPV, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Accuracy": round(np.mean(y_true == y_pred), 4),
        "F1": round(2 * precision * recall / (precision + recall), 4)
        if (precision != 0 or recall != 0)
        else 0.0,
        "PredictiveScore": round(2 * precision * NPV / (precision + NPV), 4),
        "ForecastBias": round(forecast_bias, 4),
    }
