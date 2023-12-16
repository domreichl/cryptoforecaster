import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

from utils.config import Config, TuningParameters
from utils.data_preprocessing import preprocess_data
from utils.evaluation import compute_metrics
from utils.file_handling import get_root_dir, ResultsHandler


class RNN(keras.models.Model):
    def __init__(self, n_units: int, n_layers: int, dropout_rate: float):
        super().__init__()
        self.recurrent_layers = [
            keras.layers.GRU(n_units // 2**i, return_sequences=True)
            for i in range(n_layers - 1)
        ]
        self.recurrent_layers.append(keras.layers.GRU(n_units // 2 ** len(self.layers)))
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.out = keras.layers.Dense(1, activation="sigmoid")

    def call(self, x):
        for layer in self.recurrent_layers:
            x = self.dropout(layer(x))
        return self.out(x)


def train_model(x: np.array, y: np.array, cfg: dict) -> RNN:
    model = RNN(
        cfg["units"],
        cfg["layers"],
        cfg["dropout_rate"],
    )
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(1e-3),
    )
    model.fit(
        x,
        y,
        validation_split=0.1,
        epochs=100,
        batch_size=32,
        class_weight=compute_class_weights(y),
        callbacks=[
            keras.callbacks.EarlyStopping(
                "val_loss", patience=5, restore_best_weights=True
            ),
            keras.callbacks.LearningRateScheduler(
                keras.optimizers.schedules.ExponentialDecay(1e-3, 1, 0.95),
            ),
        ],
    )
    return model


def uptrain_model(model: RNN, x: np.array, y: np.array, epochs: int) -> None:
    model.fit(x, y, epochs=epochs, batch_size=32, class_weight=compute_class_weights(y))


def compute_class_weights(y: np.array, classes: list = [0, 1]) -> dict:
    weights = compute_class_weight("balanced", classes=classes, y=y.flatten())
    return {c: weights[c] for c in classes}


def predict(model: RNN, x: np.array) -> np.array:
    return np.int32(np.squeeze(model.predict(x)) > 0.5)


def load_model() -> RNN:
    return keras.models.load_model(get_ckpt_dir())


def save_model(model: RNN) -> None:
    model.save(get_ckpt_dir())


def get_ckpt_dir() -> Path:
    return get_root_dir() / "ckpt"


def tune_model() -> None:
    tp = TuningParameters()

    def objective(trial) -> tuple:
        mts = preprocess_data(
            look_back_window=trial.suggest_int("look_back_window", 2, 365 * 4),
            forecast_window=trial.suggest_int("forecast_window", 1, 14),
            split_type="validation",
        )
        ppv_lst, acc_lst, f1_lst, ps_lst = [], [], [], []
        for _ in range(tp.replications):
            model = train_model(
                mts.x_train,
                mts.y_train,
                cfg={
                    "units": trial.suggest_int("units", 2, 512),
                    "layers": trial.suggest_int("layers", 1, 10),
                    "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
                },
            )
            y_pred = predict(model, mts.x_test)
            print("PR:", y_pred)
            print("GT:", mts.y_test)
            metrics = compute_metrics(mts.y_test, y_pred)
            print("Current Metrics:")
            [print(f" {k}: {v}") for k, v in metrics.items()]
            ppv_lst.append(metrics["Precision"])
            acc_lst.append(metrics["Accuracy"])
            f1_lst.append(metrics["F1"])
            ps_lst.append(metrics["PredictiveScore"])
        return (
            round(np.mean(ppv_lst), 4),
            round(np.mean(acc_lst), 4),
            round(np.mean(f1_lst), 4),
            round(np.mean(ps_lst), 4),
        )

    study = optuna.create_study(
        directions=["maximize"] * 4,
        sampler=optuna.samplers.GridSampler(tp.hparams),
    )
    study.optimize(objective)
    df = study.trials_dataframe()
    df = df[
        ["params_" + k for k in tp.hparams.keys()]
        + ["values_0", "values_1", "values_2", "values_3"]
    ]
    df.columns = [k for k in tp.hparams.keys()] + [
        "Precision",
        "Accuracy",
        "F1",
        "PredictiveScore",
    ]
    df["Replications"] = tp.replications
    df["BuyThreshold"] = Config().buy_threshold
    df.sort_values("Precision", ascending=False, inplace=True)
    ResultsHandler().write_csv_results(df, "tuning", append=True)


def test_model() -> None:
    mts = preprocess_data(split_type="test")
    model = train_model(mts.x_train, mts.y_train, Config().hparams)
    y_pred = predict(model, mts.x_test)
    metrics = compute_metrics(mts.y_test, y_pred)
    print(metrics)
    ResultsHandler().write_csv_results(
        pd.DataFrame(metrics, index=[0]), "test", append=True
    )
