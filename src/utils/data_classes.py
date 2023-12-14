import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class MultivariateTimeSeries:
    dates: pd.Series
    x_train: np.array
    y_train: np.array
    x_test: np.array
    y_test: np.array

    def get_slided_windows(self, origin: int) -> np.array:
        t = len(self.x_train) + origin
        x = np.concatenate([self.x_train, self.x_test], 0)
        y = np.concatenate([self.y_train, self.y_test], 0)
        x_train, y_train, x_test, y_test = x[origin:t], y[origin:t], x[t:], y[t:]
        return x_train, y_train, x_test, y_test
