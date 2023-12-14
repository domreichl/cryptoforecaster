import pandas as pd
from pathlib import Path
from typing import Union


def get_root_dir():
    return Path(__file__).parent.parent.parent


class DataHandler:
    def __init__(self):
        self.data_dir = get_root_dir() / "data"

    def load_csv_data(self, filename: Union[str, Path]) -> pd.DataFrame:
        filename += ".csv"
        if isinstance(filename, Path):
            return pd.read_csv(filename, sep=";")
        else:
            return pd.read_csv(self.data_dir.joinpath(filename), sep=";")

    def write_csv_data(self, df: pd.DataFrame, filename: str) -> None:
        df.to_csv(self.data_dir.joinpath(filename + ".csv"), sep=";", index=False)


class ResultsHandler:
    def __init__(self):
        self.results_dir = get_root_dir() / "results"

    def load_csv_results(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(self.results_dir.joinpath(filename + ".csv"), sep=";")

    def write_csv_results(
        self, df: pd.DataFrame, filename: str, append: bool = False
    ) -> None:
        path = self.results_dir.joinpath(filename + ".csv")
        if append and path.exists():
            df = pd.concat([pd.read_csv(path, sep=";"), df])
        df.to_csv(path, sep=";", index=False)
