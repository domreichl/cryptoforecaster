import click

from utils.config import Config
from utils.data_preparation import download_data, update_data
from utils.file_handling import DataHandler
from utils.models import tune_model, test_model
from utils.plotting import plot_test_results, plot_tuning_results


@click.group()
def cli():
    pass


@cli.command()
def download():
    df = download_data()
    DataHandler().write_csv_data(df, Config().symbol)


@cli.command()
def update():
    update_data(Config().symbol)


@cli.command()
def tune():
    tune_model()


@cli.command()
def test():
    test_model()


@cli.command()
@click.argument("plot_type")
def plot(plot_type: str):
    match plot_type:
        case "tuning":
            plot_tuning_results()
        case "test":
            plot_test_results()


for cmd in [download, update, tune, test, plot]:
    cli.add_command(cmd)
