import json

from importlib import import_module
from typing import Dict

import click

from threadpoolctl import threadpool_limits

from app.experiments import Experiment
from app.helpers.logger import get_logger

logger = get_logger("runner")

class ExperimentRunner:
    def __init__(
        self, experiment_type: str,
        experiment_name: str,
        params_path: str,
        working_dir: str,
        n_threads: int
    ) -> None:
        self._experiment_type = experiment_type
        self._experiment_name = experiment_name
        self._params_path = params_path
        self._working_dir = working_dir
        self._n_threads = n_threads
        
    def run(self) -> None:
        with open(self._params_path, "r") as f:
            params = json.load(f)
        experiment = self._initialize_experiment(params=params)
        with threadpool_limits(limits=self._n_threads):
            experiment.run()

    def _initialize_experiment(self, params: Dict[str, object]) -> Experiment:
        try:
            module_name = f"app.experiments.{self._experiment_type}"
            experiment_module = import_module(module_name)
        except ModuleNotFoundError:
            raise ValueError(f"Experiment module '{module_name}' was not found.")

        experiment = experiment_module.initialize_experiment(
            name=self._experiment_name,
            params=params,
            working_dir=self._working_dir
        )  # type: ignore
        return experiment

@click.command(help="Runs an experiment.")
@click.option(
    "-t",
    "--experiment-type",
    type=click.STRING,
    required=True,
)
@click.option(
    "-n",
    "--experiment-name",
    type=click.STRING,
    required=True,
)
@click.option(
    "-p",
    "--params-path",
    type=click.STRING,
    required=True,
)
@click.option(
    "-w",
    "--working-dir",
    type=click.STRING,
    required=True,
)
@click.option(
    "--n-threads",
    type=click.INT,
    required=True,
)
def main(experiment_type: str, experiment_name: str, params_path: str, working_dir: str, n_threads: int):
    ExperimentRunner(
        experiment_type=experiment_type,
        experiment_name=experiment_name,
        params_path=params_path,
        working_dir=working_dir,
        n_threads=n_threads,
    ).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
