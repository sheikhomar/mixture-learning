import os

from datetime import datetime
from pathlib import Path

import click
import numpy as np

from app.experiments.rwe import generate_experiments
from app.run_worker import JobInfo


class JobCreator:
    def __init__(self, queue_dir: str, experiment_type: str, experiment_name: str, output_dir: str) -> None:
        self._queue_dir = Path(queue_dir)
        self._experiment_type = experiment_type
        self._experiment_name = experiment_name
        self._output_dir = Path(output_dir)

    def run(self) -> None:
        ready_dir = self._queue_dir / "ready"
        os.makedirs(ready_dir, exist_ok=True)

        for exp_params in generate_experiments(self._experiment_name):
            time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
            experiment_no = f"{time_stamp}-{np.random.randint(0, 1e5):06d}"
            working_dir = self._output_dir / self._experiment_type / self._experiment_name / experiment_no
            os.makedirs(working_dir, exist_ok=True)

            params_path = working_dir / "experiment-params.json"
            exp_params.write_json(params_path)

            job_info = JobInfo()
            job_info.working_dir = working_dir
            job_info.command = f"poetry run python -m app.run_experiment"
            job_info.command_params = {
                "experiment-type": self._experiment_type,
                "experiment-name": self._experiment_name,
                "params-path": str(params_path),
                "working-dir": str(working_dir),
            }

            job_name = f"{self._experiment_type}.{self._experiment_name}.{experiment_no}.json"
            job_info.write_json(ready_dir / job_name)


@click.command(help="Create jobs.")
@click.option(
    "-q",
    "--queue-dir",
    type=click.STRING,
    required=True,
)
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
    "-o",
    "--output-dir",
    type=click.STRING,
    required=True,
)
def main(queue_dir: str, experiment_type: str, experiment_name: str, output_dir: str):
    JobCreator(
        queue_dir=queue_dir,
        experiment_type=experiment_type,
        experiment_name=experiment_name,
        output_dir=output_dir,
    ).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
