import click

from app.run_experiments import ExperimentRunner

from redis import Redis
from rq import Queue


@click.command(help="Schedule experiments.")
def main():
    q = Queue(connection=Redis())
    q.enqueue(ExperimentRunner().run, job_timeout="24h")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
