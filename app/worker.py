#!/usr/bin/env python
import click


import sys
from rq import Connection, Worker

# Preload libraries
import app


def run_worker():
    with Connection():
        w = Worker(queues=["default"], name="ml-worker")
        w.work()


@click.command(help="Runs experiments.")
def main():
    run_worker()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
