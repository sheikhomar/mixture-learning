# Mixture Learning

A Python package to learn arbitrary mixtures.

## Getting Started

This projects relies on [pyenv](https://github.com/pyenv/pyenv) and [Poetry](https://python-poetry.org/docs/).

1. Install the required Python version:

   ```bash
   pyenv install
   ```

2. Install dependencies

   ```bash
   poetry install --no-dev
   ```

3. Create jobs

   ```bash
   poetry run python -m app.create_jobs -q data/queue -o data/experiments -t rwe -n n_dim -r 1
   ```

4. Run worker

   ```bash
   poetry run python -m app.run_worker -q data/queue --n-threads 1 --max-active 1
   ```

5. Download experiments results from server

   ```bash
   rsync -av skadi:/home/omar/code/mixture-learning/data/experiments/ data/experiments-skadi
   ```
