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
   poetry install
   ```

3. Create jobs

   ```bash
   poetry run python -m app.create_jobs -q data/queue -o data/experiments -t rwe -n n_dim
   ```

4. Run worker

   ```bash
    poetry run python -m app.run_worker -q data/queue --n-threads 1 --max-active 1 
   ```
