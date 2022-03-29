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
3. Prepare Redis

   ```bash
   docker volume create ml-redis-data
   docker run --name ml-redis -v ml-redis-data:/data -p 6379:6379 -d redis redis-server --save 60 1 --loglevel warning
   ```
