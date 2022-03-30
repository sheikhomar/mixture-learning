import click

from threadpoolctl import threadpool_limits

from app.algorithms import random_walk
from app.data.datasets import SyntheticDataSet


class ExperimentRunner:
    def __init__(self) -> None:
        pass
        
    def run(self) -> None:
        with threadpool_limits(limits=1):
            self._run()
    
    def _run(self) -> None:
        data_set = SyntheticDataSet(
            n_dim=30,
            min_distance=4,
            component_size=50,
            variance=0.5,
            label_noise_proba=0.0,
        )

        X = data_set.features.to_numpy()
        y = data_set.labels.to_numpy()

        embeddings = random_walk.compute_random_walk_embeddings(
            X=X,
            y=y,
            bias_factor=1,
            allow_self_loops=False,
            distance_metric="sqeuclidean",
            n_steps=10,
        )

        print(f"Done! Embedding shape {embeddings.shape}")


@click.command(help="Runs experiments.")
def main():
    ExperimentRunner().run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
