import dataclasses, json, os

from random import random
from pathlib import Path
from typing import Dict, List, Generator

import numpy as np
import pandas as pd

from dacite import from_dict
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from app.algorithms import random_walk
from app.data.datasets import SyntheticDataSet
from app.experiments import Experiment, ExperimentParams
from app.helpers.logger import get_logger


@dataclasses.dataclass
class RandomWalkEmbeddingExperimentParams(ExperimentParams):
    n_dim: int
    min_distance: float
    component_size: int
    variance: float
    label_noise_proba: float
    n_steps: int
    distance_metric: str
    bias_factor: float
    allow_self_loops: bool
    random_seed: int


class RandomWalkEmbeddingExperiment(Experiment):
    def __init__(self, experiment_params: Dict[str, object], working_dir: str) -> None:
        super().__init__()
        self._params: RandomWalkEmbeddingExperimentParams = from_dict(
            data_class=RandomWalkEmbeddingExperimentParams, 
            data=experiment_params
        )
        self._working_dir = working_dir

    def run(self) -> None:
        logger = get_logger(name="rwe")

        logger.debug(f"Running RandomWalkEmbeddingExperiment with params: \n{self._params}")

        np.random.default_rng(self._params.random_seed)
        np.random.seed(self._params.random_seed)

        output_dir = Path(self._working_dir)
    
        logger.debug("Generating synthetic data")
        data_set = SyntheticDataSet(
            n_dim=self._params.n_dim,
            min_distance=self._params.min_distance,
            component_size=self._params.component_size,
            variance=self._params.variance,
            label_noise_proba=self._params.label_noise_proba,
        )

        X = data_set.features.to_numpy()
        y = data_set.labels.to_numpy()

        logger.debug(f"The shape of the generated data matrix is: {X.shape}")

        logger.debug(f"Computing embeddings via random walks.")
        embeddings = random_walk.compute_random_walk_embeddings(
            X=X,
            y=y,
            n_steps=self._params.n_steps,
            bias_factor=self._params.bias_factor,
            distance_metric=self._params.distance_metric,
            allow_self_loops=self._params.allow_self_loops,
        )

        logger.debug(f"Done computing embeddings. Matrix shape: {embeddings.shape}")

        n_classes = np.unique(y).shape[0]
        logger.debug(f"Running k-Means with k={n_classes}")
        kmeans = KMeans(n_clusters=n_classes)
        kmeans.fit(embeddings)

        results_path = output_dir / "clustering-results.feather"
        logger.debug(f"Writing clustering results to {results_path}")

        df_results = pd.DataFrame({
            "noisy_label": y,
            "true_label": data_set.true_labels,
            "cluster_label": kmeans.labels_
        })

        df_results.to_feather(results_path)

        with open(output_dir / "done.out", "w") as f:
            f.write("done")



def generate_default_experiment() -> Dict[str, object]:
    return dict(
        n_dim=2,
        min_distance=1,
        component_size=50,
        variance=0.5,
        label_noise_proba=0.0,
        n_steps=10,
        distance_metric="sqeuclidean",
        bias_factor=1,
        allow_self_loops=False,
        random_seed=int.from_bytes(os.urandom(3), "big"),
    )

def generate_dimension_experiments() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_dim in [2, 5, 10, 20, 30, 50, 70, 100, 150, 200]:
        params = generate_default_experiment()
        params["n_dim"] = n_dim
        yield from_dict(
            data_class=RandomWalkEmbeddingExperimentParams, 
            data=params
        )

def dimensions_vs_distance_metric() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_dim in [2, 5, 10, 20, 30, 50, 70, 100, 150, 200]:
        for metric in ["sqeuclidean", "euclidean"]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["distance_metric"] = metric
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams,
                data=params
            )

def dim_dist_steps_bias_01() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_dim in [10, 50, 100, 150, 200]:
        for min_distance in [2, 3, 4, 5, 6]:
            for n_steps in [1, 5, 10, 25, 75]:
                for bias_factor in [1, 2, 3, 5, 10]:
                    params = generate_default_experiment()
                    params["n_dim"] = n_dim
                    params["min_distance"] = min_distance
                    params["n_steps"] = n_steps
                    params["bias_factor"] = bias_factor
                    params["variance"] = 1.0
                    yield from_dict(
                        data_class=RandomWalkEmbeddingExperimentParams, 
                        data=params
                    )

def dim_dist_01() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_dim in [10, 50, 100, 150, 200]:
        for min_distance in [2, 3, 4, 5, 6, 10]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["min_distance"] = min_distance
            params["bias_factor"] = n_dim**(1/4)
            params["n_steps"] = 1
            params["variance"] = 1.0
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )

def generate_experiments(experiment_name: str) -> Generator[ExperimentParams, None, None]:
    if experiment_name == "n_dim":
        return generate_dimension_experiments()
    if experiment_name == "dist_dim":
        return dimensions_vs_distance_metric()
    if experiment_name == "dim_dist_steps_bias_01":
        return dim_dist_steps_bias_01()
    if experiment_name == "dim_dist_01":
        return dim_dist_01()
    raise Exception(f"Unknown experiment: {experiment_name}")


# Following functions are must be present for each experiment type

def initialize_experiment(name: str, params: Dict[str, object], working_dir: Path) -> Experiment:
    return RandomWalkEmbeddingExperiment(
        experiment_params=params,
        working_dir=working_dir
    )
