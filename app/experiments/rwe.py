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

    def to_json(self, file_path: Path):
        with open(file_path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4, sort_keys=False)


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

        params_str = json.dumps(dataclasses.asdict(self._params), indent=4)
        logger.debug(f"Running RandomWalkEmbeddingExperiment with params: \n{params_str}\n")

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
    for n_dim in [2, 5, 10, 20, 30, 50, 70, 100]:
        params = generate_default_experiment()
        params["n_dim"] = n_dim
        yield from_dict(
            data_class=RandomWalkEmbeddingExperimentParams, 
            data=params
        )

def generate_experiments(experiment_type: str) -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    if experiment_type == "n_dim":
        return generate_dimension_experiments()



# Following functions are must be present for each experiment type

def initialize_experiment(name: str, params: Dict[str, object], working_dir: Path) -> Experiment:
    return RandomWalkEmbeddingExperiment(
        experiment_params=params,
        working_dir=working_dir
    )
