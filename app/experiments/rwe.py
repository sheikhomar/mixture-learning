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

def dim_dist_02() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_dim in [250, 300]:
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

def dim_steps_01() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_dim in [10, 50, 100, 150, 200]:
        for n_steps in [1, 2, 5, 10]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["min_distance"] = 2
            params["bias_factor"] = n_dim**(1/4)
            params["n_steps"] = n_steps
            params["variance"] = 1.0
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def dim_steps_02() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_dim in [10, 50, 100, 150, 200]:
        for n_steps in [1, 2, 5, 10]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["min_distance"] = 3
            params["bias_factor"] = n_dim**(1/4)
            params["n_steps"] = n_steps
            params["variance"] = 1.0
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def dim_steps_03() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    n_dim = 50
    for n_steps in [10, 20, 50, 100]:
        params = generate_default_experiment()
        params["n_dim"] = n_dim
        params["min_distance"] = 2
        params["bias_factor"] = n_dim**(1/4)
        params["n_steps"] = n_steps
        params["variance"] = 1.0
        yield from_dict(
            data_class=RandomWalkEmbeddingExperimentParams, 
            data=params
        )


def dim_steps_04() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    n_dim = 100
    for n_steps in [10, 20, 50, 100]:
        params = generate_default_experiment()
        params["n_dim"] = n_dim
        params["min_distance"] = 2
        params["bias_factor"] = n_dim**(1/4)
        params["n_steps"] = n_steps
        params["variance"] = 1.0
        yield from_dict(
            data_class=RandomWalkEmbeddingExperimentParams, 
            data=params
        )


def dim_steps_no_bias_01() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_dim in [10, 50, 100]:
        for n_steps in [1, 5, 10, 20, 50, 100]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["min_distance"] = 2
            params["bias_factor"] = 1
            params["n_steps"] = n_steps
            params["variance"] = 1.0
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def no_bias_01() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for dist_factor in [3, 4, 5, 6]:
        for n_dim in [10, 50, 100, 200]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["min_distance"] = np.power(n_dim, 1/4) * dist_factor
            params["bias_factor"] = 1
            params["n_steps"] = 5
            params["variance"] = 1.0
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def no_bias_02() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_steps in [1, 5, 10, 20, 50, 100]:
        for n_dim in [10, 50, 100]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["n_steps"] = n_steps
            params["min_distance"] = np.power(n_dim, 1/4) * 3
            params["bias_factor"] = 1
            params["variance"] = 1.0
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def no_bias_03() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_steps in [1, 5, 10, 20, 50, 100]:
        for n_dim in [10, 50, 100]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["n_steps"] = n_steps
            params["min_distance"] = np.power(n_dim, 1/4) * 5
            params["bias_factor"] = 1
            params["variance"] = 1.0
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def no_bias_04() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_steps in [1, 5, 10, 20, 50, 100]:
        for n_dim in [10, 50, 100]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["n_steps"] = n_steps
            params["min_distance"] = np.power(n_dim, 1/4) * 7
            params["bias_factor"] = 1
            params["variance"] = 1.0
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )

def no_bias_single_step_01() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for dist_factor in [1, 2, 3, 4, 5, 6]:
        for n_dim in [10, 50, 100, 200, 300]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["n_steps"] = 1
            params["min_distance"] = np.power(n_dim, 1/4) * dist_factor
            params["bias_factor"] = 1
            params["variance"] = 1.0
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def no_bias_single_step_02() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_dim in [300, 200, 100, 50, 10]:
        params = generate_default_experiment()
        params["n_dim"] = n_dim
        params["n_steps"] = 1
        params["min_distance"] = np.power(n_dim, 1/4) * (2 + np.log(n_dim))
        params["bias_factor"] = 1
        params["variance"] = 1.0
        yield from_dict(
            data_class=RandomWalkEmbeddingExperimentParams, 
            data=params
        )


def no_bias_single_step_03() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for term in [0.5, 1.0, 1.5]:
        for n_dim in [300, 200, 100, 50, 10]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["n_steps"] = 1
            params["min_distance"] = np.power(n_dim, 1/4) * (term + np.log(n_dim))
            params["bias_factor"] = 1
            params["variance"] = 1.0
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def no_bias_single_step_04() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for term in [0.0]:
        for n_dim in [300, 200, 100, 50, 10]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["n_steps"] = 1
            params["min_distance"] = np.power(n_dim, 1/4) * (term + np.log(n_dim))
            params["bias_factor"] = 1
            params["variance"] = 1.0
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def no_bias_single_step_log10_01() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for term in [1.0, 2.0, 3.0]:
        for n_dim in [300, 200, 100, 50, 10]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["n_steps"] = 1
            params["min_distance"] = np.power(n_dim, 1/4) * (term + np.log10(n_dim))
            params["bias_factor"] = 1
            params["variance"] = 1.0
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def no_bias_single_step_var_01() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for variance in [2.0, 3.0]:
        for n_dim in [300, 200, 100, 50, 10]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["n_steps"] = 1
            params["min_distance"] = np.power(n_dim, 1/4) * (3 + np.log10(n_dim))
            params["bias_factor"] = 1
            params["variance"] = variance
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def no_bias_single_step_var_02() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for variance in [4.0, 5.0, 6.0]:
        for n_dim in [300, 200, 100, 50, 10]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["n_steps"] = 1
            params["min_distance"] = np.power(n_dim, 1/4) * (3 + np.log10(n_dim))
            params["bias_factor"] = 1
            params["variance"] = variance
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def no_bias_single_step_var_03() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for variance in [8.0, 10.0]:
        for n_dim in [300, 200, 100, 50, 10]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["n_steps"] = 1
            params["min_distance"] = np.power(n_dim, 1/4) * (3 + np.log10(n_dim))
            params["bias_factor"] = 1
            params["variance"] = variance
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )

def no_bias_single_step_var_04() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for variance in [2, 3, 4, 5, 6, 8, 10]:
        for n_dim in [300, 200, 100, 50, 10]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["n_steps"] = 1
            params["min_distance"] = np.power(n_dim, 1/4) * (3 + np.log10(n_dim)) * variance
            params["bias_factor"] = 1
            params["variance"] = variance
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def no_bias_single_step_var_05() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for variance in [2, 3, 4, 5, 6, 8, 10]:
        for n_dim in [300, 200, 100, 50, 10]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["n_steps"] = 1
            params["min_distance"] = np.power(n_dim, 1/4) * (3 + np.log10(n_dim)) * np.sqrt(variance)
            params["bias_factor"] = 1
            params["variance"] = variance
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def no_bias_single_step_var_06() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for variance in [2, 3, 4, 5, 6, 8, 10]:
        for n_dim in [300, 200, 100, 50, 10]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["n_steps"] = 1
            params["min_distance"] = np.power(n_dim, 1/4) * (3 + np.log10(n_dim)) * (variance + np.power(n_dim, 1/64))
            params["bias_factor"] = 1
            params["variance"] = variance
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def no_bias_single_step_var_07() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for variance in [2, 3, 4, 5, 6, 8, 10]:
        for n_dim in [100, 50, 10]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["n_steps"] = 1
            params["min_distance"] = np.power(n_dim, 1/4) * (3 + np.log10(n_dim)) * (variance + np.power(n_dim, 1/4))
            params["bias_factor"] = 1
            params["variance"] = variance
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def no_bias_single_step_var_08() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for variance in [2, 3, 4, 5, 6, 8, 10]:
        for n_dim in [100, 50, 10]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["n_steps"] = 1
            params["min_distance"] = np.power(n_dim, 1/4) * (3 + np.log10(n_dim)) * (variance + np.power(n_dim, 1/2))
            params["bias_factor"] = 1
            params["variance"] = variance
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def no_bias_single_step_var_09() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for variance in [2, 3, 4, 5, 6, 8, 10]:
        for n_dim in [100, 50, 10]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["n_steps"] = 1
            params["min_distance"] = np.power(n_dim, 1/4) * (3 + np.log10(n_dim)) * (variance + n_dim)
            params["bias_factor"] = 1
            params["variance"] = variance
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def no_bias_multi_step_var_01() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_step in [2, 3, 4, 5, 7, 10]:
        variance = 10
        for n_dim in [100, 50, 10]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["n_steps"] = n_step
            params["min_distance"] = np.power(n_dim, 1/4) * (3 + np.log10(n_dim)) * (variance + n_dim)
            params["bias_factor"] = 1
            params["variance"] = variance
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def no_bias_multi_step_01() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_steps in [6, 5, 4, 3, 2]:
        for dist_factor in [1, 2, 3, 4, 5, 6]:
            for n_dim in [300, 200, 100, 50, 10]:
                params = generate_default_experiment()
                params["n_dim"] = n_dim
                params["n_steps"] = n_steps
                params["min_distance"] = np.power(n_dim, 1/4) * dist_factor
                params["bias_factor"] = 1
                params["variance"] = 1.0
                yield from_dict(
                    data_class=RandomWalkEmbeddingExperimentParams, 
                    data=params
                )


def dim_bias_01() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_dim in [10, 50, 100, 200]:
        for beta in [1/4, 1/8, 1/16, 1/32]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["min_distance"] = 2
            params["bias_factor"] = n_dim**beta
            params["n_steps"] = 10
            params["variance"] = 1.0
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def dim_bias_02() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_dim in [10, 50, 100, 200]:
        for beta in [1/64, 1/128, 1/256, 1/512]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["min_distance"] = 2
            params["bias_factor"] = n_dim**beta
            params["n_steps"] = 10
            params["variance"] = 1.0
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def dim_bias_03() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_dim in [250, 300]:
        for beta in [1/512, 1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["min_distance"] = 2
            params["bias_factor"] = n_dim**beta
            params["n_steps"] = 10
            params["variance"] = 1.0
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def small_dist_01() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_dim in [10, 50, 100, 150, 200]:
        for min_distance in [1.5, 1.0]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["min_distance"] = min_distance
            params["bias_factor"] = n_dim**(1/4)
            params["n_steps"] = 10
            params["variance"] = 1.0
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def dist_below_one_01() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for n_dim in [10, 50, 100, 150, 200]:
        for dist_factor in [64, 16, 8, 4]:
            params = generate_default_experiment()
            params["n_dim"] = n_dim
            params["min_distance"] = 1 / np.power(n_dim, 1/dist_factor)
            params["bias_factor"] = n_dim**(1/4)
            params["n_steps"] = 10
            params["variance"] = 1.0
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def noisy_10_fixed_dist_01() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for dist_factor in [1, 3, 5, 7, 10]:
        for n_dim in [10, 50, 100, 150, 200]:
            params = generate_default_experiment()
            params["label_noise_proba"] = 0.1
            params["n_dim"] = n_dim
            params["min_distance"] = dist_factor # np.power(n_dim, 1/4) * dist_factor
            params["bias_factor"] = 1
            params["n_steps"] = 10
            params["variance"] = 1.0
            
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def noisy_10_fixed_dist_02() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for dist_factor in [3, 4, 5, 6]:
        for n_dim in [10, 50, 100, 150, 200]:
            params = generate_default_experiment()
            params["label_noise_proba"] = 0.1
            params["n_dim"] = n_dim
            params["min_distance"] = np.power(n_dim, 1/4) * dist_factor
            params["bias_factor"] = 1
            params["n_steps"] = 10
            params["variance"] = 1.0
            
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def noisy_10_fixed_dist_03() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for dist_factor in [3, 4]:
        for n_dim in [10, 50, 100, 150, 200]:
            params = generate_default_experiment()
            params["label_noise_proba"] = 0.1
            params["n_dim"] = n_dim
            params["min_distance"] = np.power(n_dim, 1/3) * dist_factor
            params["bias_factor"] = 1
            params["n_steps"] = 10
            params["variance"] = 1.0
            
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def noisy_10_fixed_dist_04() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for dist_factor in [2, 3]:
        for n_dim in [10, 50, 100, 200]:
            params = generate_default_experiment()
            params["label_noise_proba"] = 0.1
            params["n_dim"] = n_dim
            params["min_distance"] = np.power(n_dim, 1/2) * dist_factor
            params["bias_factor"] = 1
            params["n_steps"] = 10
            params["variance"] = 1.0
            
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )


def noisy_10_fixed_dist_05() -> Generator[RandomWalkEmbeddingExperimentParams, None, None]:
    for dist_factor in [2, 3]:
        for n_dim in [10, 50, 100, 200]:
            params = generate_default_experiment()
            params["label_noise_proba"] = 0.1
            params["n_dim"] = n_dim
            params["min_distance"] = np.power(n_dim, 1/2) * dist_factor
            params["bias_factor"] = 1
            params["n_steps"] = 5
            params["variance"] = 1.0
            
            yield from_dict(
                data_class=RandomWalkEmbeddingExperimentParams, 
                data=params
            )



AVAILABLE_EXPERIMENTS = [
    dim_dist_01,
    dim_dist_02,
    dim_steps_01,
    dim_steps_02,
    dim_steps_03,
    dim_steps_04,
    dim_steps_no_bias_01,
    no_bias_01,
    no_bias_02,
    no_bias_03,
    no_bias_04,
    dim_bias_01,
    dim_bias_02,
    dim_bias_03,
    small_dist_01,
    dist_below_one_01,
    noisy_10_fixed_dist_01,
    noisy_10_fixed_dist_02,
    noisy_10_fixed_dist_03,
    noisy_10_fixed_dist_04,
    noisy_10_fixed_dist_05,
    no_bias_single_step_01,
    no_bias_single_step_02,
    no_bias_single_step_03,
    no_bias_single_step_04,
    no_bias_single_step_log10_01,
    no_bias_multi_step_01,
    no_bias_single_step_var_01,
    no_bias_single_step_var_02,
    no_bias_single_step_var_03,
    no_bias_single_step_var_04,
    no_bias_single_step_var_05,
    no_bias_single_step_var_06,
    no_bias_single_step_var_07,
    no_bias_single_step_var_08,
    no_bias_single_step_var_09,
    no_bias_multi_step_var_01,
]


# Following functions are must be present for each experiment type

def initialize_experiment(name: str, params: Dict[str, object], working_dir: Path) -> Experiment:
    return RandomWalkEmbeddingExperiment(
        experiment_params=params,
        working_dir=working_dir
    )


def generate_experiments(experiment_name: str) -> Generator[ExperimentParams, None, None]:
    if experiment_name == "n_dim":
        return generate_dimension_experiments()
    if experiment_name == "dist_dim":
        return dimensions_vs_distance_metric()
    if experiment_name == "dim_dist_steps_bias_01":
        return dim_dist_steps_bias_01()

    lookup_table = {f.__name__: f for f in AVAILABLE_EXPERIMENTS }
    if experiment_name in lookup_table:
        return lookup_table[experiment_name]()

    raise Exception(f"Unknown experiment: {experiment_name}")

