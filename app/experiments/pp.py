import dataclasses, os
import json

from pathlib import Path
from typing import Dict, List, Generator

import numpy as np
import pandas as pd

from dacite import from_dict
from graspologic.simulations import sbm
from sklearn.cluster import KMeans

from app.algorithms import random_walk
from app.experiments import Experiment, ExperimentParams
from app.helpers.logger import get_logger
from app.helpers.evaluation import compute_performance_scores


@dataclasses.dataclass
class PlantedPartitionExperimentParams(ExperimentParams):
    k: int
    nodes_per_cluster: int
    p: float
    q: float
    n_steps: int
    bias_factor: float
    random_seed: int


class PlantedPartitionExperiment(Experiment):
    def __init__(self, experiment_params: Dict[str, object], working_dir: str) -> None:
        super().__init__()
        self._params: PlantedPartitionExperimentParams = from_dict(
            data_class=PlantedPartitionExperimentParams, 
            data=experiment_params
        )
        self._working_dir = working_dir

    def run(self) -> None:
        logger = get_logger(name="pp")

        logger.debug(f"Running PlantedPartitionExperiment with params: \n{self._params}")

        np.random.default_rng(self._params.random_seed)
        np.random.seed(self._params.random_seed)

        output_dir = Path(self._working_dir)
    
        logger.debug("Generating planted partition graph")
        X, y = self._generate_planted_partition_graph()

        logger.debug(f"The shape of the generated data matrix is: {X.shape}")

        logger.debug(f"Computing embeddings via random walks.")

        n_nodes = X.shape[0]
        transition_matrix = (X / X.sum(axis=1).reshape(-1, 1)).T
        embeddings = np.zeros(shape=(n_nodes, n_nodes))
        node_indices = np.arange(0, n_nodes)

        random_walk.compute_point_embeddings(
            transition_matrix=transition_matrix,
            point_indices=node_indices,
            n_steps=self._params.n_steps,
            embeddings=embeddings,
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
            "true_label": y,
            "cluster_label": kmeans.labels_
        })

        df_results.to_feather(results_path)

        scores = compute_performance_scores(df_results=df_results)
        with open(output_dir / "scores.json", "w") as fp:
            json.dump(scores, fp)

        with open(output_dir / "done.out", "w") as f:
            f.write("done")

    def _generate_planted_partition_graph(self):
        sizes = [self._params.nodes_per_cluster] * self._params.k
        labels = [label for label, size in enumerate(sizes) for _ in range(size)]
        
        n_blocks = len(sizes)
        proba_out = self._params.q
        proba_in = self._params.p

        probability_matrix = np.ones(shape=(n_blocks, n_blocks)) * proba_out
        np.fill_diagonal(probability_matrix, proba_in)
        adjacency_matrix = sbm(n=sizes, p=probability_matrix,)

        return adjacency_matrix, labels

    def _write_results(self, df_results: pd.DataFrame, output_path: Path):
        scores = compute_performance_scores(df_results=df_results)
        with open(output_path, "w") as fp:
            json.dump(scores, fp)


def generate_default_experiment() -> Dict[str, object]:
    return dict(
        k=10,
        nodes_per_cluster=100,
        p=0.6,
        q=0.1,
        n_steps=1,
        bias_factor=1,
        random_seed=int.from_bytes(os.urandom(3), "big"),
    )


def steps_01() -> Generator[object, None, None]:
    n_step = 1
    for k in [10, 20, 50, 100]:
        for q in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for p_add in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
                p = q + p_add
                params = generate_default_experiment()
                params["k"] = k
                params["n_steps"] = n_step
                params["p"] = p
                params["q"] = q

                yield from_dict(data_class=PlantedPartitionExperimentParams, data=params)


def steps_02() -> Generator[object, None, None]:
    n_step = 2
    for k in [10, 20, 50, 100]:
        for q in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for p_add in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
                p = q + p_add
                params = generate_default_experiment()
                params["k"] = k
                params["n_steps"] = n_step
                params["p"] = p
                params["q"] = q

                yield from_dict(data_class=PlantedPartitionExperimentParams, data=params)


def steps_05() -> Generator[object, None, None]:
    n_step = 5
    for k in [10, 20, 50, 100]:
        for q in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for p_add in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
                p = q + p_add
                params = generate_default_experiment()
                params["k"] = k
                params["n_steps"] = n_step
                params["p"] = p
                params["q"] = q

                yield from_dict(data_class=PlantedPartitionExperimentParams, data=params)


def steps_10() -> Generator[object, None, None]:
    n_step = 10
    for k in [10, 20, 50, 100]:
        for q in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for p_add in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
                p = q + p_add
                params = generate_default_experiment()
                params["k"] = k
                params["n_steps"] = n_step
                params["p"] = p
                params["q"] = q

                yield from_dict(data_class=PlantedPartitionExperimentParams, data=params)


def steps_100() -> Generator[object, None, None]:
    n_step = 100
    for k in [10, 20, 50, 100]:
        for q in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for p_add in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
                p = q + p_add
                params = generate_default_experiment()
                params["k"] = k
                params["n_steps"] = n_step
                params["p"] = p
                params["q"] = q

                yield from_dict(data_class=PlantedPartitionExperimentParams, data=params)


def e2_steps_20() -> Generator[object, None, None]:
    n_steps = 20
    for k in [10, 20, 50, 100]:
        for q in [0.01, 0.10, 0.15, 0.20, 0.3]:
            for p_add in [0.2, 0.30, 0.45, 0.60]:
                p = q + p_add
                params = generate_default_experiment()
                params["k"] = k
                params["n_steps"] = n_steps
                params["p"] = p
                params["q"] = q

            yield from_dict(data_class=PlantedPartitionExperimentParams, data=params)


def e2_steps_20_v2() -> Generator[object, None, None]:
    n_steps = 20
    for k in [10, 20, 50]:
        for q in [0.01, 0.10, 0.20, 0.3]:
            for p_add in [0.2, 0.3, 0.45, 0.6]:
                p = q + p_add
                params = generate_default_experiment()
                params["k"] = k
                params["n_steps"] = n_steps
                params["p"] = p
                params["q"] = q
                yield from_dict(data_class=PlantedPartitionExperimentParams, data=params)


AVAILABLE_EXPERIMENTS = [
    steps_01,
    steps_02,
    steps_05,
    steps_10,
    steps_100,
    e2_steps_20,
    e2_steps_20_v2,
]


# Following functions must be present in each experiment module

def initialize_experiment(name: str, params: Dict[str, object], working_dir: Path) -> Experiment:
    return PlantedPartitionExperiment(
        experiment_params=params,
        working_dir=working_dir
    )


def generate_experiments(experiment_name: str) -> Generator[ExperimentParams, None, None]:
    lookup_table = {f.__name__: f for f in AVAILABLE_EXPERIMENTS }
    if experiment_name in lookup_table:
        return lookup_table[experiment_name]()

    raise Exception(f"Unknown experiment: {experiment_name}")
