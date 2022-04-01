
import numpy as np
from numba import jit

from sklearn.metrics import pairwise_distances

from app.helpers.logger import get_logger

logger = get_logger("random_walk")


@jit(nopython=True, fastmath=True)
def random_walk_step(transition_matrix: np.ndarray, energy_vector: np.ndarray) -> np.ndarray:
    energy_vector = np.dot(transition_matrix, energy_vector)  # Power method
    energy_vector = energy_vector / np.linalg.norm(energy_vector)  # Normalize
    return energy_vector


@jit(nopython=True, fastmath=True)
def embed_point_using_random_walk(transition_matrix: np.ndarray, starting_point: int, n_steps: int):
    r = np.zeros(transition_matrix.shape[0])
    r[starting_point] = 1
    for _ in range(n_steps):
        r = random_walk_step(transition_matrix, r)
    return r


@jit(nopython=True, fastmath=True)
def compute_point_embeddings(transition_matrix: np.ndarray, point_indices: np.ndarray, n_steps: int, embeddings: np.ndarray):
    for point_index in point_indices:
        r = np.zeros(transition_matrix.shape[0])
        r[point_index] = 1
        for _ in range(n_steps):
            r = random_walk_step(transition_matrix, r)
        embeddings[point_index] = r


def compute_random_walk_embeddings(X: np.ndarray, y: np.ndarray, distance_metric: str, allow_self_loops: bool, bias_factor: float, n_steps: int):
    logger.debug("Computing pairwise distances")
    distance_matrix = pairwise_distances(X, metric=distance_metric)

    logger.debug("Computing similarity matrix")

    # Due to curse of dimensionality, the average distance between pair of points
    # increase as the number of dimensions are increased. To avoid convergence issues
    # with the random walk, divide the pair-wise distances with the number of dimensions.
    n_dim = X.shape[1]
    similarity_matrix = np.exp(-distance_matrix / n_dim)
    
    if not allow_self_loops:
        # Set the diagonal entries to zero because we
        # there are no self loops in the graph.
        np.fill_diagonal(similarity_matrix, val=0.0)

    classes = np.unique(y)
    n_points = similarity_matrix.shape[0]

    embeddings = np.zeros(shape=(n_points, n_points))
    
    for target_class in classes:
        logger.debug(f"Processing class {target_class}")
        S = similarity_matrix.copy()

        filter_vector = np.array(y == target_class).reshape(-1, 1)
        filter_mask = np.dot(filter_vector, filter_vector.T)
        S[filter_mask] *= bias_factor

        transition_matrix = (S / S.sum(axis=1).reshape(-1, 1)).T
        
        class_point_indicies = np.where(y == target_class)[0]
        
        compute_point_embeddings(
            transition_matrix=transition_matrix,
            point_indices=class_point_indicies,
            n_steps=n_steps,
            embeddings=embeddings
        )
    return embeddings
