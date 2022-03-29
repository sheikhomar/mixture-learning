
import numpy as np
from numba import jit

from sklearn.metrics import pairwise_distances


@jit(nopython=True, fastmath=True)
def random_walk_step(transition_matrix: np.ndarray, energy_vector: np.ndarray) -> np.ndarray:
    energy_vector = np.dot(transition_matrix, energy_vector)  # Power method
    energy_vector = energy_vector / np.linalg.norm(energy_vector)  # Normalize
    return energy_vector


@jit(nopython=True, fastmath=True)
def embed_point_using_random_walk(transition_matrix: np.ndarray, starting_point: int, n_steps: int=1000):
    r = np.zeros(transition_matrix.shape[0])
    r[starting_point] = 1
    for _ in range(n_steps):
        r = random_walk_step(transition_matrix, r)
    return r


def compute_random_walk_embeddings(X: np.ndarray, y: np.ndarray, bias_factor: float, distance_metric: str):
    distance_matrix = pairwise_distances(X, metric=distance_metric)
    similarity_matrix = np.exp(-distance_matrix)
    
    classes = np.unique(y)
    n_points = X.shape[0]
    
    embeddings = np.zeros(shape=(n_points, n_points))
    
    for target_class in classes:
        S = similarity_matrix.copy()

        filter_vector = np.array(y == target_class).reshape(-1, 1)
        filter_mask = np.dot(filter_vector, filter_vector.T)
        S[filter_mask] *= bias_factor

        transition_matrix = (S / S.sum(axis=1).reshape(-1, 1)).T
        
        class_point_indicies = np.where(y == target_class)[0]
        
        for point_index in class_point_indicies:
            embedding = embed_point_using_random_walk(transition_matrix, point_index)
            embeddings[point_index] = embedding
    return embeddings
