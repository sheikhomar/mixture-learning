import pandas as pd
import numpy as np


def generate_simplified_data(dim: int, min_dist: float, component_size: int, variance: float=0.5):
    means = np.eye(dim) * min_dist
    raw_data = [
        np.random.multivariate_normal(
            mean=means[m],
            size=component_size,
            cov=np.eye(dim) * variance
        )
        for m in range(means.shape[0])
    ]
    data_frames = [
        pd.DataFrame({ **{ f"feat_{j}": X[:, j] for j in range(X.shape[1]) }, 'true_label': i })
        for i, X in enumerate(raw_data)
    ]
    return pd.concat(data_frames).reset_index(drop=True)
