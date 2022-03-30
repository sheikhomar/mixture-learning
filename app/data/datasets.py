import abc

import numpy as np

from app.data.synthetic import generate_simplified_data
from app.data.noise import apply_noise_to_labels

class DataSet(abc.ABC):
    @property
    @abc.abstractmethod
    def size(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_classes(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def label_attribute(self) -> str:
        """The name for the label attribute."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def labels(self) -> np.ndarray:
        """Returns the labels of all points in the data set."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def features(self) -> np.ndarray:
        """Returns the features of all points in the data set."""
        raise NotImplementedError


class SyntheticDataSet(DataSet):
    def __init__(self, n_dim: int, min_distance: float, component_size: int, variance: float=0.5, label_noise_proba: float=0.0) -> None:
        super().__init__()
        df_data = generate_simplified_data(
            dim=n_dim,
            min_dist=min_distance,
            component_size=component_size,
            variance=variance
        )

        apply_noise_to_labels(df=df_data, label_noise_proba=label_noise_proba)

        self._feature_names = [n for n in df_data.columns if n.startswith("feat_")]
        self._n_classes = df_data[self.label_attribute].nunique()
        self._df = df_data

    @property
    def size(self) -> int:
        return self._df.shape[0]

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def label_attribute(self) -> str:
        return "noisy_label"

    @property
    def true_label_attribute(self) -> str:
        return "true_label"

    @property
    def features(self) -> np.ndarray:
        return self._df[self._feature_names]

    @property
    def labels(self) -> np.ndarray:
        return self._df[self.label_attribute]

    @property
    def true_labels(self) -> np.ndarray:
        return self._df[self.true_label_attribute]
