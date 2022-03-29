import pandas as pd
import numpy as np


def apply_noise_to_labels(df: pd.DataFrame, label_noise_proba: float):
    df["is_label_corrupted"] = np.random.binomial(n=1, p=label_noise_proba, size=df.shape[0])
    
    available_labels = set(df.true_label.unique())

    def compute_noisy_label(row):
        current_label = row["true_label"]
        if row["is_label_corrupted"]:
            other_labels = list(available_labels - set([current_label]))
            return np.random.choice(other_labels)
        return current_label
    
    df["noisy_label"] = df.apply(compute_noisy_label, axis=1).astype(int)
