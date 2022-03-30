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


def compute_corruption_ratio_per_class(df: pd.DataFrame):
    def count_proportion_of_corruption(df_sub_group):
        filter_corrupted = df_sub_group["is_label_corrupted"] == 1
        n_group_size = df_sub_group.shape[0]
        df_corrupted_count = df_sub_group[filter_corrupted][["is_label_corrupted"]].count()
        return df_corrupted_count / n_group_size
        
    df_result = df.groupby(["true_label"]).apply(count_proportion_of_corruption)
    df_result.rename(columns={"is_label_corrupted": "corruption_ratio"}, inplace=True)
    return df_result
