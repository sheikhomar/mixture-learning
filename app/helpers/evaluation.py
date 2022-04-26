import pandas as pd

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


def assign_labels_based_on_clusters(df_results: pd.DataFrame, new_column_name: str="pred_label") -> pd.DataFrame:
    """
    Assign labels based on the majority voting within each cluster.
    """
    
    # For each cluster, count the number of points that belong to different the class labels
    df_cluster_sizes = df_results.reset_index().groupby("cluster_label")[["index"]].count()
    df_cluster_size_per_label = df_results.reset_index().groupby(["cluster_label", "noisy_label"])[["index"]].count()
    df_cluster_size_per_label["cluster_label_confidence"] = df_cluster_size_per_label / df_cluster_sizes
    df_cluster_size_per_label.rename(columns={"index": "cluster_label_count"}, inplace=True)
    
    # For each cluster, find the class label with the largest number of points.
    df_result_counts = df_cluster_size_per_label.reset_index()

    idx = df_result_counts.groupby(['cluster_label'])['cluster_label_count'].transform(max) == df_result_counts['cluster_label_count']
    df_result_counts["is_largest"] = idx

    # Create a mapping table with `cluster_label` => `new_label``
    # Ensure that each cluster has only one label assignments
    df_new_label_map = df_result_counts[df_result_counts["is_largest"] == True].groupby(["cluster_label", "is_largest"]).sample(1)

    # Select only the columns that we are interested in.
    df_new_label_map = df_new_label_map[["cluster_label", "noisy_label", "cluster_label_confidence"]].reset_index(drop=True).copy()

    # Rename column
    df_new_label_map.rename(columns={"noisy_label": new_column_name, "cluster_label_confidence": f"{new_column_name}_confidence"}, inplace=True)

    # Merge the original results table with the mapping table.
    df_merged_data = pd.merge(
        left=df_results,
        right=df_new_label_map,
        left_on="cluster_label",
        right_on="cluster_label"
    )

    return df_merged_data


def compute_performance_scores(df_results: pd.DataFrame):
    df_results = assign_labels_based_on_clusters(df_results=df_results)
    return dict(
        accuracy_score = accuracy_score(
            y_true=df_results["true_label"],
            y_pred=df_results["pred_label"]
        ),
        precision_score = precision_score(
            y_true=df_results["true_label"],
            y_pred=df_results["pred_label"],
            average="macro",
            zero_division=0,
        ),
        recall_score = recall_score(
            y_true=df_results["true_label"],
            y_pred=df_results["pred_label"],
            average="macro",
            zero_division=0,
        ),
        f1_score = f1_score(
            y_true=df_results["true_label"],
            y_pred=df_results["pred_label"],
            average="macro",
            zero_division=0,
        ),
    )
