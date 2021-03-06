from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go


def plot_2d(df_data: pd.DataFrame, color_attr: str="noisy_label"):
    df_plot_data = df_data.reset_index()
    df_plot_data.sort_values(color_attr, inplace=True)
    df_plot_data[color_attr] = df_plot_data[color_attr].astype(str)
    hover_data = {
        "feat_0": False, "feat_1": False,
        "true_label": False, "noisy_label": False,
        "True Label ": df_plot_data["true_label"],
        "Noisy Label ": df_plot_data["noisy_label"],
    }
    if "cluster_label" in df_plot_data.columns:
        hover_data["Cluster Label "] = df_plot_data["cluster_label"]
    if "pred_label" in df_plot_data.columns:
        hover_data["Predicted Label "] = df_plot_data["pred_label"]
    fig = px.scatter(
        df_plot_data,
        x="feat_0",
        y="feat_1",
        color=color_attr,
        opacity=0.5,
        color_discrete_sequence=px.colors.qualitative.G10,
        hover_name="index",
        hover_data=hover_data,
    )
    fig.update_layout(
        legend=dict(
            title=color_attr,
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        autosize=False,
        width=800,
        height=600,
        margin=dict(l=5, r=5, t=5, b=5),
    )
    fig.update_xaxes(showticklabels=True, title="")
    fig.update_yaxes(showticklabels=True, title="")
    fig.show()


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
