from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go


def plot_2d(df_data: pd.DataFrame):
    df_plot_data = df_data.reset_index()
    df_plot_data["noisy_label"] = df_plot_data["noisy_label"].astype(str)
    df_plot_data.sort_values("noisy_label", inplace=True)
    fig = px.scatter(
        df_plot_data,
        x="feat_0",
        y="feat_1",
        color="noisy_label",
        opacity=0.5,
        color_discrete_sequence=px.colors.qualitative.G10,
        hover_name="index",
        hover_data={
            "feat_0": False, "feat_1": False,
            "true_label": False, "noisy_label": False,
            "True Label ": df_plot_data["true_label"],
            "Noisy Label ": df_plot_data["noisy_label"],
        },
    )
    fig.update_layout(
        legend=dict(
            title="Given Label",
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
    # For each cluster, find the class label with the largest number of points.
    df_result_counts = df_results.reset_index().groupby(["cluster_label", "noisy_label"])[["index"]].count().reset_index()
    df_result_counts = df_result_counts.rename(columns={"index": "count"})
    idx = df_result_counts.groupby(['cluster_label'])['count'].transform(max) == df_result_counts['count']
    
    # Create mapping table with `cluster_label` => `new_label``
    df_new_label_map = df_result_counts[idx][["cluster_label", "noisy_label"]].reset_index(drop=True).copy()
    df_new_label_map.rename(columns={"noisy_label": new_column_name}, inplace=True)

    # Merge the original results table with the mapping table.
    df_merged_data = pd.merge(
        left=df_results,
        right=df_new_label_map,
        left_on="cluster_label",
        right_on="cluster_label"
    )

    return df_merged_data
