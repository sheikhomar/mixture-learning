import json

from datetime import datetime
from pathlib import Path

import pandas as pd

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

import utils


def extract_performance_info(experiment_dir: Path):
    df_results = pd.read_feather(experiment_dir / "clustering-results.feather")
    df_results = utils.assign_labels_based_on_clusters(df_results=df_results)
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


def extract_experiment_result(job_path: Path):
    with open(job_path, "r") as f:
        job_info = json.load(f)
        
    experiment_dir = job_path.parent

    with open(experiment_dir / "experiment-params.json", "r") as f:
        experiment_params = json.load(f)
    
    started_at = datetime.fromisoformat(job_info["started_at"])
    completed_at = datetime.fromisoformat(job_info["completed_at"])
    duration_secs = (completed_at - started_at).total_seconds()
    
    result_dict = dict(
        experiment_type = job_info["command_params"]["experiment-type"],
        experiment_name = job_info["command_params"]["experiment-name"],
        **extract_performance_info(experiment_dir),
        **experiment_params,
        started_at = started_at,
        completed_at = completed_at,
        duration_secs = duration_secs,
        job_path = str(job_path),
    )
    return result_dict


def get_result_summary(job_paths):
    raw_data = [extract_experiment_result(jp) for jp in job_paths]
    df_result_summary = pd.DataFrame(raw_data)
    return df_result_summary
