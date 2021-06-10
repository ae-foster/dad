import os
import argparse
import math
import pandas as pd

import torch
import pyro

import mlflow
import mlflow.pytorch

from experiment_tools.pyro_tools import auto_seed
from experiment_tools.output_utils import get_mlflow_meta

from contrastive.mi import PriorContrastiveEstimation, NestedMonteCarloEstimation


def evaluate_policy(
    experiment_id,
    run_id=None,
    device="cpu",
    seed=-1,
    n_rollout=100 * 100,
    num_inner_samples=5000,
    T=None,
):
    """
    If run_id is none, evaluate everything in experiment_id
    """
    pyro.clear_param_store()
    seed = auto_seed(seed)
    factor = 100
    n_rollout = n_rollout // factor
    if not os.path.exists("mlflow_outputs"):
        os.makedirs("mlflow_outputs")

    if run_id:
        experiment_run_ids = [run_id]
    else:
        filter_string = ""
        meta = get_mlflow_meta(experiment_id=experiment_id, filter_string=filter_string)
        meta = [m for m in meta if "eval_seed" not in m.data.params.keys()]
        experiment_run_ids = [run.info.run_id for run in meta]
    print(experiment_run_ids)

    for run_id in experiment_run_ids:
        model_location = f"mlruns/{experiment_id}/{run_id}/artifacts/model"
        temporal_model = mlflow.pytorch.load_model(model_location, map_location=device)

        lower = PriorContrastiveEstimation(factor, num_inner_samples)
        upper = NestedMonteCarloEstimation(factor, num_inner_samples)

        auto_seed(seed)  # to ensure both bounds are calculated with the same rollouts
        lower_bounds = torch.tensor(
            [-lower.loss(temporal_model.model) for _ in range(n_rollout)]
        )
        auto_seed(seed)  # to ensure both bounds are calculated with the same rollouts
        upper_bounds = torch.tensor(
            [-upper.loss(temporal_model.model) for _ in range(n_rollout)]
        )

        res = pd.DataFrame(
            {
                "lower": lower_bounds.mean().item(),
                "lower_se": lower_bounds.std().item() / math.sqrt(n_rollout),
                "upper": upper_bounds.mean().item(),
                "upper_se": upper_bounds.std().item() / math.sqrt(n_rollout),
            },
            index=[f"T={temporal_model.T}"],
        ).T
        res.to_csv("mlflow_outputs/ht_eval.csv")

        with mlflow.start_run(run_id=run_id, experiment_id=experiment_id) as run:
            mlflow.log_param("eval_seed", seed)
            mlflow.log_artifact(
                "mlflow_outputs/ht_eval.csv", artifact_path="evaluation"
            )
            mlflow.log_metric("eval_MI_upper", res.loc["upper"][0])
            mlflow.log_metric("eval_MI_lower", res.loc["lower"][0])

        print("model: ", model_location)
        print("Results:\n", res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deep Adaptive Design example: Hyperbolic Temporal Discounting."
    )
    parser.add_argument("--experiment-id", default="2", type=str)
    parser.add_argument("--run-id", default=None, type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()

    seed = auto_seed(args.seed)
    evaluate_policy(
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        device=args.device,
        seed=args.seed,
    )
