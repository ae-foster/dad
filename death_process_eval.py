import argparse
import math
import os

import torch
import pyro
import pandas as pd

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from experiment_tools.output_utils import get_mlflow_meta
from experiment_tools.pyro_tools import auto_seed


def evaluate_policy(
    experiment_id,
    run_id=None,
    seed=-1,
    n_rollout=1000,  # number of rollouts
    device="cuda",
):

    pyro.clear_param_store()
    seed = auto_seed(seed)
    if not os.path.exists("mlflow_outputs"):
        os.makedirs("mlflow_outputs")
    if run_id:
        experiment_run_ids = [run_id]
    else:
        filter_string = ""
        meta = get_mlflow_meta(experiment_id=experiment_id, filter_string=filter_string)
        meta = [m for m in meta if "eval_seed" not in m.data.params.keys()]
        experiment_run_ids = [run.info.run_id for run in meta]
        from_source = [
            True if "from_source" in m.data.params.keys() else False for m in meta
        ]
    print(experiment_run_ids)

    for i, run_id in enumerate(experiment_run_ids):
        if from_source[i]:
            ## calculate average metric
            client = MlflowClient()
            metric = client.get_metric_history(run_id, "information_gain")
            igs = [m.value for m in metric]
            n_rollout = len(igs)
            num_experiments = int(client.get_run(run_id).data.params["num_experiments"])
            information_gain = torch.tensor(igs)
        else:
            model_location = f"mlruns/{experiment_id}/{run_id}/artifacts/model"
            deathprocess = mlflow.pytorch.load_model(
                model_location, map_location=device
            )
            num_experiments = deathprocess.T
            deathprocess.eval(n_trace=1, theta=torch.tensor(1.5, device=device))

            grid_min, grid_max, grid_n = 0, 20, 1000
            data = deathprocess.rollout(
                n_rollout, torch.linspace(grid_min, grid_max, grid_n, device=device)
            )
            prior_log_prob = data.nodes["theta"]["log_prob"]
            mesh_density = math.exp(-prior_log_prob.logsumexp(0)[0].item())
            posterior_log_prob = sum(
                node["log_prob"]
                for node in data.nodes.values()
                if node["type"] == "sample" and node.get("subtype") != "design_sample"
            )
            posterior_log_prob = (
                posterior_log_prob
                - posterior_log_prob.logsumexp(0)
                - math.log(mesh_density)
            )
            posterior_entropy = (
                mesh_density * posterior_log_prob.exp() * (-posterior_log_prob)
            ).sum(0)
            prior_entropy = (
                mesh_density * prior_log_prob.exp() * (-prior_log_prob)
            ).sum(0)
            information_gain = prior_entropy - posterior_entropy

        print(information_gain.mean(), information_gain.std() / math.sqrt(n_rollout))
        res = pd.DataFrame(
            {
                "EIG_mean": information_gain.mean().item(),
                "EIG_se": (information_gain.std() / math.sqrt(n_rollout)).item(),
            },
            index=[num_experiments],
        )

        res.to_csv("mlflow_outputs/dp_eval.csv")
        with mlflow.start_run(run_id=run_id, experiment_id=experiment_id) as run:
            mlflow.log_param("eval_seed", seed)
            mlflow.log_artifact(
                "mlflow_outputs/dp_eval.csv", artifact_path="evaluation"
            )
            mlflow.log_metric("eval_MI", information_gain.mean().item())

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deep Adaptive Design: Death Process Evaluation."
    )
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--experiment-id", default="9", type=str)
    parser.add_argument("--run-id", default=None, type=str)
    parser.add_argument("--num-rollouts", default=10000, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()

    evaluate_policy(
        seed=args.seed,
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        device=args.device,
        n_rollout=args.num_rollouts,
    )
