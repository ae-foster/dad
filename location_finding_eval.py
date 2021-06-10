import os
import math
import pickle
from tqdm import tqdm
import argparse
from collections import defaultdict

import pandas as pd

import torch
import pyro

import mlflow
import mlflow.pytorch

from experiment_tools.pyro_tools import auto_seed
from experiment_tools.output_utils import get_mlflow_meta
from contrastive.mi import PriorContrastiveEstimation, NestedMonteCarloEstimation
from neural.modules import LazyFn
from location_finding import HiddenObjects


def make_data_source(experiment_id, run_id, T, device="cuda", n=1):
    fname = f"mlruns/{experiment_id}/{run_id}/artifacts/hostories/results_vi.pickle"
    with open(fname, "rb") as f:
        data = pickle.load(f)

    sample = defaultdict(list)
    latent_name = "theta"
    for history in data["loop"]:
        sample[latent_name].append(history["theta"])

        for i in range(T):
            sample[f"y{i+1}"].append(history[f"y{i+1}"])
            sample[f"xi{i+1}"].append(history[f"xi{i+1}"])

        if len(sample[latent_name]) == n:
            record = {k: torch.stack(v, 0).to(device) for k, v in sample.items()}
            yield record
            sample = defaultdict(list)


def get_data_source_meta(experiment_id, run_id):
    meta = get_mlflow_meta(experiment_id=experiment_id)
    meta = [m for m in meta if run_id == m.info.run_id][0]
    fname = f"mlruns/{experiment_id}/{run_id}/artifacts/hostories/results_vi.pickle"
    with open(fname, "rb") as f:
        data = pickle.load(f)
    out = {
        "n_rollout": len(data["loop"]),
        "noise_scale": float(meta.data.params["noise_scale"]),
        "p": int(meta.data.params["p"]),
        "K": int(meta.data.params["num_sources"]),
        "num_experiments": int(meta.data.params["num_experiments"]),
    }
    return out


def evaluate_run(
    experiment_id,
    run_id,
    num_experiments_to_perform,
    num_inner_samples,
    device,
    n_rollout,
    from_source=False,
    seed=-1,
    theta_prior_loc=None,
    theta_prior_covmat=None,
):
    pyro.clear_param_store()
    model_location = f"mlruns/{experiment_id}/{run_id}/artifacts/model"
    seed = auto_seed(seed)
    factor = 16
    n_rollout = n_rollout // factor

    EIGs_mean = pd.DataFrame(columns=["lower", "upper"])
    EIGs_se = pd.DataFrame(columns=["lower", "upper"])

    data_source = None
    for t_exp in num_experiments_to_perform:
        if from_source:
            meta = get_data_source_meta(experiment_id, run_id)
            if t_exp is None:
                t_exp = meta["num_experiments"]
            data_source = make_data_source(
                experiment_id, run_id, T=t_exp, device=device, n=1
            )
            design_dim = (1, meta["p"])
            ho_model = HiddenObjects(
                design_net=LazyFn(
                    lambda *args: None, prototype=torch.ones(design_dim, device=device),
                ),
                theta_loc=torch.zeros((meta["K"], meta["p"]), device=device),
                theta_covmat=torch.eye(meta["p"], device=device),
                noise_scale=meta["noise_scale"] * torch.ones(1, device=device),
                p=meta["p"],
                K=meta["K"],
                T=t_exp,  # run_results["num_experiments"],
            )
            factor = 1
            n_rollout = meta["n_rollout"]
        else:
            # load model, set number of experiments
            ho_model = mlflow.pytorch.load_model(model_location, map_location=device)
            if t_exp:
                ho_model.T = t_exp
            else:
                t_exp = ho_model.T

        pce_loss_upper = NestedMonteCarloEstimation(
            factor, num_inner_samples, data_source=data_source
        )

        EIG_proxy_upper = torch.zeros(n_rollout)
        EIG_proxy_lower = torch.zeros(n_rollout)

        auto_seed(seed)
        EIG_proxy_upper = torch.tensor(
            [-pce_loss_upper.loss(ho_model.model) for _ in range(n_rollout)]
        )

        if from_source:
            # make a new generator
            data_source = make_data_source(
                experiment_id, run_id, T=t_exp, device=device, n=1
            )
        pce_loss_lower = PriorContrastiveEstimation(
            factor, num_inner_samples, data_source=data_source
        )
        auto_seed(seed)
        EIG_proxy_lower = torch.tensor(
            [-pce_loss_lower.loss(ho_model.model) for _ in range(n_rollout)]
        )

        EIGs_mean.loc[t_exp, "lower"] = EIG_proxy_lower.mean().item()
        EIGs_mean.loc[t_exp, "upper"] = EIG_proxy_upper.mean().item()
        EIGs_se.loc[t_exp, "lower"] = EIG_proxy_lower.std().item() / math.sqrt(
            n_rollout
        )
        EIGs_se.loc[t_exp, "upper"] = EIG_proxy_upper.std().item() / math.sqrt(
            n_rollout
        )

    EIGs_mean["stat"] = "mean"
    EIGs_se["stat"] = "se"
    res = pd.concat([EIGs_mean, EIGs_se])
    print("\n")
    print(res)
    if not os.path.exists("mlflow_outputs"):
        os.makedirs("mlflow_outputs")
    res.to_csv("mlflow_outputs/eval.csv")

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id) as run:
        mlflow.log_param("n_rollouts", n_rollout * factor)
        mlflow.log_param("eval_seed", seed)
        mlflow.log_artifact("mlflow_outputs/eval.csv", artifact_path="evaluation")
        if len(num_experiments_to_perform) == 1:
            mlflow.log_metric("eval_MI_lower", EIGs_mean.loc[t_exp, "lower"])
            mlflow.log_metric("eval_MI_upper", EIGs_mean.loc[t_exp, "upper"])
    return res


def evaluate_experiment(
    experiment_id,
    num_experiments_to_perform=[None],
    num_inner_samples=int(5e5),
    device="cpu",
    n_rollout=2048,
    seed=-1,
):
    # filter_string = (
    #     f"params.num_experiments='{30}' "
    #     f"params.adam_beta1='{0.8}' "
    #     f"params.gamma='{0.8}'"
    # )
    filter_string = ""
    meta = get_mlflow_meta(experiment_id=experiment_id, filter_string=filter_string)
    # run those that haven't yet been evaluated
    meta = [m for m in meta if "eval_seed" not in m.data.params.keys()]
    from_source = [
        True if "from_source" in m.data.params.keys() else False for m in meta
    ]
    experiment_run_ids = [run.info.run_id for run in meta]
    print(experiment_run_ids)
    for i, run_id in enumerate(experiment_run_ids):
        print(f"Evaluating run {i+1} out of {len(experiment_run_ids)} runs")
        evaluate_run(
            experiment_id=experiment_id,
            run_id=run_id,
            num_experiments_to_perform=num_experiments_to_perform,
            num_inner_samples=num_inner_samples,
            device=device,
            n_rollout=n_rollout,
            seed=-1,
            from_source=from_source[i],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deep Adaptive Design example: Hidden Object Detection."
    )
    parser.add_argument("--experiment-id", default="28", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--n-rollout", default=2048, type=int)
    parser.add_argument("--num-inner-samples", default=int(5e5), type=int)
    parser.add_argument("--num-experiments-to-perform", nargs="+", default=[None])

    args = parser.parse_args()
    args.num_experiments_to_perform = [
        int(x) if x else x for x in args.num_experiments_to_perform
    ]
    evaluate_experiment(
        experiment_id=args.experiment_id,
        n_rollout=args.n_rollout,
        seed=args.seed,
        num_inner_samples=args.num_inner_samples,
        num_experiments_to_perform=args.num_experiments_to_perform,
        device=args.device,
    )
