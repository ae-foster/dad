import os
import pickle
import argparse
import math

import warnings
from pyro.infer.trace_elbo import _compute_log_r, is_identically_zero
from pyro.util import torch_isnan
from pyro.poutine.util import prune_subsample_sites
from pyro.contrib.util import lexpand

import torch
from torch import nn
from pyro.infer.util import torch_item
import pyro
import pyro.distributions as dist
from tqdm import trange
import mlflow

from neural.modules import LazyFn, BatchDesignBaseline
from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from contrastive.mi import PriorContrastiveEstimationScoreGradient

from death_process import DeathProcess
from extra_distributions.truncated_normal import LowerTruncatedNormal


def fit_posterior_vi(
    prior_loc,
    prior_scale,
    N,
    xis,
    ys,
    num_steps=2500,
    initial_lr=0.1,
    momentum=0.1,
    gamma=0.99,
):
    assert len(xis) == len(ys)
    data_dict = {}
    for i, xi in enumerate(xis):
        data_dict[f"xi{i + 1}"] = xi
    for i, y in enumerate(ys):
        data_dict[f"y{i + 1}"] = y
    model = pyro.condition(
        DeathProcess(
            LazyFn(
                lambda *args, **kwargs: None,
                torch.tensor([1.0], device=prior_loc.device),
            ),
            theta_loc=prior_loc,
            theta_scale=prior_scale,
            T=len(xis),
            N=N,
        ).model,
        data=data_dict,
    )

    def guide():
        guide_loc = torch.nn.functional.softplus(
            pyro.param("guide_loc", prior_loc.clone())
        )
        guide_scale = (
            torch.nn.functional.softplus(pyro.param("guide_scale", prior_scale.clone()))
            + 1e-3
        )
        pyro.sample(
            "theta",
            LowerTruncatedNormal(
                guide_loc, guide_scale, torch.tensor(0.0, device=guide_loc.device)
            ),
        )

    optimizer = torch.optim.SGD
    scheduler = pyro.optim.ExponentialLR(
        {
            "optimizer": optimizer,
            "optim_args": {"lr": initial_lr, "momentum": momentum},
            "gamma": gamma,
        }
    )
    svi = pyro.infer.SVI(
        model=model, guide=guide, optim=scheduler, loss=pyro.infer.Trace_ELBO()
    )

    t = trange(1, num_steps + 1, desc="VI Loss: 0.000 ")
    for i in t:
        loss = svi.step()
        if torch_isnan(loss):
            breakpoint()
        loss = torch_item(loss)
        t.set_description("VI Loss: {:.3f} ".format(loss))
        if i % 1000 == 0:
            scheduler.step()

    return_loc = torch.nn.functional.softplus(pyro.param("guide_loc").detach().clone())
    return_scale = (
        torch.nn.functional.softplus(pyro.param("guide_scale").detach().clone()) + 1e-3
    )
    return return_loc, return_scale


def optimise_design(
    posterior_loc,
    posterior_scale,
    N,
    device,
    num_outer_samples=100,
    num_inner_samples=100,
    num_steps=2500,
    initial_lr=0.1,
    momentum=0.1,
    gamma=0.99,
):
    design_net = BatchDesignBaseline(1, 1).to(device)
    death_process = DeathProcess(
        design_net=design_net,
        theta_loc=posterior_loc,
        theta_scale=posterior_scale,
        T=1,
        N=N,
    )
    optimizer = torch.optim.SGD
    scheduler = pyro.optim.ExponentialLR(
        {
            "optimizer": optimizer,
            "optim_args": {"lr": initial_lr, "momentum": momentum},
            "gamma": gamma,
        }
    )
    mlflow.log_param("special_lr", "annealed")
    pce_loss = PriorContrastiveEstimationScoreGradient(
        num_outer_samples, num_inner_samples
    )
    oed = OED(death_process.model, scheduler, pce_loss)

    loss_history = []
    t = trange(1, num_steps + 1, desc="OED Loss: 0.000 ")
    for i in t:
        loss = oed.step()
        if torch_isnan(loss):
            breakpoint()
        loss = torch_item(loss)
        t.set_description("OED Loss: {:.3f} ".format(loss))
        loss_history.append(loss)
        if i % 1000 == 0:
            scheduler.step()

    # Note: do not run softplus, this is done automatically by model
    chosen_design = design_net().detach().clone()
    return chosen_design


# This method is for evaluation
def compute_posterior(rollout, device, T, N):
    theta_prior_loc = torch.tensor(1.0, device=device)
    theta_prior_scale = torch.tensor(1.0, device=device)
    deathprocess = DeathProcess(
        design_net=LazyFn(lambda *args: None, torch.tensor(0.0, device=device)),
        theta_loc=theta_prior_loc,
        theta_scale=theta_prior_scale,
        T=T,
        N=N,
    )

    grid_min, grid_max, grid_n = 0, 20, 5000
    grid = torch.linspace(grid_min, grid_max, grid_n, device=device)
    rollout = {
        name: lexpand(torch.tensor(value, device=device), grid_n)
        for name, value in rollout.items()
    }
    rollout["theta"] = grid

    def conditional_model():
        with pyro.plate_stack("vectorization", (grid_n,)):
            pyro.condition(deathprocess.model, data=rollout)()

    condition_trace = pyro.poutine.trace(conditional_model).get_trace()
    condition_trace = prune_subsample_sites(condition_trace)
    condition_trace.compute_log_prob()

    prior_log_prob = condition_trace.nodes["theta"]["log_prob"]
    mesh_density = math.exp(-prior_log_prob.logsumexp(0).item())
    posterior_log_prob = sum(
        node["log_prob"]
        for node in condition_trace.nodes.values()
        if node["type"] == "sample" and node.get("subtype") != "design_sample"
    )
    posterior_log_prob = (
        posterior_log_prob - posterior_log_prob.logsumexp(0) - math.log(mesh_density)
    )

    return grid, posterior_log_prob, mesh_density, prior_log_prob


def main(
    seed,
    device,
    mlflow_experiment_name,
    num_loop,
    T=4,
    N=50,
    vi_num_steps=3000,
    vi_lr=1e-3,
    vi_gamma=0.9,
    vi_momentum=0.1,
    oed_num_steps=5000,
    oed_lr=1e-3,
    oed_gamma=0.9,
    oed_momentum=0.1,
    oed_num_inner_samples=50,
    oed_num_outer_samples=50,
):
    pyro.clear_param_store()
    seed = auto_seed(seed)
    pyro.set_rng_seed(seed)

    mlflow.set_experiment(mlflow_experiment_name)
    # Log everything
    mlflow.log_param("seed", seed)
    mlflow.log_param("num_loop", num_loop)
    mlflow.log_param("num_experiments", T)
    mlflow.log_param("num_people", N)
    mlflow.log_param("vi_num_steps", vi_num_steps)
    mlflow.log_param("vi_lr", vi_lr)
    mlflow.log_param("vi_gamma", vi_gamma)
    mlflow.log_param("vi_momentum", vi_momentum)
    mlflow.log_param("oed_num_steps", oed_num_steps)
    mlflow.log_param("oed_lr", oed_lr)
    mlflow.log_param("oed_gamma", oed_gamma)
    mlflow.log_param("oed_momentum", oed_momentum)
    mlflow.log_param("oed_num_inner_samples", oed_num_inner_samples)
    mlflow.log_param("oed_num_outer_samples", oed_num_outer_samples)
    mlflow.log_param("from_source", True)

    results_vi = {"loop": [], "seed": seed}
    for i in range(num_loop):
        results = main_loop(
            i,
            device,
            T=T,
            N=N,
            vi_num_steps=vi_num_steps,
            vi_lr=vi_lr,
            vi_gamma=vi_gamma,
            vi_momentum=vi_momentum,
            oed_num_steps=oed_num_steps,
            oed_lr=oed_lr,
            oed_gamma=oed_gamma,
            oed_momentum=oed_momentum,
            oed_num_inner_samples=oed_num_inner_samples,
            oed_num_outer_samples=oed_num_outer_samples,
        )
        results_vi["loop"].append(results)

    # Log the results dict as an artifact
    if not os.path.exists("./mlflow_outputs"):
        os.makedirs("./mlflow_outputs")
    with open("./mlflow_outputs/results_vi.pickle", "wb") as f:
        # this will be overwritten every time, data from different runs is in
        # mlruns/id/<hash>/artifacts/results.pickle.
        pickle.dump(results_vi, f)
    # torch.save(results, "./mlflow_outputs/results.pickle")
    mlflow.log_artifacts("mlflow_outputs")
    print("Done.")
    # --------------------------------------------------------------------------


def main_loop(
    run,
    device,
    T=4,
    N=50,
    vi_num_steps=3000,
    vi_lr=1e-3,
    vi_gamma=0.9,
    vi_momentum=0.1,
    oed_num_steps=5000,
    oed_lr=1e-3,
    oed_gamma=0.9,
    oed_momentum=0.1,
    oed_num_inner_samples=50,
    oed_num_outer_samples=50,
):
    pyro.clear_param_store()
    prior_loc = torch.tensor(1.0, device=device)
    prior_scale = torch.tensor(1.0, device=device)
    true_theta = LowerTruncatedNormal(prior_loc, prior_scale, 0.0).sample()

    designs_so_far = []
    observations_so_far = []

    # Initialize the posterior at the prior
    posterior_loc = prior_loc.clone()
    posterior_scale = prior_scale.clone()

    # Use a hard-coded first design
    print(f"Step 1/{T} of Run {run + 1}")
    # Use a precomputed value for the first design to save time
    design0 = torch.tensor([0.9455], device=device)
    true_model_to_sample = pyro.condition(
        DeathProcess(
            LazyFn(lambda *args, **kwargs: design0, torch.tensor([1.0], device=device)),
            theta_loc=prior_loc,
            theta_scale=prior_scale,
            T=1,
            N=N,
        ).model,
        data={"theta": true_theta},
    )
    outcome = true_model_to_sample()[0]
    designs_so_far.append(design0)
    observations_so_far.append(outcome)

    for t in range(1, T):
        print(f"Step {t + 1}/{T} of Run {run + 1}")
        pyro.clear_param_store()

        new_loc, new_scale = fit_posterior_vi(
            prior_loc,
            prior_scale,
            N,
            designs_so_far,
            observations_so_far,
            num_steps=vi_num_steps,
            initial_lr=vi_lr,
            gamma=vi_gamma,
            momentum=vi_momentum,
        )
        posterior_loc = new_loc.detach().clone()
        posterior_scale = new_scale.detach().clone()
        print("Fitted posterior", posterior_loc, posterior_scale)

        design = optimise_design(
            posterior_loc,
            posterior_scale,
            N,
            device,
            num_steps=oed_num_steps,
            initial_lr=oed_lr,
            gamma=oed_gamma,
            momentum=oed_momentum,
            num_inner_samples=oed_num_inner_samples,
            num_outer_samples=oed_num_outer_samples,
        )
        print("design", design)

        true_model_to_sample = pyro.condition(
            DeathProcess(
                LazyFn(
                    lambda *args, **kwargs: design, torch.tensor([1.0], device=device)
                ),
                theta_loc=prior_loc,
                theta_scale=prior_scale,
                T=1,
                N=N,
            ).model,
            data={"theta": true_theta},
        )
        outcome = true_model_to_sample()[0]
        print("Response", outcome)
        designs_so_far.append(design)
        observations_so_far.append(outcome)

    data_dict = {}
    for i, xi in enumerate(designs_so_far):
        data_dict[f"xi{i + 1}"] = xi
    for i, y in enumerate(observations_so_far):
        data_dict[f"y{i + 1}"] = y.item()
    grid, posterior_log_prob, mesh_density, prior_log_prob = compute_posterior(
        data_dict, device, T, N
    )
    posterior_entropy = (
        mesh_density * posterior_log_prob.exp() * (-posterior_log_prob)
    ).sum(0)
    prior_entropy = (mesh_density * prior_log_prob.exp() * (-prior_log_prob)).sum(0)
    information_gain = prior_entropy - posterior_entropy
    print("IG:", information_gain)
    mlflow.log_metric("information_gain", information_gain.item())

    results = {
        "designs": [d.detach().cpu() for d in designs_so_far],
        "observations": [y.detach().cpu() for y in observations_so_far],
        "information_gain": information_gain.detach().cpu(),
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VI baseline: Death Process.")
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-loop", default=50, type=int)
    parser.add_argument("--vi-num-steps", default=5000, type=int)
    parser.add_argument("--vi-lr", default=1e-3, type=float)
    parser.add_argument("--vi-gamma", default=0.9, type=float)
    parser.add_argument("--vi-momentum", default=0.1, type=float)
    parser.add_argument("--oed-num-steps", default=5000, type=int)
    parser.add_argument("--oed-lr", default=1e-3, type=float)
    parser.add_argument("--oed-gamma", default=0.9, type=float)
    parser.add_argument("--oed-momentum", default=0.1, type=float)
    parser.add_argument("--oed-num-inner-samples", default=50, type=int)
    parser.add_argument("--oed-num-outer-samples", default=50, type=int)
    parser.add_argument("--num-experiments", default=4, type=int)  # == T
    parser.add_argument("--num-people", default=50, type=int)  # == N
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--mlflow-experiment-name", default="TEST", type=str)
    args = parser.parse_args()

    main(
        seed=args.seed,
        num_loop=args.num_loop,
        device=args.device,
        T=args.num_experiments,
        N=args.num_people,
        vi_num_steps=args.vi_num_steps,
        vi_lr=args.vi_lr,
        vi_gamma=args.vi_gamma,
        vi_momentum=args.vi_momentum,
        oed_num_steps=args.oed_num_steps,
        oed_lr=args.oed_lr,
        oed_gamma=args.oed_gamma,
        oed_momentum=args.oed_momentum,
        oed_num_inner_samples=args.oed_num_inner_samples,
        oed_num_outer_samples=args.oed_num_outer_samples,
        mlflow_experiment_name=args.mlflow_experiment_name,
    )
