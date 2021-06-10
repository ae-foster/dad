import os
import pickle
import argparse
import time

import warnings
from pyro.infer.trace_elbo import _compute_log_r, is_identically_zero
from pyro.util import torch_isnan

import torch
from torch import nn
from pyro.infer.util import torch_item
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from tqdm import trange
import mlflow

from neural.modules import LazyFn, BatchDesignBaseline
from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from contrastive.mi import PriorContrastiveEstimation

from location_finding import HiddenObjects


def fit_posterior_vi(
    xis,
    ys,
    prior_loc=None,
    prior_covmat=None,
    noise_scale=0.5,
    num_sources=2,
    p=2,
    num_steps=2500,
    initial_lr=0.1,
    momentum=0.1,
    gamma=0.99,
    device="cuda",
):
    """fit a variational posterior given design-obs pairs"""
    assert len(xis) == len(ys)
    data_dict = {}
    for i, xi in enumerate(xis):
        data_dict[f"xi{i + 1}"] = xi
    for i, y in enumerate(ys):
        data_dict[f"y{i + 1}"] = y

    model = pyro.condition(
        HiddenObjects(
            design_net=LazyFn(
                lambda *args, **kwargs: None,
                prototype=torch.ones(1, p, device=device),
            ),
            base_signal=0.1,
            max_signal=1e-4,
            noise_scale=noise_scale * torch.tensor(1.0, device=device),
            theta_loc=prior_loc,  # prior on theta mean hyperparam
            theta_covmat=prior_covmat,  # prior on theta covariance hyperparam
            p=p,
            K=num_sources,
            T=len(xis),
        ).model,
        data=data_dict,
    )

    guide = AutoDiagonalNormal(model)
    optimizer = torch.optim.SGD
    scheduler = pyro.optim.ExponentialLR(
        {
            "optimizer": optimizer,
            "optim_args": {
                "lr": initial_lr,
                "momentum": momentum,
            },
            "gamma": gamma,
        }
    )
    svi = pyro.infer.SVI(
        model=model, guide=guide, optim=scheduler, loss=pyro.infer.Trace_ELBO()
    )

    t = range(1, num_steps + 1)
    for i in t:
        loss = svi.step()
        if torch_isnan(loss):
            breakpoint()
        loss = torch_item(loss)
        # t.set_description("VI Loss: {:.3f} ".format(loss))
        if i % 1000 == 0:
            scheduler.step()

    return_loc = guide.get_posterior().mean.detach().clone()
    # return_loc = pyro.param("guide_loc").detach().clone()
    return_scale = guide.get_posterior().stddev.detach().clone() + 1e-3
    # return_covmat = None
    # return_scale = pyro.param("guide_scale").detach().clone() + 1e-3
    return return_loc, return_scale


def optimise_design(
    posterior_loc,
    posterior_scale,
    noise_scale,
    p=2,
    num_sources=2,
    device="cuda",
    num_outer_samples=100,
    num_inner_samples=100,
    num_steps=2500,
    initial_lr=0.1,
    momentum=0.1,
    gamma=0.99,
):
    design_net = BatchDesignBaseline(1, (1, p)).to(device)
    covmat = torch.cat(
        [torch.diag(x).unsqueeze(0) for x in (posterior_scale ** 2).reshape(2, 2)]
    )
    ho_model = HiddenObjects(
        design_net=design_net,
        base_signal=0.1,
        max_signal=1e-4,
        theta_loc=posterior_loc.reshape(2, 2),
        theta_covmat=covmat,
        T=1,
        p=p,
        K=num_sources,
        noise_scale=noise_scale * torch.tensor(1.0, device=device),
    )
    optimizer = torch.optim.SGD
    scheduler = pyro.optim.ExponentialLR(
        {
            "optimizer": optimizer,
            "optim_args": {"lr": initial_lr, "momentum": momentum},
            "gamma": gamma,
        }
    )
    pce_loss = PriorContrastiveEstimation(num_outer_samples, num_inner_samples)
    oed = OED(ho_model.model, scheduler, pce_loss)

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


def main(
    seed,
    device,
    mlflow_experiment_name,
    num_loop,
    T=30,
    noise_scale=0.5,
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
    mlflow.log_param("p", 2)
    mlflow.log_param("num_sources", 2)
    mlflow.log_param("num_loop", num_loop)
    mlflow.log_param("num_experiments", T)

    mlflow.log_param("base_signal", 0.1)
    mlflow.log_param("max_signal", 1e-4)
    mlflow.log_param("noise_scale", noise_scale)
    # VI params
    mlflow.log_param("vi_num_steps", vi_num_steps)
    mlflow.log_param("vi_lr", vi_lr)
    mlflow.log_param("vi_gamma", vi_gamma)
    mlflow.log_param("vi_momentum", vi_momentum)
    # oed params
    mlflow.log_param("num_steps", oed_num_steps)
    mlflow.log_param("lr", oed_lr)
    mlflow.log_param("gamma", oed_gamma)
    mlflow.log_param("momentum", oed_momentum)
    mlflow.log_param("num_inner_samples", oed_num_inner_samples)
    mlflow.log_param("num_outer_samples", oed_num_outer_samples)

    results_vi = {"loop": [], "seed": seed}
    timings = []
    for i in range(num_loop):
        tic = time.time()
        results = main_loop(
            i,
            device=device,
            T=T,
            noise_scale=noise_scale,
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
        toc = time.time()
        print(toc - tic)
        timings.append([toc - tic])
        results_vi["loop"].append(results)
        print(f"This was run {i+1} out of {num_loop+1}")

    print(timings)
    # Log the results dict as an artifact
    if not os.path.exists("./mlflow_outputs"):
        os.makedirs("./mlflow_outputs")
    with open("./mlflow_outputs/results_vi.pickle", "wb") as f:
        pickle.dump(results_vi, f)

    mlflow.log_artifact("mlflow_outputs/results_vi.pickle", artifact_path="hostories")
    mlflow.log_param("from_source", True)
    print(f"Done.")
    # --------------------------------------------------------------------------


def main_loop(
    run,
    device,
    T=30,
    noise_scale=0.5,
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
    num_sources=2,
    p=2,
):
    pyro.clear_param_store()

    theta_loc = torch.zeros((num_sources, p), device=device)
    theta_covmat = torch.eye(p, device=device)
    prior = torch.distributions.MultivariateNormal(theta_loc, theta_covmat)
    ## Sample a parameter
    true_theta = prior.sample()

    designs_so_far = []
    observations_so_far = []

    # Initialize the posterior randomly
    posterior_loc = prior.sample()  # check if needs to be reshaped.
    posterior_scale = theta_covmat.clone()

    # Use a hard-coded first design
    print(f"Step 1/{T} of Run {run + 1}")
    design0 = theta_loc[[0]] + 0.01  # this is the true optimal
    true_model_to_sample = pyro.condition(
        HiddenObjects(
            design_net=LazyFn(
                lambda *args, **kwargs: design0,
                prototype=torch.ones(1, p, device=device),
            ),
            base_signal=0.1,
            max_signal=1e-4,
            T=1,
            noise_scale=noise_scale * torch.tensor(1.0, device=device),
            p=p,
            K=num_sources,
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
            xis=designs_so_far,
            ys=observations_so_far,
            prior_loc=theta_loc,
            prior_covmat=theta_covmat,
            noise_scale=noise_scale,
            num_sources=2,
            p=2,
            num_steps=vi_num_steps,
            initial_lr=vi_lr,
            gamma=vi_gamma,
            momentum=vi_momentum,
            device=device,
        )
        posterior_loc = new_loc.detach().clone()
        posterior_scale = new_scale.detach().clone()

        design = optimise_design(
            posterior_loc,
            posterior_scale,
            noise_scale=noise_scale,
            device=device,
            num_steps=oed_num_steps,
            initial_lr=oed_lr,
            gamma=oed_gamma,
            momentum=oed_momentum,
            num_inner_samples=oed_num_inner_samples,
            num_outer_samples=oed_num_outer_samples,
        )

        true_model_to_sample = pyro.condition(
            HiddenObjects(
                design_net=LazyFn(
                    lambda *args, **kwargs: design,
                    prototype=torch.ones(1, p, device=device),
                ),
                T=1,
                base_signal=0.1,
                max_signal=1e-4,
                noise_scale=noise_scale * torch.tensor(1.0, device=device),
                p=p,
                K=num_sources,
            ).model,
            data={"theta": true_theta},
        )
        outcome = true_model_to_sample()[0]
        designs_so_far.append(design)
        observations_so_far.append(outcome)

    data_dict = {}
    for i, xi in enumerate(designs_so_far):
        data_dict[f"xi{i + 1}"] = xi.cpu()
    for i, y in enumerate(observations_so_far):
        data_dict[f"y{i + 1}"] = y.cpu()
    data_dict["theta"] = true_theta.cpu()

    print("Done.")
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Location Finding Example: Variational baseline"
    )
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-loop", default=3, type=int)
    parser.add_argument("--vi-num-steps", default=5000, type=int)
    parser.add_argument("--vi-lr", default=1e-3, type=float)
    parser.add_argument("--vi-gamma", default=0.9, type=float)
    parser.add_argument("--vi-momentum", default=0.1, type=float)
    parser.add_argument("--oed-num-steps", default=5000, type=int)
    parser.add_argument("--oed-lr", default=1e-3, type=float)
    parser.add_argument("--oed-gamma", default=0.9, type=float)
    parser.add_argument("--oed-momentum", default=0.1, type=float)
    parser.add_argument("--oed-num-inner-samples", default=1024, type=int)
    parser.add_argument("--oed-num-outer-samples", default=1024, type=int)
    parser.add_argument("--num-experiments", default=5, type=int)  # == T
    parser.add_argument("--noise_scale", default=0.5, type=int)  # == N
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--mlflow-experiment-name", default="location_finding", type=str
    )
    args = parser.parse_args()
    if not os.path.exists("./mlflow_outputs_vi"):
        os.makedirs("./mlflow_outputs_vi")

    main(
        seed=args.seed,
        num_loop=args.num_loop,
        device=args.device,
        T=args.num_experiments,
        noise_scale=args.noise_scale,
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
