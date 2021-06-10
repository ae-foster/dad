import os
import pickle
import argparse

import numpy as np
import pandas as pd

import torch
from torch import nn
from pyro.infer.util import torch_item
from pyro.contrib.util import lexpand, rexpand
from pyro.poutine.util import prune_subsample_sites
import pyro
import pyro.distributions as dist
from tqdm import trange

import mlflow
import mlflow.pytorch

from neural.modules import (
    SetEquivariantDesignNetwork,
    BatchDesignBaseline,
)

from oed.primitives import observation_sample, compute_design
from experiment_tools.pyro_tools import auto_seed

from oed.design import OED
from contrastive.mi import (
    PriorContrastiveEstimationScoreGradient,
    PriorContrastiveEstimationDiscreteObsTotalEnum,
)

from extra_distributions.truncated_normal import LowerTruncatedNormal


class EncoderNetwork(nn.Module):
    def __init__(
        self,
        design_dim,
        osbervation_dim,
        hidden_dim,
        encoding_dim,
        n_hidden_layers=2,
        activation=nn.Softplus,
    ):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.activation_layer = activation()
        self.input_layer = nn.Linear(design_dim + osbervation_dim, hidden_dim)
        if n_hidden_layers > 1:
            self.middle = nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation())
                    for _ in range(n_hidden_layers - 1)
                ]
            )
        else:
            self.middle = nn.Identity()
        self.output_layer = nn.Linear(hidden_dim, encoding_dim)

    def forward(self, xi, y, **kwargs):
        inputs = torch.stack([xi, y], dim=-1)
        x = self.input_layer(inputs)
        x = self.activation_layer(x)
        x = self.middle(x)
        x = self.output_layer(x)
        return x


class EmitterNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_hidden_layers=2,
        activation=nn.Softplus,
    ):
        super().__init__()
        self.activation_layer = activation()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        if n_hidden_layers > 1:
            self.middle = nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation())
                    for _ in range(n_hidden_layers - 1)
                ]
            )
        else:
            self.middle = nn.Identity()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, r):
        x = self.input_layer(r)
        x = self.activation_layer(x)
        x = self.middle(x)
        x = self.output_layer(x)
        return x


class DeathProcess(nn.Module):
    """Model class for Death Process experiment."""

    def __init__(
        self,
        design_net,
        theta_loc=None,  # prior on theta
        theta_scale=None,  # prior on theta
        theta_dist="truncated normal",
        N=50,
        T=2,
    ):
        super().__init__()
        # theta prior hyperparams
        self.design_net = design_net
        self.theta_loc = theta_loc if theta_loc is not None else torch.tensor(1.0)
        self.theta_scale = theta_scale if theta_scale is not None else torch.tensor(1.0)
        if theta_dist == "truncated normal":
            self.theta_prior_dist = LowerTruncatedNormal(
                self.theta_loc, self.theta_scale, 0.0
            )
        elif theta_dist == "lognormal":
            self.theta_prior_dist = dist.LogNormal(self.theta_loc, self.theta_scale)
        else:
            raise ValueError("Invalid option: `theta_dist`=%s." % theta_dist)
        self.T = T  # number of experiments
        self.N = N  # number of people
        self.softplus = nn.Softplus()

    def model(self):
        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)

        ########################################################################
        # Sample latent variables
        ########################################################################
        # Theta has empty shape
        theta = pyro.sample("theta", self.theta_prior_dist)
        theta = theta.clamp(min=1e-10, max=1e10)

        y_outcomes = []
        xi_designs = []

        for t in range(self.T):
            ####################################################################
            # Get a design xi
            ####################################################################
            xi = compute_design(
                f"xi{t + 1}", self.design_net.lazy(*zip(xi_designs, y_outcomes))
            )
            # Have to convert to positive time, and to squeeze out dimension
            xi = self.softplus(xi.squeeze(-1))

            ####################################################################
            # Sample y
            ####################################################################
            death_probability = 1 - (-xi * theta).exp()
            y = observation_sample(
                f"y{t + 1}", dist.Binomial(total_count=self.N, probs=death_probability)
            )
            y_outcomes.append(y)
            xi_designs.append(xi)

        return y_outcomes

    def eval(self, n_trace=2, theta=None):
        self.design_net.eval()
        if theta is not None:
            model = pyro.condition(self.model, data={"theta": theta})
        else:
            model = self.model
        output = []
        with torch.no_grad():
            for i in range(n_trace):

                # print("Example run {}".format(i))
                trace = pyro.poutine.trace(model).get_trace()

                true_theta = trace.nodes["theta"]["value"].item()
                # print(f"--- True Theta: {true_theta} ---")
                run_xis = []
                run_ys = []

                for t in range(self.T):
                    xi = trace.nodes[f"xi{t + 1}"]["value"].item()
                    run_xis.append(xi)
                    # print(f"xi{t + 1}: {xi}")

                    y = trace.nodes[f"y{t + 1}"]["value"].item()
                    run_ys.append(y)
                    # print(f"y{t + 1}: {y}")

                run_df = pd.DataFrame(
                    {
                        "designs": run_xis,
                        "observations": run_ys,
                        "order": list(range(1, self.T + 1)),
                    }
                )
                run_df["run_id"] = i + 1
                run_df["theta"] = true_theta
                output.append(run_df)
                # print("-------- * --------")

        print("returning output")
        return pd.concat(output)

    def rollout(self, n_rollout, grid):
        self.design_net.eval()

        grid_size = grid.shape[0]

        def vectorized_model():
            with pyro.plate("vectorization", n_rollout):
                return self.model()

        with torch.no_grad():
            trace = pyro.poutine.trace(vectorized_model).get_trace()
            trace.nodes["theta"]["value"] = torch.tensor([1.50], device=trace.nodes["theta"]["value"].device)
            trace = prune_subsample_sites(trace)
            trace.compute_log_prob()

            data = {
                name: lexpand(node["value"], grid_size)
                for name, node in trace.nodes.items()
                if node.get("subtype") in ["observation_sample", "design_sample"]
            }
            data["theta"] = rexpand(grid, n_rollout)

            def conditional_model():
                with pyro.plate_stack("vectorization", (grid_size, n_rollout)):
                    pyro.condition(self.model, data=data)()

            condition_trace = pyro.poutine.trace(conditional_model).get_trace()
            condition_trace = prune_subsample_sites(condition_trace)
            condition_trace.compute_log_prob()

        return condition_trace


def single_run(
    seed,
    num_steps,
    num_inner_samples,  # L in denom
    num_outer_samples,  # N to estimate outer E
    lr,  # learning rate of adam optim
    gamma,  # scheduler for adam optim
    T,  # number of experiments
    N,  # number of people
    device,
    hidden_dim,
    encoding_dim,
    num_layers,
    arch,
    complete_enum,
    mlflow_experiment_name,
):

    pyro.clear_param_store()
    seed = auto_seed(seed)
    pyro.set_rng_seed(seed)
    mlflow.set_experiment(mlflow_experiment_name)

    mlflow.log_param("seed", seed)
    mlflow.log_param("num_experiments", T)
    mlflow.log_param("lr", lr)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("encoding_dim", encoding_dim)
    mlflow.log_param("num_layers", num_layers)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("complete_enum", gamma)
    mlflow.log_param("arch", arch)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("num_inner_samples", num_inner_samples)
    mlflow.log_param("num_outer_samples", num_outer_samples)
    ## set up model
    if arch == "static":
        design_net = BatchDesignBaseline(T, 1).to(device)
    else:
        encoder = EncoderNetwork(
            1, 1, hidden_dim, encoding_dim, n_hidden_layers=num_layers
        )
        emitter = EmitterNetwork(
            encoding_dim, hidden_dim, 1, n_hidden_layers=num_layers
        )

        if arch == "sum":
            design_net = SetEquivariantDesignNetwork(
                encoder, emitter, empty_value=torch.ones(1)
            ).to(device)
        else:
            raise ValueError(f"Unexpected architecture specification: '{arch}'.")

    theta_prior_loc = torch.tensor(1.0, device=device)
    theta_prior_scale = torch.tensor(1.0, device=device)
    death_process = DeathProcess(
        design_net=design_net,
        theta_loc=theta_prior_loc,
        theta_scale=theta_prior_scale,
        T=T,
        N=N,
    )

    # Annealed LR optimiser --------
    optimizer = torch.optim.Adam
    scheduler = pyro.optim.ExponentialLR(
        {
            "optimizer": optimizer,
            "optim_args": {"lr": lr, "betas": [0.9, 0.999], "weight_decay": 0,},
            "gamma": gamma,
        }
    )
    if complete_enum:
        pce_loss = PriorContrastiveEstimationDiscreteObsTotalEnum(
            num_outer_samples, num_inner_samples
        )
    else:
        pce_loss = PriorContrastiveEstimationScoreGradient(
            num_outer_samples, num_inner_samples
        )
    oed = OED(death_process.model, scheduler, pce_loss)
    # ----------

    # optimise
    loss_history = []
    t = trange(1, num_steps + 1, desc="Loss: 0.000 ")
    for i in t:
        loss = oed.step()
        loss = torch_item(loss)
        t.set_description("Loss: {:.3f} ".format(loss))
        loss_history.append(loss)
        if i % 50 == 0:
            mlflow.log_metric("loss", oed.evaluate_loss())
        if i % 1000 == 0:
            scheduler.step()

    mlflow.log_metric(
        "loss_diff50", np.mean(loss_history[-51:-1]) / np.mean(loss_history[0:50]) - 1
    )
    # evaluate and store results
    runs_output = death_process.eval()
    results = {
        "design_network": design_net.cpu(),
        "seed": seed,
        "loss_history": loss_history,
        "runs_output": runs_output,
    }

    # Log model
    print("Storing model to MlFlow... ", end="")
    # store the model:
    mlflow.pytorch.log_model(death_process.cpu(), "model")
    ml_info = mlflow.active_run().info
    model_loc = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/model"
    print(f"Model sotred in {model_loc}.")

    print(f"Run completed {mlflow.active_run().info.artifact_uri}.")
    print(f"The experiment-id of this run is {ml_info.experiment_id}")

    # --------------------------------------------------------------------------

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deep Adaptive Design example: Death Process."
    )
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-steps", default=5000, type=int)
    parser.add_argument("--num-inner-samples", default=100, type=int)
    parser.add_argument("--num-outer-samples", default=200, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--num-experiments", default=4, type=int)  # == T
    parser.add_argument("--num-people", default=50, type=int)  # == N
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--hidden-dim", default=128, type=int)
    parser.add_argument("--encoding-dim", default=16, type=int)
    parser.add_argument("--complete-enum", default=False, type=bool)
    parser.add_argument(
        "--num-layers", default=2, type=int, help="Number of hidden layers."
    )
    parser.add_argument(
        "--arch",
        default="sum",
        type=str,
        help="Architecture",
        choices=["static", "sum", "filter"],
    )
    parser.add_argument("--mlflow-experiment-name", default="Default", type=str)
    args = parser.parse_args()

    single_run(
        seed=args.seed,
        num_steps=args.num_steps,
        num_inner_samples=args.num_inner_samples,
        num_outer_samples=args.num_outer_samples,
        lr=args.lr,
        gamma=args.gamma,
        device=args.device,
        T=args.num_experiments,
        N=args.num_people,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim,
        num_layers=args.num_layers,
        arch=args.arch,
        complete_enum=args.complete_enum,
        mlflow_experiment_name=args.mlflow_experiment_name,
    )
