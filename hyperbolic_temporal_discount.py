import os
import pickle
import argparse
import math

import numpy as np
import pandas as pd

import torch
from torch import nn
from pyro.infer.util import torch_item
from pyro.contrib.util import lexpand, rexpand
from torch.distributions.utils import broadcast_all
from pyro.poutine.util import prune_subsample_sites
import pyro
import pyro.distributions as dist
from tqdm import trange
import mlflow
import mlflow.pytorch

from neural.modules import SetEquivariantDesignNetwork, BatchDesignBaseline
from oed.primitives import observation_sample, latent_sample, compute_design
from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from contrastive.mi import (
    PriorContrastiveEstimationDiscreteObsTotalEnum,
    PriorContrastiveEstimationScoreGradient,
)


class EncoderNetwork(nn.Module):
    def __init__(
        self,
        design_dim,
        osbervation_dim,
        hidden_dim,
        encoding_dim,
        include_t,
        T,
        n_hidden_layers=2,
        activation=nn.Softplus,
    ):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.include_t = include_t
        self.T = T
        self.activation_layer = activation()
        if include_t:
            input_dim = design_dim + 1
        else:
            input_dim = design_dim
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
        self.output_layer_0 = nn.Linear(hidden_dim, encoding_dim)
        self.output_layer_1 = nn.Linear(hidden_dim, encoding_dim)

    def forward(self, xi, y, t):
        if self.include_t:
            t = xi.new_tensor(t) / self.T
            x = torch.cat([lexpand(t, *xi.shape[:-1]), xi], axis=-1)
        else:
            x = xi
        x = self.input_layer(x)
        x = self.activation_layer(x)
        x = self.middle(x)
        x_0 = self.output_layer_0(x)
        x_1 = self.output_layer_1(x)
        x = y.unsqueeze(-1) * x_1 + (1.0 - y).unsqueeze(-1) * x_0
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


class HyperbolicTemporalDiscounting(nn.Module):
    """Hyperbolic Temporal Discounting example."""

    def __init__(
        self,
        design_net,
        alpha_prior_distribution,
        epsilon_prior_distribution,
        log_k_loc,
        log_k_scale,
        d_a=0.0,
        r_b=100.0,
        T=2,
    ):
        super().__init__()
        # theta prior hyperparams
        self.design_net = design_net
        self.alpha_prior_distribution = alpha_prior_distribution
        self.epsilon_prior_distribution = epsilon_prior_distribution
        self.log_k_loc = log_k_loc
        self.log_k_scale = log_k_scale
        self.d_a = d_a
        self.r_b = r_b
        self.T = T  # number of experiments
        self.sigmoid = nn.Sigmoid()

    def transform_xi(self, xi, shift=0.0):

        d_b, r_a = xi[..., 0], xi[..., 1]
        # Put this logic inside the design net?
        # Return transformed or untransformed inputs?
        d_b = (d_b - shift).exp()
        # print(d_b.min(), d_b.max())
        r_a = self.r_b * self.sigmoid(r_a)
        return r_a, d_b

    def model(self):
        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)

        ########################################################################
        # Sample latent variables
        ########################################################################
        k = latent_sample("log_k", dist.Normal(self.log_k_loc, self.log_k_scale)).exp()
        # Use this as an offset to help with initialization
        log_k_mean = self.log_k_loc + 0.5 * self.log_k_scale * self.log_k_scale

        alpha = latent_sample("alpha", self.alpha_prior_distribution)
        alpha = 1e-3 + alpha.abs()

        epsilon = latent_sample("epsilon", self.epsilon_prior_distribution)

        y_outcomes = []
        xi_designs = []
        for t in range(self.T):
            ####################################################################
            # Get a design xi
            ####################################################################
            xi = compute_design(
                f"xi{t + 1}", self.design_net.lazy(*zip(xi_designs, y_outcomes)),
            )
            r_a, d_b = self.transform_xi(xi, shift=log_k_mean)

            ####################################################################
            # Sample y
            ####################################################################
            v_a = r_a / (1.0 + k * self.d_a)
            v_b = self.r_b / (1.0 + k * d_b)
            erf_arg = (v_a - v_b) / alpha
            psi = epsilon + (1.0 - 2.0 * epsilon) * (0.5 + 0.5 * torch.erf(erf_arg))

            y = observation_sample(f"y{t + 1}", dist.Bernoulli(probs=psi))
            y_outcomes.append(y)
            xi_designs.append(xi)

        return y_outcomes

    def eval(self, n_trace=3, latents=None, verbose=True):
        ## latents -> dict {latent: value}  to condition model on
        self.design_net.eval()
        if latents:
            model = pyro.condition(self.model, data=latents)
        else:
            model = self.model
        output = []
        # Use the same offset
        log_k_mean = self.log_k_loc + 0.5 * self.log_k_scale * self.log_k_scale
        with torch.no_grad():
            for i in range(n_trace):
                trace = pyro.poutine.trace(model).get_trace()

                true_k = trace.nodes["log_k"]["value"].exp().item()
                true_alpha = (1e-3 + trace.nodes["alpha"]["value"].abs()).item()
                true_epsilon = trace.nodes["epsilon"]["value"].item()
                if verbose:
                    print("Example run {}".format(i))
                    print(f"--- True k: {true_k}")
                    print(f"--- True alpha: {true_alpha}")
                    print(f"--- True epsilon: {true_epsilon}")
                run_r_as = []
                run_d_bs = []
                run_ys = []

                xi_designs = []
                y_outcomes = []

                for t in range(self.T):
                    xi = trace.nodes[f"xi{t + 1}"]["value"]
                    xi_designs.append(xi)
                    r_a, d_b = self.transform_xi(xi, shift=log_k_mean)
                    r_a, d_b = r_a.item(), d_b.item()
                    run_r_as.append(r_a)
                    run_d_bs.append(d_b)
                    v_a = r_a / (1 + true_k * self.d_a)
                    v_b = self.r_b / (1 + true_k * d_b)
                    erf_arg = torch.tensor((v_a - v_b) / true_alpha)
                    psi = true_epsilon + (1 - 2 * true_epsilon) * (
                        0.5 + 0.5 * torch.erf(erf_arg)
                    )
                    psi = psi.item()

                    y = trace.nodes[f"y{t + 1}"]["value"]
                    y_outcomes.append(y)
                    run_ys.append(y.item())
                    if verbose:
                        print(f"xi{t + 1}: r_a = {r_a}, d_b = {d_b}")
                        print(f"v_a = {v_a}, v_b = {v_b}")
                        print(f"psi = {psi}")  # prob of accepting delayed reward
                        print(f"y{t + 1}: {y}")

                run_df = pd.DataFrame(
                    {
                        "r_a": run_r_as,
                        "d_b": run_d_bs,
                        "observations": run_ys,
                        "order": list(range(1, self.T + 1)),
                    }
                )
                run_df["run_id"] = i + 1
                run_df["k"] = true_k
                run_df["alpha"] = true_alpha
                run_df["epsilon"] = true_epsilon
                output.append(run_df)

        # print("returning output")
        return pd.concat(output)


def single_run(
    seed,
    num_steps,
    num_inner_samples,  # L in denom
    num_outer_samples,  # N to estimate outer E
    learn_alpha,  # whether to learn alpha as well as log k
    learn_epsilon,  # whether to learn epsilon also
    lr,  # learning rate of sgd optim
    gamma,  # scheduler for sgd optim
    T,  # number of experiments
    device,
    hidden_dim,
    encoding_dim,
    num_layers,
    arch,
    mlflow_experiment_name,
    complete_enum=False,
    include_t=False,
):

    pyro.clear_param_store()
    seed = auto_seed(seed)
    pyro.set_rng_seed(seed)
    mlflow.set_experiment(mlflow_experiment_name)
    if not os.path.exists("mlflow_outputs"):
        os.makedirs("mlflow_outputs")

    mlflow.log_param("seed", seed)
    mlflow.log_param("num_experiments", T)
    mlflow.log_param("lr", lr)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("encoding_dim", encoding_dim)
    mlflow.log_param("num_layers", num_layers)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("arch", arch)
    mlflow.log_param("learn_alpha", learn_alpha)
    mlflow.log_param("learn_epsilon", learn_epsilon)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("complete_enum", complete_enum)
    mlflow.log_param("num_outer_samples", num_outer_samples)
    mlflow.log_param("num_inner_samples", num_inner_samples)
    mlflow.log_param("include_t", include_t)

    ## set up model
    if arch == "static":
        design_net = BatchDesignBaseline(T, 2).to(device)
    else:
        design_dim = 2
        encoder = EncoderNetwork(
            design_dim=design_dim,
            osbervation_dim=1,
            hidden_dim=hidden_dim,
            encoding_dim=encoding_dim,
            include_t=include_t,
            T=T,
            n_hidden_layers=num_layers,
        )
        emitter = EmitterNetwork(
            input_dim=encoding_dim,
            hidden_dim=hidden_dim,
            output_dim=design_dim,
            n_hidden_layers=num_layers,
        )

        if arch == "sum":
            design_net = SetEquivariantDesignNetwork(
                encoder, emitter, empty_value=torch.ones(design_dim)
            ).to(device)
        else:
            raise ValueError(f"Unexpected architecture specification: '{arch}'.")

    if learn_alpha:
        alpha_prior_scale = torch.tensor(2.0, device=device)
        alpha_prior_distribution = dist.Normal(0.0, alpha_prior_scale)
    else:
        alpha_prior_distribution = dist.Delta(torch.tensor(2.0, device=device))
    if learn_epsilon:
        l, u = torch.tensor(0.01, device=device), torch.tensor(0.2, device=device)
        epsilon_prior_dist = dist.Uniform(l, u)
    else:
        epsilon_prior_dist = dist.Delta(torch.tensor(0.01, device=device))
    log_k_prior_loc = torch.tensor(-4.25, device=device)
    log_k_prior_scale = torch.tensor(1.5, device=device)
    temporal_model = HyperbolicTemporalDiscounting(
        design_net=design_net,
        alpha_prior_distribution=alpha_prior_distribution,
        epsilon_prior_distribution=epsilon_prior_dist,
        log_k_loc=log_k_prior_loc,
        log_k_scale=log_k_prior_scale,
        d_a=0.0,
        r_b=100.0,
        T=T,
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
            num_outer_samples=num_outer_samples, num_inner_samples=num_inner_samples
        )
    else:
        pce_loss = PriorContrastiveEstimationScoreGradient(
            num_outer_samples=num_outer_samples, num_inner_samples=num_inner_samples
        )

    oed = OED(temporal_model.model, scheduler, pce_loss)

    # ----------
    # optimise
    loss_history = []
    t = trange(0, num_steps, desc="Loss: 0.000 ")
    for i in t:
        loss = oed.step()
        loss = torch_item(loss)
        t.set_description("Loss: {:.3f} ".format(loss))
        loss_history.append(loss)
        if i % 100 == 0:
            mlflow.log_metric("loss", oed.evaluate_loss())  # oed.evaluate_loss()
        if i % 1000 == 0:
            scheduler.step()

    mlflow.log_metric(
        "loss_diff50", np.mean(loss_history[-51:-1]) / np.mean(loss_history[0:50]) - 1
    )
    # evaluate and store results
    runs_output = temporal_model.eval()  ###
    results = {
        "design_network": design_net.cpu(),
        "seed": seed,
        "loss_history": loss_history,
        "runs_output": runs_output,
    }
    # log model ----------------------------
    print("Storing model to MlFlow... ", end="")
    # store the model:
    mlflow.pytorch.log_model(temporal_model.cpu(), "model")
    ml_info = mlflow.active_run().info
    model_loc = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/model"
    print(f"Model sotred in {model_loc}.")

    print(f"Run completed {mlflow.active_run().info.artifact_uri}.")
    print(f"The experiment-id of this run is {ml_info.experiment_id}")
    # --------------------------------------------------------------------------
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deep Adaptive Design example: Hyperbolic Temporal Discounting."
    )
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-steps", default=5000, type=int)
    parser.add_argument("--num-inner-samples", default=150, type=int)
    parser.add_argument("--num-outer-samples", default=250, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--gamma", default=0.96, type=float)
    parser.add_argument("--num-experiments", default=2, type=int)  # == T
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--hidden-dim", default=256, type=int)
    parser.add_argument("--encoding-dim", default=16, type=int)
    parser.add_argument("--learn-alpha", default=False, action="store_true")
    parser.add_argument("--learn-eps", default=False, action="store_true")
    parser.add_argument("--complete-enum", default=False, action="store_true")
    parser.add_argument("--include-t", default=False, action="store_true")
    parser.add_argument(
        "--num-layers", default=2, type=int, help="Number of hidden layers."
    )
    parser.add_argument(
        "--arch",
        default="sum",
        type=str,
        help="Architecture",
        choices=["static", "sum"],
    )
    parser.add_argument("--mlflow-experiment-name", default="Default", type=str)
    args = parser.parse_args()

    single_run(
        seed=args.seed,
        num_steps=args.num_steps,
        num_inner_samples=args.num_inner_samples,
        num_outer_samples=args.num_outer_samples,
        learn_alpha=args.learn_alpha,
        learn_epsilon=args.learn_eps,
        lr=args.lr,
        gamma=args.gamma,
        device=args.device,
        T=args.num_experiments,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim,
        num_layers=args.num_layers,
        arch=args.arch,
        mlflow_experiment_name=args.mlflow_experiment_name,
        complete_enum=args.complete_enum,
        include_t=args.include_t,
    )
