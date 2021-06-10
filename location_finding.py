import argparse

import numpy as np
import pandas as pd

import torch
from torch import nn

import pyro
import pyro.distributions as dist
from pyro.infer.util import torch_item

from tqdm import trange

import mlflow
import mlflow.pytorch

from neural.modules import (
    SetEquivariantDesignNetwork,
    BatchDesignBaseline,
    RandomDesignBaseline,
    rmv,
)

from oed.primitives import observation_sample, latent_sample, compute_design
from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from contrastive.mi import PriorContrastiveEstimation


class EncoderNetwork(nn.Module):
    """Encoder network for location finding example"""

    def __init__(self, design_dim, osbervation_dim, hidden_dim, encoding_dim):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.design_dim_flat = design_dim[0] * design_dim[1]
        input_dim = self.design_dim_flat + osbervation_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, encoding_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, xi, y, **kwargs):
        xi = xi.flatten(-2)
        inputs = torch.cat([xi, y], dim=-1)

        x = self.linear1(inputs)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


class EmitterNetwork(nn.Module):
    """Emitter network for location finding example"""

    def __init__(self, encoding_dim, design_dim):
        super().__init__()
        self.design_dim = design_dim
        self.design_dim_flat = design_dim[0] * design_dim[1]
        self.linear = nn.Linear(encoding_dim, self.design_dim_flat)

    def forward(self, r):
        xi_flat = self.linear(r)
        return xi_flat.reshape(xi_flat.shape[:-1] + self.design_dim)


class HiddenObjects(nn.Module):
    """Location finding example"""

    def __init__(
        self,
        design_net,
        base_signal=0.1,  # G-map hyperparam
        max_signal=1e-4,  # G-map hyperparam
        theta_loc=None,  # prior on theta mean hyperparam
        theta_covmat=None,  # prior on theta covariance hyperparam
        noise_scale=None,  # this is the scale of the noise term
        p=1,  # physical dimension
        K=1,  # number of sources
        T=2,  # number of experiments
    ):
        super().__init__()
        self.design_net = design_net
        self.base_signal = base_signal
        self.max_signal = max_signal
        # Set prior:
        self.theta_loc = theta_loc if theta_loc is not None else torch.zeros((K, p))
        self.theta_covmat = theta_covmat if theta_covmat is not None else torch.eye(p)
        self.theta_prior = dist.MultivariateNormal(
            self.theta_loc, self.theta_covmat
        ).to_event(1)
        # Observations noise scale:
        self.noise_scale = noise_scale if noise_scale is not None else torch.tensor(1.0)
        self.n = 1  # batch=1
        self.p = p  # dimension of theta (location finding example will be 1, 2 or 3).
        self.K = K  # number of sources
        self.T = T  # number of experiments

    def forward_map(self, xi, theta):
        """Defines the forward map for the hidden object example
        y = G(xi, theta) + Noise.
        """
        # two norm squared
        sq_two_norm = (xi - theta).pow(2).sum(axis=-1)
        sq_two_norm_inverse = (self.max_signal + sq_two_norm).pow(-1)
        # sum over the K sources, add base signal and take log.
        mean_y = torch.log(self.base_signal + sq_two_norm_inverse.sum(-1, keepdim=True))
        return mean_y

    def model(self):
        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)

        ########################################################################
        # Sample latent variables theta
        ########################################################################
        theta = latent_sample("theta", self.theta_prior)
        y_outcomes = []
        xi_designs = []

        # T-steps experiment
        for t in range(self.T):
            ####################################################################
            # Get a design xi; shape is [num-outer-samples x 1 x 1]
            ####################################################################
            xi = compute_design(
                f"xi{t + 1}", self.design_net.lazy(*zip(xi_designs, y_outcomes))
            )

            ####################################################################
            # Sample y at xi; shape is [num-outer-samples x 1]
            ####################################################################
            mean = self.forward_map(xi, theta)
            sd = self.noise_scale
            y = observation_sample(f"y{t + 1}", dist.Normal(mean, sd).to_event(1))

            y_outcomes.append(y)
            xi_designs.append(xi)

        return y_outcomes

    def forward(self, theta=None):
        """Run the policy"""
        self.design_net.eval()
        if theta is not None:
            model = pyro.condition(self.model, data={"theta": theta})
        else:
            model = self.model
        designs = []
        observations = []

        with torch.no_grad():
            trace = pyro.poutine.trace(model).get_trace()
            for t in range(self.T):
                xi = trace.nodes[f"xi{t + 1}"]["value"]
                designs.append(xi)

                y = trace.nodes[f"y{t + 1}"]["value"]
                observations.append(y)
        return torch.cat(designs).unsqueeze(1), torch.cat(observations).unsqueeze(1)

    def eval(self, n_trace=3, theta=None, verbose=True):
        """run the policy, print output and return in a pandas df"""
        self.design_net.eval()
        if theta is not None:
            model = pyro.condition(self.model, data={"theta": theta})
        else:
            model = self.model

        output = []
        true_thetas = []
        with torch.no_grad():
            for i in range(n_trace):
                print("\nExample run {}".format(i + 1))
                trace = pyro.poutine.trace(model).get_trace()
                true_theta = trace.nodes["theta"]["value"].cpu()
                if verbose:
                    print(f"*True Theta: {true_theta}*")
                run_xis = []
                run_ys = []
                # Print optimal designs, observations for given theta
                for t in range(self.T):
                    xi = trace.nodes[f"xi{t + 1}"]["value"].cpu().reshape(-1)
                    run_xis.append(xi)
                    y = trace.nodes[f"y{t + 1}"]["value"].cpu().item()
                    run_ys.append(y)
                    if verbose:
                        print(f"xi{t + 1}: {xi}")
                        print(f" y{t + 1}: {y}")

                run_df = pd.DataFrame(torch.stack(run_xis).numpy())
                run_df.columns = [f"xi_{i}" for i in range(self.p)]
                run_df["observations"] = run_ys
                run_df["order"] = list(range(1, self.T + 1))
                run_df["run_id"] = i + 1
                output.append(run_df)
                true_thetas.append(true_theta.numpy())
        print(pd.concat(output))
        return pd.concat(output), true_thetas


def single_run(
    seed,
    num_steps,
    num_inner_samples,  # L in denom
    num_outer_samples,  # N to estimate outer E
    lr,  # learning rate of adam optim
    gamma,  # scheduler for adam optim
    p,  # number of physical dim
    K,  # number of sources
    T,  # number of experiments
    noise_scale,
    base_signal,
    max_signal,
    device,
    hidden_dim,
    encoding_dim,
    mlflow_experiment_name,
    design_network_type,  # "dad" or "static" or "random"
    adam_betas_wd=[0.9, 0.999, 0],  # these are the defaults
):

    pyro.clear_param_store()
    seed = auto_seed(seed)
    *adam_betas, adam_weight_decay = adam_betas_wd

    ### Set up model ###
    n = 1  # batch dim
    encoder = EncoderNetwork((n, p), n, hidden_dim, encoding_dim)
    emitter = EmitterNetwork(encoding_dim, (n, p))
    # Design net: takes pairs [design, observation] as input
    if design_network_type == "static":
        design_net = BatchDesignBaseline(T, (n, p)).to(device)
    elif design_network_type == "random":
        design_net = RandomDesignBaseline(T, (n, p)).to(device)
        num_steps = 0  # no gradient steps needed
    elif design_network_type == "dad":
        design_net = SetEquivariantDesignNetwork(
            encoder, emitter, empty_value=torch.ones(n, p) * 0.01
        ).to(device)
    else:
        raise ValueError(f"design_network_type={design_network_type} not supported.")

    ### Set up Mlflow logging ### ------------------------------------------------------
    mlflow.set_experiment(mlflow_experiment_name)
    ## Reproducibility
    mlflow.log_param("seed", seed)
    ## Model hyperparams
    mlflow.log_param("base_signal", base_signal)
    mlflow.log_param("max_signal", max_signal)
    mlflow.log_param("noise_scale", noise_scale)
    mlflow.log_param("num_experiments", T)
    mlflow.log_param("num_sources", K)
    mlflow.log_param("physical_dim", p)

    ## Design network hyperparams
    mlflow.log_param("design_network_type", design_network_type)
    if design_network_type == "dad":
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("encoding_dim", encoding_dim)
    mlflow.log_param("num_inner_samples", num_inner_samples)
    mlflow.log_param("num_outer_samples", num_outer_samples)

    ## Optimiser hyperparams
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("lr", lr)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("adam_beta1", adam_betas[0])
    mlflow.log_param("adam_beta2", adam_betas[1])
    mlflow.log_param("adam_weight_decay", adam_weight_decay)
    # ----------------------------------------------------------------------------------

    ### Prior hyperparams ###
    # The prior is K independent * p-variate Normals. For example, if there's 1 source
    # (K=1) in 2D (p=2), then we have 1 bivariate Normal.
    theta_prior_loc = torch.zeros((K, p), device=device)  # mean of the prior
    theta_prior_covmat = torch.eye(p, device=device)  # covariance of the prior
    # noise of the model: the sigma in N(G(theta, xi), sigma)
    noise_scale_tensor = noise_scale * torch.tensor(
        1.0, dtype=torch.float32, device=device
    )
    # fix the base and the max signal in the G-map
    ho_model = HiddenObjects(
        design_net=design_net,
        base_signal=base_signal,
        max_signal=max_signal,
        theta_loc=theta_prior_loc,
        theta_covmat=theta_prior_covmat,
        noise_scale=noise_scale_tensor,
        p=p,
        K=K,
        T=T,
    )

    ### Set-up optimiser ###
    optimizer = torch.optim.Adam
    # Annealed LR. Set gamma=1 if no annealing required
    scheduler = pyro.optim.ExponentialLR(
        {
            "optimizer": optimizer,
            "optim_args": {
                "lr": lr,
                "betas": adam_betas,
                "weight_decay": adam_weight_decay,
            },
            "gamma": gamma,
        }
    )
    ### Set-up loss ###
    pce_loss = PriorContrastiveEstimation(num_outer_samples, num_inner_samples)

    oed = OED(ho_model.model, scheduler, pce_loss)

    ### Optimise ###
    loss_history = []
    num_steps_range = trange(0, num_steps, desc="Loss: 0.000 ")
    for i in num_steps_range:
        loss = oed.step()
        loss = torch_item(loss)
        loss_history.append(loss)
        # Log every 50 losses -> too slow (and unnecessary to log everything)
        if i % 50 == 0:
            num_steps_range.set_description("Loss: {:.3f} ".format(loss))
            loss_eval = oed.evaluate_loss()
            mlflow.log_metric("loss", loss_eval)
        # Decrease LR at every 1K steps
        if i % 1000 == 0:
            scheduler.step()

    # log some basic metrics: %decrease in loss over the entire run
    if len(loss_history) == 0:
        # this happens when we have random designs - there are no grad updates
        loss = torch_item(pce_loss.differentiable_loss(ho_model.model))
        mlflow.log_metric("loss", loss)
        mlflow.log_metric("loss_diff50", 0)
        mlflow.log_metric("loss_av50", loss)
    else:
        loss_diff50 = np.mean(loss_history[-51:-1]) / np.mean(loss_history[0:50]) - 1
        mlflow.log_metric("loss_diff50", loss_diff50)
        loss_av50 = np.mean(loss_history[-51:-1])
        mlflow.log_metric("loss_av50", loss_av50)

    ho_model.eval()
    # Store the results dict as an artifact
    print("Storing model to MlFlow... ", end="")
    mlflow.pytorch.log_model(ho_model.cpu(), "model")
    ml_info = mlflow.active_run().info
    model_loc = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/model"
    print(f"Model sotred in {model_loc}. Done.")
    print(f"The experiment-id of this run is {ml_info.experiment_id}")

    return ho_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Deep Adaptive Design example: Hidden Object Detection."
    )
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-steps", default=500, type=int)
    parser.add_argument("--num-inner-samples", default=100, type=int)
    parser.add_argument("--num-outer-samples", default=200, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("-p", default=2, type=int)
    parser.add_argument("--num-experiments", default=5, type=int)  # == T
    parser.add_argument("--num-sources", default=2, type=int)  # == K
    parser.add_argument("--noise-scale", default=0.5, type=float)
    parser.add_argument("--base-signal", default=0.1, type=float)
    parser.add_argument("--max-signal", default=1e-4, type=float)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--hidden-dim", default=128, type=int)
    parser.add_argument("--encoding-dim", default=8, type=int)
    parser.add_argument("--design-network-type", default="dad", type=str)
    parser.add_argument("--adam-betas-wd", nargs="+", default=[0.8, 0.998, 0])
    parser.add_argument(
        "--mlflow-experiment-name", default="locfin_camera_version", type=str
    )

    args = parser.parse_args()

    single_run(
        seed=args.seed,
        num_steps=args.num_steps,
        num_inner_samples=args.num_inner_samples,
        num_outer_samples=args.num_outer_samples,
        lr=args.lr,
        gamma=args.gamma,
        device=args.device,
        p=args.p,
        K=args.num_sources,
        T=args.num_experiments,
        noise_scale=args.noise_scale,
        base_signal=args.base_signal,
        max_signal=args.max_signal,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim,
        mlflow_experiment_name=args.mlflow_experiment_name,
        design_network_type=args.design_network_type,
        adam_betas_wd=args.adam_betas_wd,
    )
