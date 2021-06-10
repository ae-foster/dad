import math

import torch
from torch.distributions.utils import broadcast_all

import pyro
from pyro import poutine
from pyro.poutine.util import prune_subsample_sites
from pyro.util import warn_if_nan
from pyro.infer.util import torch_item
from pyro.contrib.util import lexpand, rexpand


class MutualInformation(object):
    def __init__(self, num_outer_samples, data_source=None):
        self.data_source = data_source
        self.num_outer_samples = num_outer_samples

    def _vectorized(self, fn, *shape, name="vectorization_plate"):
        """
        Wraps a callable inside an outermost :class:`~pyro.plate` to parallelize
        MI computation over `num_particles`, and to broadcast batch shapes of
        sample site functions in accordance with the `~pyro.plate` contexts
        within which they are embedded.
        :param fn: arbitrary callable containing Pyro primitives.
        :return: wrapped callable.
        """

        def wrapped_fn(*args, **kwargs):
            with pyro.plate_stack(name, shape):
                return fn(*args, **kwargs)

        return wrapped_fn

    def get_primary_rollout(self, model, args, kwargs, graph_type="flat", detach=False):
        if self.data_source is None:
            model = self._vectorized(
                model, self.num_outer_samples, name="outer_vectorization"
            )
        else:
            data = next(self.data_source)
            model = pyro.condition(
                self._vectorized(
                    model, self.num_outer_samples, name="outer_vectorization"
                ),
                data=data,
            )

        trace = poutine.trace(model, graph_type=graph_type).get_trace(*args, **kwargs)
        if detach:
            trace.detach_()
        trace = prune_subsample_sites(trace)
        trace.compute_log_prob()
        return trace


class PriorContrastiveEstimation(MutualInformation):
    def __init__(self, num_outer_samples=100, num_inner_samples=10, data_source=None):
        super().__init__(num_outer_samples, data_source=data_source)
        self.num_inner_samples = num_inner_samples

    def compute_observation_log_prob(self, trace):
        """
        Computes the log probability of observations given latent variables and designs.
        :param trace: a Pyro trace object
        :return: the log prob tensor
        """
        return sum(
            node["log_prob"]
            for node in trace.nodes.values()
            if node.get("subtype") == "observation_sample"
        )

    def get_contrastive_rollout(
        self,
        trace,
        model,
        args,
        kwargs,
        existing_vectorization,
        graph_type="flat",
        detach=False,
    ):
        sampled_observation_values = {
            name: lexpand(node["value"], self.num_inner_samples)
            for name, node in trace.nodes.items()
            if node.get("subtype") in ["observation_sample", "design_sample"]
        }
        conditional_model = self._vectorized(
            pyro.condition(model, data=sampled_observation_values),
            self.num_inner_samples,
            *existing_vectorization,
            name="inner_vectorization",
        )
        trace = poutine.trace(conditional_model, graph_type=graph_type).get_trace(
            *args, **kwargs
        )
        if detach:
            trace.detach_()
        trace = prune_subsample_sites(trace)
        trace.compute_log_prob()

        return trace

    def differentiable_loss(self, model, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the design
        """
        primary_trace = self.get_primary_rollout(model, args, kwargs)
        contrastive_trace = self.get_contrastive_rollout(
            primary_trace, model, args, kwargs, [self.num_outer_samples]
        )

        obs_log_prob_primary = self.compute_observation_log_prob(primary_trace)
        obs_log_prob_contrastive = self.compute_observation_log_prob(contrastive_trace)

        obs_log_prob_combined = torch.cat(
            [lexpand(obs_log_prob_primary, 1), obs_log_prob_contrastive]
        ).logsumexp(0)

        loss = (obs_log_prob_combined - obs_log_prob_primary).mean(0)

        warn_if_nan(loss, "loss")
        return loss

    def loss(self, model, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using prior contrastive estimation; == -EIG proxy
        """
        loss_to_constant = torch_item(self.differentiable_loss(model, *args, **kwargs))
        return loss_to_constant - math.log(self.num_inner_samples + 1)


class NestedMonteCarloEstimation(PriorContrastiveEstimation):
    def differentiable_loss(self, model, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the design
        """
        primary_trace = self.get_primary_rollout(model, args, kwargs)
        contrastive_trace = self.get_contrastive_rollout(
            primary_trace, model, args, kwargs, [self.num_outer_samples]
        )

        obs_log_prob_primary = self.compute_observation_log_prob(primary_trace)
        obs_log_prob_contrastive = self.compute_observation_log_prob(contrastive_trace)
        obs_log_prob_combined = obs_log_prob_contrastive.logsumexp(0)
        loss = (obs_log_prob_combined - obs_log_prob_primary).mean(0)

        warn_if_nan(loss, "loss")
        return loss

    def loss(self, model, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using prior contrastive estimation
        """
        loss_to_constant = torch_item(self.differentiable_loss(model, *args, **kwargs))
        return loss_to_constant - math.log(self.num_inner_samples)


class PriorContrastiveEstimationDiscreteObsTotalEnum(PriorContrastiveEstimation):
    def __init__(self, num_outer_samples, num_inner_samples, data_source=None):
        super().__init__(
            num_outer_samples=num_outer_samples,
            num_inner_samples=num_inner_samples,
            data_source=data_source,
        )
        self.T = None
        self.design_size = None
        self.y_support_size = None
        self.y_possible_paths = None

    def get_all_possible_paths(self, model, *args, **kwargs):
        """Expand histories"""
        main_model_trace = pyro.poutine.trace(model).get_trace()
        obs_samples = [
            node
            for node in main_model_trace.nodes.values()
            if node.get("subtype") == "observation_sample"
        ]
        obs_support = obs_samples[0]["fn"].enumerate_support()
        self.y_support_size = obs_support.shape[0]  # e.g for hyperbolic temp = 2
        self.T = len(obs_samples)

        design_samples = [
            node
            for node in main_model_trace.nodes.values()
            if node.get("subtype") == "design_sample"
        ]
        if len(design_samples) > 0:
            # Make sure there are design variables
            self.design_size = design_samples[0]["value"].shape[0]

        y_possible_paths = []
        for t in range(self.T):
            expansion = [self.y_support_size] * t
            tensor = rexpand(obs_support.clone().detach(), *expansion)
            y_possible_paths.append(tensor)
        # list of size T; each element is of shape support_size**T
        y_possible_paths = broadcast_all(*y_possible_paths)
        # convert the above to a dictionary
        y_possible_paths = {f"y{t + 1}": y_possible_paths[t] for t in range(self.T)}

        self.y_possible_paths = y_possible_paths

    def expand_model_traces(self, model, *args, **kwargs):
        if not self.y_possible_paths:
            # compute possible paths just once
            self.get_all_possible_paths(model, *args, **kwargs)

        y_possible_paths_outer = {
            k: rexpand(v, self.num_outer_samples)
            for k, v in self.y_possible_paths.items()
        }
        y_possible_paths_inner = {
            k: rexpand(v, self.num_inner_samples)
            for k, v in self.y_possible_paths.items()
        }
        design_expansion = pyro.condition(
            self._vectorized(
                model, *([self.y_support_size] * self.T), name="design_expansion"
            ),
            data=self.y_possible_paths,
        )
        # sample a design path for each possible y path
        trace_designs = pyro.poutine.trace(design_expansion).get_trace()
        trace_designs = prune_subsample_sites(trace_designs)
        # dict with T elements; each element is of shape [support_size**T, design_size]
        designs = {
            name: node["value"]
            for name, node in trace_designs.nodes.items()
            if node.get("subtype") == "design_sample"
        }
        # add new dimension of size <INNER> at position -2 to designs and paths
        designs_outer = {
            k: v.unsqueeze(-2).expand(
                [self.y_support_size] * self.T
                + [self.num_outer_samples, self.design_size]
            )
            for k, v in designs.items()
        }
        designs_inner = {
            k: v.unsqueeze(-2).expand(
                [self.y_support_size] * self.T
                + [self.num_inner_samples, self.design_size]
            )
            for k, v in designs.items()
        }

        def outer_vectorization():
            with pyro.plate("outer_vectorization", self.num_outer_samples):
                return model()

        def inner_vectorization():
            with pyro.plate("inner_vectorization", (self.num_inner_samples)):
                return model()

        trace_pce_primary = pyro.poutine.trace(outer_vectorization).get_trace()
        trace_pce_contrastive = pyro.poutine.trace(inner_vectorization).get_trace()
        data_primary = {
            name: node["value"]
            for name, node in trace_pce_primary.nodes.items()
            if node.get("subtype") == "latent_sample"
        }
        # add the y-paths enumeration to the data dict:
        data_primary.update(y_possible_paths_outer)
        data_primary.update(designs_outer)

        data_contrastive = {
            name: node["value"]
            for name, node in trace_pce_contrastive.nodes.items()
            if node.get("subtype") == "latent_sample"
        }
        data_contrastive.update(y_possible_paths_inner)
        data_contrastive.update(designs_inner)

        # Expand and condition
        def expanded_model_primary():
            with pyro.plate_stack(
                "expansion_primary", [2] * self.T + [self.num_outer_samples]
            ):
                # condition on the whole data:
                return pyro.condition(model, data=data_primary)()

        def expanded_model_contrastive():
            with pyro.plate_stack(
                "expansion_contrastive", [2] * self.T + [self.num_inner_samples]
            ):
                # condition on the whole data:
                return pyro.condition(model, data=data_contrastive)()

        ## expand primary trace and compute log probs
        trace_pce_primary = pyro.poutine.trace(expanded_model_primary).get_trace()
        trace_pce_primary = prune_subsample_sites(trace_pce_primary)
        # compute the log-probabilities at each site
        trace_pce_primary.compute_log_prob()

        ## expand Contrastive trace and compute log probs
        trace_pce_contrastive = pyro.poutine.trace(
            expanded_model_contrastive
        ).get_trace()
        trace_pce_contrastive = prune_subsample_sites(trace_pce_contrastive)
        # compute the log-probabilities at each site
        trace_pce_contrastive.compute_log_prob()

        return trace_pce_primary, trace_pce_contrastive

    def differentiable_loss(self, model, *args, **kwargs):
        """
        Surrogate loss using complete enumeration, which can be differentiated
        """
        trace_pce_primary, trace_pce_contrastive = self.expand_model_traces(
            model, args, kwargs
        )
        h_T_log_prob_primary = sum(
            node["log_prob"]
            for node in trace_pce_primary.nodes.values()
            if node.get("name") in [f"y{i}" for i in range(1, self.T + 1)]
        )
        assert h_T_log_prob_primary.shape == torch.Size(
            [self.y_support_size] * self.T + [self.num_outer_samples]
        )
        # entropy of the conditional (numerator); shape is [INNER]
        # Sum over all paths [entropy of the likelihood, p(h_T|theta, pi)]
        likelihood_entropy = (-h_T_log_prob_primary * h_T_log_prob_primary.exp()).sum(
            dim=list(range(0, self.T))
        )
        # average the above over theta
        conditional_entropy = likelihood_entropy.mean(0)

        ## Denominator ###
        h_T_log_prob_contrastive = sum(
            node["log_prob"]
            for node in trace_pce_contrastive.nodes.values()
            if node.get("name") in [f"y{i}" for i in range(1, self.T + 1)]
        )
        assert h_T_log_prob_contrastive.shape == torch.Size(
            [self.y_support_size] * self.T + [self.num_inner_samples]
        )
        # \int p(theta)*p(h_t| theta, pi) which is part of the denom, which is
        # denom = E_theta\sum_h_t p(h_t|theta, pi) log(p(h_t|pi)) = \sum_h_t(..)

        # marg_log_probs = h_T_log_prob_contrastive.logsumexp(-1, keepdims=True)
        h_T_log_prob_combined = torch.cat(
            [
                rexpand(h_T_log_prob_contrastive, self.num_outer_samples),
                h_T_log_prob_primary.unsqueeze(-2),
            ],
            dim=-2,
        )
        assert h_T_log_prob_combined.shape == torch.Size(
            [self.y_support_size] * self.T
            + [self.num_inner_samples + 1, self.num_outer_samples]
        )
        marg_log_probs = h_T_log_prob_combined.logsumexp(-2)
        # \sum_h_t marg_log_probs * log (marg_log_probs)
        marginal_entropy = (
            -(h_T_log_prob_primary.exp() * marg_log_probs)
            .sum(dim=list(range(0, self.T)))
            .mean()
        )
        mi_estimate = -conditional_entropy + marginal_entropy
        loss = -mi_estimate
        warn_if_nan(loss, "loss")
        return loss


class PriorContrastiveEstimationScoreGradient(PriorContrastiveEstimation):
    def __init__(self, num_outer_samples, num_inner_samples):
        super().__init__(
            num_outer_samples=num_outer_samples, num_inner_samples=num_inner_samples
        )

    def differentiable_loss(self, model, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the design
        -> log p(h_T|theta, pi_phi)* const(g_phi) + g_phi
        """
        primary_trace = self.get_primary_rollout(model, args, kwargs)
        contrastive_trace = self.get_contrastive_rollout(
            primary_trace, model, args, kwargs, [self.num_outer_samples]
        )

        obs_log_prob_primary = self.compute_observation_log_prob(primary_trace)
        obs_log_prob_contrastive = self.compute_observation_log_prob(contrastive_trace)
        obs_log_prob_combined = torch.cat(
            [lexpand(obs_log_prob_primary, 1), obs_log_prob_contrastive]
        ).logsumexp(0)

        with torch.no_grad():
            g_no_grad = obs_log_prob_primary - obs_log_prob_combined

        loss = -(g_no_grad * obs_log_prob_primary - obs_log_prob_combined).mean(0)

        warn_if_nan(loss, "loss")
        return loss

    def loss(self, model, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using prior contrastive estimation
        """
        basic_pce = super().differentiable_loss(model, *args, **kwargs)
        loss_to_constant = torch_item(basic_pce)
        loss = loss_to_constant - math.log(self.num_inner_samples + 1)
        return loss
