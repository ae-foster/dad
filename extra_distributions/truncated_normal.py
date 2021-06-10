import math
from numbers import Number

from pyro.distributions import TorchDistribution
import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all


class LowerTruncatedNormal(TorchDistribution, ExponentialFamily):
    r"""
    Created a Normal distribution that is truncated from below.

    Example::

        >>> m = LowerTruncatedNormal(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([-1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
        truncation (float or Tensor): point to truncate the Normal
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'truncation': constraints.real}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, truncation, validate_args=None):
        self.loc, self.scale, self.truncation = broadcast_all(loc, scale, truncation)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(LowerTruncatedNormal, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LowerTruncatedNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.truncation = self.truncation.expand(batch_shape)
        super(LowerTruncatedNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        raw_samples = self.icdf(u)
        # You do need a clamp here, for the occasional bad sample
        samples = torch.nn.functional.relu(raw_samples - self.truncation) + self.truncation
        return samples

    def _normal_log_prob(self, value):
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Number) else self.scale.log()
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # For now, ignore values in an invalid range
        return self._normal_log_prob(value) - self._normal_cdf(2 * self.loc - self.truncation).log()

    def _normal_cdf(self, value):
        return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))

    def _normal_icdf(self, value):
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (self._normal_cdf(value) - self._normal_cdf(self.truncation)).clamp(min=0.)

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        cdf_truncation = self._normal_cdf(self.truncation)
        rescaled_value = cdf_truncation + (1. - cdf_truncation) * value
        return self._normal_icdf(rescaled_value)
