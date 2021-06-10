import warnings

from pyro.poutine.runtime import am_i_wrapped, apply_stack


def sample_with_type(type_string, name, fn, *args, **kwargs):
    """
    Calls the stochastic function `fn` with additional side-effects depending
    on `name` and the enclosing context (e.g. an inference algorithm).
    See `Intro I <http://pyro.ai/examples/intro_part_i.html>`_ and
    `Intro II <http://pyro.ai/examples/intro_part_ii.html>`_ for a discussion.
    :param type_string: annotate this site with a particular type
    :param name: name of sample
    :param fn: distribution class or function
    :param obs: observed datum (optional; should only be used in context of
        inference) optionally specified in kwargs
    :param dict infer: Optional dictionary of inference parameters specified
        in kwargs. See inference documentation for details.
    :returns: sample
    """
    obs = kwargs.pop("obs", None)
    infer = kwargs.pop("infer", {}).copy()
    # check if stack is empty
    # if stack empty, default behavior (defined here)
    if not am_i_wrapped():
        if obs is not None and not infer.get("_deterministic"):
            warnings.warn(
                "trying to observe a value outside of inference at " + name,
                RuntimeWarning,
            )
            return obs
        return fn(*args, **kwargs)
    # if stack not empty, apply everything in the stack?
    else:
        # initialize data structure to pass up/down the stack
        msg = {
            "type": "sample",
            "subtype": type_string,
            "name": name,
            "fn": fn,
            "is_observed": False,
            "args": args,
            "kwargs": kwargs,
            "value": None,
            "infer": infer,
            "scale": 1.0,
            "mask": None,
            "cond_indep_stack": (),
            "done": False,
            "stop": False,
            "continuation": None,
        }
        # handle observation
        if obs is not None:
            msg["value"] = obs
            msg["is_observed"] = True
        # apply the stack and return its return value
        apply_stack(msg)
        return msg["value"]


def observation_sample(name, fn, *args, **kwargs):
    return sample_with_type("observation_sample", name, fn, *args, **kwargs)


def compute_design(name, fn, *args, **kwargs):
    return sample_with_type("design_sample", name, fn, *args, **kwargs)


def latent_sample(name, fn, *args, **kwargs):
    return sample_with_type("latent_sample", name, fn, *args, **kwargs)
