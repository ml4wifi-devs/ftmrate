import dataclasses
from argparse import ArgumentParser
from functools import partial
from typing import Tuple, Dict, Union

import os.path

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import seaborn as sns
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import bijectors as tfb
from chex import Array, Scalar, PRNGKey

from ml4wifi.params_fit import generate_rwpm, load_parameters_file, CSV_FILES_DIR
from ml4wifi.utils.measurement_manager import DEFAULT_INTERVAL
from ml4wifi.envs import sde
import ml4wifi.agents.particle_filter as pf

@chex.dataclass
class Params:
    sigma_r:Union[Array,Scalar]
    sigma_v:Union[Array,Scalar]

@dataclasses.dataclass
class HParams:
    min_xy: Scalar = -50.0
    max_xy: Scalar = 50.0
    min_v: Scalar = -4.0
    max_v: Scalar = 4.0
    particles_num: jnp.int32 = 1024

HP=HParams()
def log_prob(params:Params,observations:Array)->chex.Numeric:
    """

      Returns:
        incremental_log_marginal_likelihoods: float ,
          giving the natural logarithm of an unbiased estimate of
          `p(observations[t] | observations[:t])` at each timestep `t`. Note that
          (by [Jensen's inequality](
          https://en.wikipedia.org/wiki/Jensen%27s_inequality))
          this is *smaller* in expectation than the true
          `log p(observations[t] | observations[:t])`.
    """

    llt = sde.ContinuousLocalLinearTrend()
    # Restore default `jaxify` behavoiur for `ContinuousLocalLinearTrend`
    # symbls _ [\sigma_v, \sigma_x, t]
    transition_fn, transition_cov_fn,_ = sde.OrnsteinUhlenbeckProcess.jaxify(llt, cholesky=True)

    def pf_transition_fn(particles: Dict[str, Array], t_delta: Scalar) -> tfd.Distribution:
        # Variance is a random walk; is the same in components and independent axes.
        F = transition_fn(params.sigma_v,params.sigma_r, t_delta)
        Q = transition_cov_fn(params.sigma_v,params.sigma_r, t_delta)

        d = dict(
            x=tfd.MultivariateNormalTriL(particles['x'] @ F.T, Q),
            y=tfd.MultivariateNormalTriL(particles['y'] @ F.T, Q)
        )
        return tfd.JointDistributionNamed(d)

    _, lps = tfp.experimental.mcmc.infer_trajectories(
        observations=observations,
        initial_state_prior=pf.initial_state_prior_fn([HP.min_xy, HP.min_v], [HP.max_xy, HP.max_v]),
        transition_fn=lambda _, particles: pf_transition_fn(particles, DEFAULT_INTERVAL),
        observation_fn=lambda _, particles: pf.observation_fn(particles),
        num_particles=HP.particles_num,
        num_transitions_per_observation=1,
        seed=jax.random.PRNGKey(42)
    )

    return jnp.sum(lps)


if __name__ == '__main__':


    p = Params(sigma_r=0.1, sigma_v=0.2)
    d=jnp.asarray([1.,2,3,4])
    l=log_prob(p,d)
    dl = jax.jit(jax.grad(log_prob))(p,d)

    pass