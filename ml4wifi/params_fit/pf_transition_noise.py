import dataclasses
from argparse import ArgumentParser
from functools import partial
from typing import Tuple, Dict, Union

import os.path

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import seaborn as sns
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import bijectors as tfb
from chex import Array, Scalar, PRNGKey
from jax.scipy.optimize import minimize
from jax.flatten_util import ravel_pytree
import jax.tree_util as tree

from ml4wifi.params_fit import generate_rwpm, load_parameters_file, CSV_FILES_DIR
from ml4wifi.params_fit.sde_sigma import generate_run_fn
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

    n_datasets=8
    n_frames=128

    data = jnp.empty((n_datasets, n_frames, 1))
    key = jax.random.PRNGKey(42)

    for i in range(n_datasets):
        data, key = generate_run_fn(data, key, i, n_frames)

    p = Params(sigma_r=1.3, sigma_v=0.015)
    p = tree.tree_map(jnp.asarray,p)
    #ap, unravel_fn = ravel_pytree(p)

    #@jax.jit
    def loss(theta:Params)->Scalar:
        theta = tree.tree_map(jax.nn.softplus, theta)
        ll = jax.vmap(log_prob,in_axes=(None,0))(theta,data[...,0])
        return -ll.mean()

    opt = optax.adam(learning_rate=0.01)

    opt_state = opt.init(p)

    @jax.jit
    def update(params, opt_state):
        l, grads = jax.value_and_grad(loss)(params)
        updates, new_opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return l,new_params, new_opt_state

    losses=[]
    params = p
    for _ in range(100):
        l, params, opt_state = update(params,opt_state)
        losses.append(np.asarray(l))

    print(tree.tree_map(jax.nn.softplus, params))


    #hat = minimize(loss,ap,method="BFGS",options=dict(maxiter=8000))


    pass