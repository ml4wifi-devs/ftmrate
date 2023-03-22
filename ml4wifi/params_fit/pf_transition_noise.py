import dataclasses
import logging
from argparse import ArgumentParser
from typing import Dict, Union

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
from chex import Array, Scalar
from tensorflow_probability.substrates.jax import distributions as tfd

import ml4wifi.agents.particle_filter as pf
from ml4wifi.envs import sde
from ml4wifi.params_fit import generate_rwpm
from ml4wifi.params_fit.sde_sigma import generate_run_fn
from ml4wifi.utils.measurement_manager import DEFAULT_INTERVAL


@chex.dataclass
class Params:
    sigma_r: Union[Array, Scalar]
    sigma_v: Union[Array, Scalar]

    @property
    def constrained(self):
        return tree.tree_map(jax.nn.softplus,self)


@dataclasses.dataclass
class HParams:
    min_xy: Scalar = -50.0
    max_xy: Scalar = 50.0
    min_v: Scalar = -4.0
    max_v: Scalar = 4.0
    particles_num: jnp.int32 = 1024


HP = HParams()


def log_prob(params: Params, observations: Array) -> chex.Numeric:
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
    transition_fn, transition_cov_fn, _ = sde.OrnsteinUhlenbeckProcess.jaxify(llt, cholesky=True)

    def pf_transition_fn(particles: Dict[str, Array], t_delta: Scalar) -> tfd.Distribution:
        # Variance is a random walk; is the same in components and independent axes.
        F = transition_fn(params.sigma_v, params.sigma_r, t_delta)
        Q = transition_cov_fn(params.sigma_v, params.sigma_r, t_delta)

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


@chex.dataclass
class TrainState:
    params: Params
    opt_state: optax.OptState
    key: chex.PRNGKey


if __name__ == '__main__':


    args = ArgumentParser()
    args.add_argument('--lr', default=0.15, type=float)
    #args.add_argument('--decay', default=0.95, type=float)
    args.add_argument('--n_datasets', default=8, type=int)
    args.add_argument('--n_frames', default=2001, type=int)
    #args.add_argument('--n_samples', default=2000, type=int)
    args.add_argument('--n_steps', default=800, type=int)
    #args.add_argument('--output_name', default='parameters.csv', type=str)
    args.add_argument('--plot', action='store_true', default=False)
    args.add_argument('--seed', default=42, type=int)
    args = args.parse_args()

    n_datasets = args.n_datasets
    n_frames = args.n_frames

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    logging.info(args)



    p = Params(sigma_r=1.3, sigma_v=0.015)
    p = tree.tree_map(jnp.asarray, p)

    opt = optax.adam(learning_rate=args.lr)

    ts = TrainState(
        params=p,
        opt_state=opt.init(p),
        key=jax.random.PRNGKey(args.seed)
    )


    def loss(theta: Params, data: chex.Array) -> Scalar:
        theta = tree.tree_map(jax.nn.softplus, theta)
        ll = jax.vmap(log_prob, in_axes=(None, 0))(theta, data)
        return -ll.mean()


    @jax.jit
    def update(train_state: TrainState, data: Array):
        del data
        k1, k2 = jax.random.split(train_state.key)

        k = jax.random.split(k2, n_datasets)
        gen = lambda key: generate_rwpm(key,
                                        time_total=int((n_frames - 1) * DEFAULT_INTERVAL),
                                        measurement_interval=DEFAULT_INTERVAL,
                                        frames_total=n_frames
                                        )
        _, distance_measurement, *_ = jax.vmap(gen)(k)

        l, grads = jax.value_and_grad(loss)(train_state.params, distance_measurement)
        updates, new_opt_state = opt.update(grads, train_state.opt_state)
        new_params = optax.apply_updates(train_state.params, updates)
        return train_state.replace(params=new_params, opt_state=new_opt_state, key=k1), l

    logging.info('scan')
    ts, losses = jax.lax.scan(update, ts, xs=None, length=args.n_steps)

    logging.info(str(ts.params.constrained))


    pass
