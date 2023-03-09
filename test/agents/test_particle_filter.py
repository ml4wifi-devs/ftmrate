import unittest
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from chex import Scalar, Array, PRNGKey
from tensorflow_probability.substrates import jax as tfp

from ml4wifi.agents.particle_filter import particle_filter, observation_fn, transition_fn, initial_state_prior_fn, \
    _dict_2r

tfd = tfp.distributions

min_xy: Scalar = -50.0
max_xy: Scalar = 50.0
min_v: Scalar = -4.0
max_v: Scalar = 4.0
particles_num: jnp.int32 = 1024
measurement_interval: Scalar = 0.5


@jax.jit
def tfp_estimate(observations: Array) -> Array:
    trajectories, _ = tfp.experimental.mcmc.infer_trajectories(
        observations=observations,
        initial_state_prior=initial_state_prior_fn([min_xy, min_v], [max_xy, max_v]),
        transition_fn=lambda _, particles: transition_fn(particles, measurement_interval),
        observation_fn=lambda _, particles: observation_fn(particles),
        num_particles=particles_num,
        num_transitions_per_observation=1,
        seed=jax.random.PRNGKey(42)
    )

    r = _dict_2r(trajectories)
    return jnp.mean(r, axis=1)


@jax.jit
def pf_estimate(observations: Array, time: Array) -> Array:
    agent = particle_filter(min_xy, max_xy, min_v, max_v, particles_num)

    def scan_fn(carry: Tuple, x: Array) -> Tuple:
        key, state, distance, time = *carry, *x
        key, update_key, sample_key = jax.random.split(key, 3)

        state = agent.update(state, update_key, distance, time)
        distance_distribution = agent.sample(state, sample_key, time)

        return (key, state), distance_distribution.mean()

    key, init_key = jax.random.split(jax.random.PRNGKey(42))
    init = (key, agent.init(init_key))

    return jax.lax.scan(scan_fn, init, jnp.stack([observations, time], axis=1))[1]


@chex.dataclass
class State:
    dmstate: dict
    agent_state: dict
    t: chex.Array
    r: chex.Array = None

    @property
    def true_r(self):
        s = self.dmstate
        return jnp.sqrt(jnp.square(s['x'][..., 0]) + jnp.square(s['y'][..., 0]))


def _data_2d(key: PRNGKey) -> Tuple:
    agent = particle_filter()
    init, update, sample = agent.values()

    init_key, key = jax.random.split(key)
    sim0 = init(init_key)

    t_delta = 0.1
    start = State(
        t=0,
        dmstate=jax.tree_util.tree_map(lambda x: x[:1, :], sim0.particles.particles),
        r=jnp.asarray([0.]),
        agent_state=init(init_key)
    )

    @jax.jit
    def scan_fn(prev: State, k: jax.random.PRNGKey) -> Tuple[State, State]:
        k1, k2, k3, k4 = jax.random.split(k, 4)
        s = prev.replace(
            dmstate=transition_fn(prev.dmstate, t_delta).sample(seed=k1),
            t=prev.t + t_delta,
        )
        s = s.replace(r=observation_fn(prev.dmstate).sample(seed=k2))
        s = s.replace(agent_state=update(prev.agent_state, k3, s.r, s.t))
        return s, s

    scan_keys = jax.random.split(key, 64)
    send, states = jax.lax.scan(scan_fn, start, scan_keys)

    return states.true_r.ravel(), states.r.ravel(), None, states.t.ravel()


class ParticleFilterTestCase(unittest.TestCase):
    plot = True
    save_fig = False

    def test_particle_filter(self):
        distance_true, distance_measurement, _, time = _data_2d(jax.random.PRNGKey(42))

        tfp_estimations = tfp_estimate(distance_measurement[1:])
        pf_estimations = pf_estimate(distance_measurement, time)

        # compare excluding warmup stage (first 5 steps)
        self.assertTrue(jnp.allclose(tfp_estimations[5:], pf_estimations[6:], atol=0.25, rtol=0.05))

        if self.plot:
            self.create_plot((time, distance_true, distance_measurement, tfp_estimations, pf_estimations))

    def create_plot(self, results: Tuple) -> None:
        time, distance_true, distance_measurement, tfp_estimations, pf_estimations = results
        difference = tfp_estimations - pf_estimations[1:]

        plt.rcParams['lines.linewidth'] = 1

        plt.plot(time, distance_true, label='True')
        plt.plot(time[1:], tfp_estimations, label='TFP estimation')
        plt.plot(time, pf_estimations, label='PF estimation', color="tab:red")
        plt.plot(time, distance_measurement, linewidth=0.75, linestyle=':', color='black', alpha=1., label='Observed')

        plt.xlabel('Time [s]')
        plt.ylabel('Estimated distance [m]')
        plt.title(f'Particle filter validation\n'
                  f'TFP/PF difference: {difference.mean():.5f} +/- {difference.std():.5f} [m]')

        plt.legend()
        plt.savefig('PF validation.pdf') if self.save_fig else plt.show()


class PF2DTest(unittest.TestCase):
    def test_agent(self):
        key = jax.random.PRNGKey(42)
        agent = particle_filter()
        state = agent.init(key)
        state = agent.update(state, key, 10., 2.)
        samples = agent.sample(state, key, 2.5)
        self.assertTrue(jnp.isfinite(samples.mean()))


class GradPf(unittest.TestCase):
    """
    Based on https://github.com/tensorflow/probability/blob/9a11541598a2fc08c3fd3c08e92cdda5514c72cc/tensorflow_probability/python/experimental/mcmc/particle_filter_test.py#L599
    """

    def test_fit(self):
        data_2d = _data_2d(jax.random.PRNGKey(42))

        def marginal_log_likelihood(noise_scale):
            def _observation_fn(particles: dict[str, Array]) -> tfd.Distribution:
                r = _dict_2r(particles)
                return tfp.bijectors.Shift(r)(tfd.Normal(loc=0, scale=jax.nn.softplus(noise_scale)))

            _, lps = tfp.experimental.mcmc.infer_trajectories(
                observations=data_2d[1],
                initial_state_prior=initial_state_prior_fn([min_xy, min_v], [max_xy, max_v]),
                transition_fn=lambda _, particles: transition_fn(particles, measurement_interval),
                observation_fn=lambda _, particles: _observation_fn(particles),
                num_particles=particles_num,
                num_transitions_per_observation=1,
                seed=jax.random.PRNGKey(42)
            )

            return jnp.sum(lps)

        grads_value_fn = jax.value_and_grad(marginal_log_likelihood)
        value, grad = grads_value_fn(1.)
        self.assertIsNotNone(grad)


if __name__ == '__main__':
    unittest.main()
