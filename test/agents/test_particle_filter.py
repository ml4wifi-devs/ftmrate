import unittest
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from chex import Scalar, Array, PRNGKey
from tensorflow_probability.substrates import jax as tfp

from ml4wifi.agents.particle_filter import particle_filter, observation_fn, transition_fn
from ml4wifi.params_fit.common import generate_data

tfd = tfp.distributions


min_d: Scalar = 0.0
max_d: Scalar = 50.0
min_v: Scalar = -4.0
max_v: Scalar = 4.0
particles_num: jnp.int32 = 1000
measurement_interval: Scalar = 0.5


class UniformPF(tfd.Uniform):
    """
    Custom uniform distribution to force 1D initial weights in TFP Particle Filter.
    """

    def _prob(self, x):
        return jnp.empty(len(x))


@jax.jit
def tfp_estimate(observations: Array) -> Array:
    trajectories, _ = tfp.experimental.mcmc.infer_trajectories(
        observations=observations,
        initial_state_prior=UniformPF([min_d, min_v], [max_d, max_v]),
        transition_fn=lambda _, particles: transition_fn(particles, measurement_interval),
        observation_fn=lambda _, particles: observation_fn(particles),
        num_particles=particles_num,
        num_transitions_per_observation=1,
        seed=jax.random.PRNGKey(42)
    )
    return jnp.mean(trajectories[..., 0], axis=1)


@jax.jit
def pf_estimate(observations: Array, time: Array) -> Array:
    agent = particle_filter(min_d, max_d, min_v, max_v, particles_num)

    def scan_fn(carry: Tuple, x: Array) -> Tuple:
        key, state, distance, time = *carry, *x
        key, update_key, sample_key = jax.random.split(key, 3)

        state = agent.update(state, update_key, distance, time)
        distance_distribution = agent.sample(state, sample_key, time)

        return (key, state), distance_distribution.mean()

    key, init_key = jax.random.split(jax.random.PRNGKey(42))
    init = (key, agent.init(init_key))
    
    return jax.lax.scan(scan_fn, init, jnp.stack([observations, time], axis=1))[1]


class ParticleFilterTestCase(unittest.TestCase):
    plot = True
    save_fig = False

    def test_particle_filter(self):
        distance_true, distance_measurement, _, time = generate_data(jax.random.PRNGKey(42), 0.0, 50.0, 1.0, 0.5, 101)

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


if __name__ == '__main__':
    unittest.main()
