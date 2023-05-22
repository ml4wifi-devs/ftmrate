import unittest
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from chex import Scalar, Array
from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax.distributions.linear_gaussian_ssm import build_kalman_filter_step, KalmanFilterState

from ml4wifi.agents.kalman_filter import kalman_filter
from ml4wifi.envs.sde import ContinuousLocalLinearTrend
from ml4wifi.params_fit import KF_SENSOR_NOISE
from ml4wifi.params_fit.common import generate_data

tfd = tfp.distributions
tfl = tf.linalg


distance_init: Scalar = 0.0
drift_init: Scalar = 0.0
distance_init_noise: Scalar = 50.0
drift_init_noise: Scalar = 4.0

observation_matrix = tfl.LinearOperatorFullMatrix(jnp.array([[1., 0.]]))  # H matrix
observation_noise = tfd.MultivariateNormalLinearOperator(                 # R matrix
    loc=jnp.array([0.]),
    scale=tfl.LinearOperatorFullMatrix(jnp.array([[KF_SENSOR_NOISE]]))
)
F_fn, Q_fn = ContinuousLocalLinearTrend().jaxify(cholesky=False)


@jax.jit
def tfp_estimate(observations: Array, time: Array) -> Array:

    def get_transition_matrix_for_timestep(timestep: jnp.int32):
        t_delta = time[timestep] - time[timestep - 1]
        return tfl.LinearOperatorFullMatrix(F_fn(t_delta))

    def get_transition_noise_for_timestep(timestep: jnp.int32):
        t_delta = time[timestep] - time[timestep - 1]
        return tfd.MultivariateNormalLinearOperator(
            loc=jnp.array([0., 0.]),
            scale=tfl.LinearOperatorFullMatrix(Q_fn(t_delta))
        )

    def get_observation_matrix_for_timestep(timestep: jnp.int32):
        return observation_matrix

    def get_observation_noise_for_timestep(timestep: jnp.int32):
        return observation_noise

    kalman_onestep = build_kalman_filter_step(
        get_transition_matrix_for_timestep,
        get_transition_noise_for_timestep,
        get_observation_matrix_for_timestep,
        get_observation_noise_for_timestep
    )

    state_mean_prior = jnp.array([[distance_init], [drift_init]])
    state_cov_prior = jnp.array([
        [distance_init_noise ** 2, 0.],
        [0., drift_init_noise ** 2]
    ])

    state_prior = KalmanFilterState(
        state_mean_prior,
        state_cov_prior,
        state_mean_prior,
        state_cov_prior,
        jnp.array([[distance_init]]),
        jnp.array([[KF_SENSOR_NOISE]]),
        tfd.Normal(distance_init, KF_SENSOR_NOISE).log_prob(distance_init),
        1
    )

    def scan_fn(state: KalmanFilterState, observation: Scalar) -> Tuple:
        state = kalman_onestep(state, jnp.expand_dims(observation, axis=(0, 1)))
        distance_estimate = state.filtered_mean[0, 0]
        return state, distance_estimate

    return jax.lax.scan(scan_fn, state_prior, observations)[1]


@jax.jit
def kf_estimate(observations: Array, time: Array) -> Array:
    agent = kalman_filter()

    def scan_fn(carry: Tuple, x: Array) -> Tuple:
        key, state, distance, time = *carry, *x
        key, update_key, sample_key = jax.random.split(key, 3)

        state = agent.update(state, update_key, distance, time)
        distance_distribution = agent.sample(state, sample_key, time)

        return (key, state), distance_distribution.mean()

    key, init_key = jax.random.split(jax.random.PRNGKey(42))
    init = (key, agent.init(init_key))

    return jax.lax.scan(scan_fn, init, jnp.stack([observations, time], axis=1))[1]


class KalmanFilterTestCase(unittest.TestCase):
    plot = True
    save_fig = False

    def test_kalman_filter(self):
        distance_true, distance_measurement, _, time = generate_data(jax.random.PRNGKey(42), 0.0, 50.0, 1.0, 0.5, 101)

        tfp_estimations = tfp_estimate(distance_measurement[1:], time)
        kf_estimations = kf_estimate(distance_measurement, time)

        # compare excluding warmup stage (first 5 steps)
        self.assertTrue(jnp.allclose(tfp_estimations[5:], kf_estimations[6:], atol=0.15))

        if self.plot:
            self.create_plot((time, distance_true, distance_measurement, tfp_estimations, kf_estimations))

    def create_plot(self, results: Tuple) -> None:
        time, distance_true, distance_measurement, tfp_estimations, kfd_estimations = results
        difference = tfp_estimations - kfd_estimations[1:]

        plt.rcParams['lines.linewidth'] = 1

        plt.plot(time, distance_true, label='True')
        plt.plot(time[1:], tfp_estimations, label='TFP estimation')
        plt.plot(time, kfd_estimations, label='KF estimation', color="tab:red")
        plt.plot(time, distance_measurement, linewidth=0.75, linestyle=':', color='black', alpha=1., label='Observed')
    
        plt.xlabel('Time [s]')
        plt.ylabel('Estimated distance [m]')
        plt.title(f'Kalman filter validation\n'
                  f'TFP/KF difference: {difference.mean():.5f} +/- {difference.std():.5f} [m]')

        plt.legend()
        plt.savefig('KF validation.pdf') if self.save_fig else plt.show()


if __name__ == '__main__':
    unittest.main()
