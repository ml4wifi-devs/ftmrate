import unittest

import jax
import jax.numpy as jnp
import sympy as sm
import tensorflow_probability.substrates.jax as tfp

import ml4wifi.envs.sde as sde
from ml4wifi.params_fit import KF_SENSOR_NOISE

tfd = tfp.distributions


class SDETestCase(unittest.TestCase):
    def test_symbols(self):
        oup = sde.OrnsteinUhlenbeckProcess(
            beta=-sm.ImmutableMatrix([
                [0, 1],
                [0, 0]
            ]),
            sigma=sm.ImmutableMatrix([
                [sm.Symbol('\sigma_x'), 0],
                [0, sm.Symbol('\sigma_v')]
            ])
        )
        self.assertTrue(isinstance(oup.transition_covariance, sm.ImmutableDenseMatrix))

    def test_jaxify(self):
        cllt = sde.ContinuousLocalLinearTrend()
        t = 3.22

        F_fn, Q_fn = cllt.jaxify(True)
        F, Q = F_fn(t), Q_fn(t)

        ...

    def test_forecast(self):
        cllt = sde.ContinuousLocalLinearTrend()
        t = 3.22

        F_fn, Q_fn = cllt.jaxify(True)
        F, Q = F_fn(t), Q_fn(t)

        lgssm = tfd.LinearGaussianStateSpaceModel(
            num_timesteps=1,  # just one forecast
            transition_matrix=F,
            transition_noise=tfd.MultivariateNormalTriL(scale_tril=Q),
            observation_matrix=jnp.array([[1., 0.]]),
            observation_noise=tfd.MultivariateNormalDiag(scale_diag=jnp.array([KF_SENSOR_NOISE])),  # measurement error
            initial_state_prior=tfd.MultivariateNormalDiag(scale_diag=jnp.array([0., 0.]))
        )

        path = lgssm.sample(seed=jax.random.PRNGKey(42))
        results = lgssm.forward_filter(path)

        obs_dist = tfd.MultivariateNormalFullCovariance(
            loc=results.observation_means,
            covariance_matrix=results.observation_covs
        )

        # check https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LinearGaussianStateSpaceModel#forward_filter for details

        ...


if __name__ == '__main__':
    unittest.main()
