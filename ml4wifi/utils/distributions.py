import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


class FiniteDiscreteQ(tfd.FiniteDiscrete):
    """
    The finite discrete distribution with quantile function.
    """

    def _quantile(self, p):
        cumsum = jnp.cumsum(self.probs_parameter())
        idx = jnp.searchsorted(cumsum, p)
        return self.outcomes[idx]


class DeterministicQ(tfd.Deterministic):
    """
    The deterministic distribution with quantile function.
    """

    def _quantile(self, p):
        return self.loc
