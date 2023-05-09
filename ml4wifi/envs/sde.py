import dataclasses
import functools as ft
from typing import Tuple, Callable, List

import numpy as np
import sympy as sm
from sympy.abc import tau, t

from ml4wifi.params_fit import SIGMA_V, SIGMA_X

DynamicArray = Callable[[float], np.ndarray]
Params = List[sm.Symbol]
StateSpaceSpec = Tuple[DynamicArray, DynamicArray, Params]


@dataclasses.dataclass
class OrnsteinUhlenbeckProcess:
    """
    Symbolic representation of the general Ornstein Uhlenbeck process.
    The process satisfies the SDE
    dX = −βXdt + σdW .
    Refs:
    https://www.math.nyu.edu/~goodman/teaching/MonteCarlo07/notes/sde2.pdf
    """

    beta: sm.ImmutableMatrix
    sigma: sm.ImmutableMatrix

    @ft.cached_property
    def transition_matrix(self):
        """
        Transition matrix for discretized process at time step t
        """

        return sm.simplify(sm.exp(-self.beta * t))

    @ft.cached_property
    def transition_covariance(self):
        """
        Transition noise covariance for discretized process at time step t
        """

        beta, sigma = self.beta, self.sigma
        return sm.simplify(
            sm.integrate(sm.exp(-beta * (t - tau)) @ sigma @ sigma.T @ sm.exp(-beta * (t - tau)).T, (tau, 0, t)))

    @ft.cached_property
    def free_symbols(self) -> Params:
        """
        Model params ad time
        """

        return sorted(self.transition_covariance.free_symbols, key=lambda s: s.name)

    def jaxify(self, cholesky: bool = True) -> StateSpaceSpec:
        """
        Construct functions of jax Arrays and time producing Linear Gausian State Space Model parameters for Given ptocess.
        If cholesky is true, the covariance is returned as cholesky decomposition suitable for `tfp.distributions.MultivariateNormalTriL`
        """

        F = self.transition_matrix
        cov = self.transition_covariance
        cov = cov.cholesky(hermitian=False) if cholesky else cov
        inputs = self.free_symbols
        transition_fn = sm.lambdify(inputs, F, modules='numpy')
        transition_cov_fn = sm.lambdify(inputs, cov, modules='numpy')
        return transition_fn, transition_cov_fn, inputs


class ContinuousLocalLinearTrend(OrnsteinUhlenbeckProcess):
    """
    ContinuousLocalLinearTrend is the continuous equivalent of the local linear trend.

    Our model looks like this:
    `dv = sigma_v dW1`
    `dr = v dt + sigma_r dW2`
    where w1 and w2 are independent Wiener processes.
    We define the state s = [r, v] then
    ds = beta s + sigma dw and beta matrix encode the above equations.

    refs:
    https://www.wolfram.com/mathematica/new-in-10/enhanced-random-processes/identify-regularly-sampled-ornstein-uhlenbeck-proc.html
    """

    def __init__(self):
        super().__init__(beta=-sm.ImmutableMatrix([[0, 1], [0, 0]]),
                         sigma=sm.ImmutableMatrix(
                             [[sm.Symbol('\sigma_x'), 0], [0, sm.Symbol('\sigma_v')]]))

    def jaxify(self, cholesky: bool = True) -> Tuple[DynamicArray, DynamicArray]:
        transition_fn, transition_cov_fn, _ = super().jaxify(cholesky)

        transition_fn = ft.partial(transition_fn, SIGMA_V, SIGMA_X)
        transition_cov_fn = ft.partial(transition_cov_fn, SIGMA_V, SIGMA_X)

        return transition_fn, transition_cov_fn
