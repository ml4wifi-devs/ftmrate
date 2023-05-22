from dataclasses import dataclass
from functools import partial, cached_property
from typing import Callable, Tuple, List

import numpy as np
import sympy as sm
import tensorflow_probability.substrates.numpy.distributions as tfd
from sympy.abc import tau, t


DynamicArray = Callable[[float], np.ndarray]
Params = List[sm.Symbol]
StateSpaceSpec = Tuple[DynamicArray, DynamicArray, Params]


@dataclass
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

    @cached_property
    def transition_matrix(self) -> sm.Expr:
        return sm.simplify(sm.exp(-self.beta * t))

    @cached_property
    def transition_covariance(self) -> sm.Expr:
        beta, sigma = self.beta, self.sigma
        return sm.simplify(
            sm.integrate(sm.exp(-beta * (t - tau)) @ sigma @ sigma.T @ sm.exp(-beta * (t - tau)).T, (tau, 0, t)))

    @cached_property
    def free_symbols(self) -> Params:
        return sorted(self.transition_covariance.free_symbols, key=lambda s: s.name)

    def to_numpy(self) -> StateSpaceSpec:
        F = self.transition_matrix
        cov = self.transition_covariance.cholesky(hermitian=False)
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

    def __init__(self, sigma_x: float, sigma_v: float) -> None:
        super().__init__(beta=-sm.ImmutableMatrix([[0, 1], [0, 0]]),
                         sigma=sm.ImmutableMatrix(
                             [[sm.Symbol('\sigma_x'), 0], [0, sm.Symbol('\sigma_v')]]))

        self.sigma_x = sigma_x
        self.sigma_v = sigma_v

    def to_numpy(self) -> Tuple[DynamicArray, DynamicArray]:
        transition_fn, transition_cov_fn, _ = super().to_numpy()

        transition_fn = partial(transition_fn, self.sigma_v, self.sigma_x)
        transition_cov_fn = partial(transition_cov_fn, self.sigma_v, self.sigma_x)

        return transition_fn, transition_cov_fn


@dataclass
class BaseAgent:
    init: Callable
    update: Callable
    sample: Callable


@dataclass
class KalmanFilterState:
    state: np.ndarray
    uncertainty: np.ndarray
    time: float


def kalman_filter(sensor_noise: float, sigma_x: float, sigma_v: float) -> BaseAgent:
    """
    Kalman Filter for FTM distance estimation.

    Parameters
    ----------
    sensor_noise : float
        Sensor noise variance
    sigma_x : float
        Distance process noise variance for SDE dynamics
    sigma_v : float
        Drift process noise variance for SDE dynamics

    Returns
    -------
    agent : BaseAgent
        Container for functions of the KF
    """

    F_fn, Q_fn = ContinuousLocalLinearTrend(sigma_x, sigma_v).to_numpy()
    H = np.array([[1., 0.]])
    R = np.array([[sensor_noise]])
    
    def init(
            distance_init: float = 0.0,
            drift_init: float = 0.0,
            distance_init_noise: float = 50.0,
            drift_init_noise: float = 4.0,
            timestamp: float = 0.0
    ) -> KalmanFilterState:
        """
        Returns the Kalman filter initial state.

        Parameters
        ----------
        distance_init : float, optional
            Initial distance mean, by default 0
        drift_init : float, optional
            Initial drift mean, by default 0
        distance_init_noise : float, optional
            Initial distance variance, by default 50 (The approximate range of modern AP)
        drift_init_noise : float, optional
            Initial drift variance, by default 4 (Humans usually don't run while on Wi-Fi)
        timestamp : float, optional
            Initialization timestamp, by default 0

        Returns
        -------
        state : KalmanFilterState
            Initial Kalman filter state
        """

        return KalmanFilterState(
            state=np.array([[distance_init], [drift_init]]),
            uncertainty=np.array([
                [distance_init_noise ** 2, 0.],
                [0., drift_init_noise ** 2]
            ]),
            time=timestamp
        )

    def update(
            state: KalmanFilterState,
            distance: float,
            time: float
    ) -> KalmanFilterState:
        """
        Performs one step of the Kalman filter algorithm, thus returns the updated state of the agent.

        Parameters
        ----------
        state : KalmanFilterState
            Previous agent state
        distance : float
            Distance measurement
        time : float
            Current time

        Returns
        -------
        state : KalmanFilterState
            Updated agent state
        """
        
        # prediction
        t_delta = time - state.time
        F = F_fn(t_delta)
        x = F @ state.state
        P = F @ state.uncertainty @ F.T + Q_fn(t_delta)

        # update
        z = np.array([distance])
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        x = x + K @ (z - H @ x)
        P = P - K @ H @ P

        return KalmanFilterState(
            state=x,
            uncertainty=P,
            time=time
        )

    def sample(
            state: KalmanFilterState,
            time: float
    ) -> tfd.Distribution:
        """
        Estimates distance distribution from current Kalman filter state.

        Parameters
        ----------
        state : KalmanFilterState
            Current agent state
        time : float
            Current time

        Returns
        -------
        dist : tfd.Distribution
            Predicted distance distribution at t=time
        """

        t_delta = time - state.time
        F = F_fn(t_delta)
        x = F @ state.state
        P = F @ state.uncertainty @ F.T + Q_fn(t_delta)

        return tfd.Normal(loc=x[0, 0], scale=np.sqrt(P[0, 0]))

    return BaseAgent(
        init=init,
        update=update,
        sample=sample
    )
