import jax
import jax.numpy as jnp
from chex import Array, Scalar, dataclass, PRNGKey
from tensorflow_probability.substrates import jax as tfp

from ml4wifi.agents.base_managers_container import BaseAgent, BaseManagersContainer
from ml4wifi.envs import ContinuousLocalLinearTrend
from ml4wifi.params_fit import KF_SENSOR_NOISE

tfd = tfp.distributions


@dataclass
class KalmanFilterState:
    state: Array
    uncertainty: Array
    time: Scalar


def kalman_filter() -> BaseAgent:
    """
    Kalman Filter for FTM distance estimation.

    Returns
    -------
    agent : BaseAgent
        Container for functions of the KF
    """

    F_fn, Q_fn = ContinuousLocalLinearTrend().jaxify(cholesky=False)
    H = jnp.array([[1., 0.]])
    R = jnp.array([[KF_SENSOR_NOISE]])  # calculated as the variance of normal distribution witch is most
                                        # similar to distance_noise distribution in terms of KL divergence
    
    def init(
            key: PRNGKey,
            distance_init: Scalar = 0.0,
            drift_init: Scalar = 0.0,
            distance_init_noise: Scalar = 50.0,
            drift_init_noise: Scalar = 4.0,
            timestamp: float = 0.0
    ) -> KalmanFilterState:
        """
        Returns the Kalman filter initial state.

        Parameters
        ----------
        key : PRNGKey
            JAX random generator key
        distance_init : Scalar, optional
            Initial distance mean, by default 0
        drift_init : Scalar, optional
            Initial drift mean, by default 0
        distance_init_noise : Scalar, optional
            Initial distance variance, by default 50 (The approximate range of modern AP)
        drift_init_noise : Scalar, optional
            Initial drift variance, by default 4 (Humans usually don't run while on WiFi)
        timestamp : float, optional
            Initialization timestamp, by default 0

        Returns
        -------
        state : KalmanFilterState
            Initial Kalman filter state
        """

        return KalmanFilterState(
            state=jnp.array([[distance_init], [drift_init]]),
            uncertainty=jnp.array([
                [distance_init_noise ** 2, 0.],
                [0., drift_init_noise ** 2]
            ]),
            time=timestamp
        )

    def update(
            state: KalmanFilterState,
            key: PRNGKey,
            distance: Scalar,
            time: Scalar
    ) -> KalmanFilterState:
        """
        Performs one step of the Kalman filter algorithm, thus returns the updated state of the agent.

        Parameters
        ----------
        state : KalmanFilterState
            Previous agent state
        key : PRNGKey
            JAX random generator key
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
        z = jnp.array([distance])
        K = P @ H.T @ jnp.linalg.inv(H @ P @ H.T + R)
        x = x + K @ (z - H @ x)
        P = P - K @ H @ P

        return KalmanFilterState(
            state=x,
            uncertainty=P,
            time=time
        )

    def sample(
            state: KalmanFilterState,
            key: PRNGKey,
            time: Scalar
    ) -> tfd.Distribution:
        """
        Estimates distance distribution from current Kalman filter state.

        Parameters
        ----------
        state : KalmanFilterState
            Current agent state
        key : PRNGKey
            JAX random generator key
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

        return tfd.Normal(loc=x[0, 0], scale=jnp.sqrt(P[0, 0]))

    return BaseAgent(
        init=jax.jit(init),
        update=jax.jit(update),
        sample=jax.jit(sample)
    )


class ManagersContainer(BaseManagersContainer):
    def __init__(self, seed: int) -> None:
        super().__init__(seed, kalman_filter)
