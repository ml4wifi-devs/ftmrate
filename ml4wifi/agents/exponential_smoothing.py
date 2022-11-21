import jax
import tensorflow_probability.substrates.jax as tfp
from chex import dataclass, Scalar, PRNGKey

from ml4wifi.agents import BaseAgent, BaseManagersContainer
from ml4wifi.utils.distributions import DeterministicQ
from ml4wifi.params_fit import ES_ALPHA, ES_BETA

tfd = tfp.distributions


@dataclass
class ExponentialSmoothingState:
    level: Scalar
    trend: Scalar
    time: Scalar


def exponential_smoothing(
        alpha: Scalar = ES_ALPHA,
        beta: Scalar = ES_BETA
) -> BaseAgent:

    def init(key: PRNGKey) -> ExponentialSmoothingState:
        """
        Returns the exponential smoothing agent initial state.

        Parameters
        ----------
        key : PRNGKey
            JAX random generator key

        Returns
        -------
        state : ExponentialSmoothingState
            Initial exponential smoothing agent state
        """

        return ExponentialSmoothingState(
            level=0.0,
            trend=0.0,
            time=-1.0
        )

    def update(
            state: ExponentialSmoothingState,
            key: PRNGKey,
            distance: Scalar,
            time: Scalar
    ) -> ExponentialSmoothingState:
        """
        Performs one step of the exponential smoothing, returns the updated state of the agent.

        Parameters
        ----------
        state : ExponentialSmoothingState
            Previous agent state
        key : PRNGKey
            JAX random generator key
        distance : float
            Distance measurement
        time : float
            Current time

        Returns
        -------
        state : ExponentialSmoothingState
            Updated agent state
        """

        def initial_update() -> ExponentialSmoothingState:
            return ExponentialSmoothingState(
                level=distance,
                trend=0.0,
                time=time
            )

        def es_update() -> ExponentialSmoothingState:
            new_level = alpha * distance + (1 - alpha) * (state.level + state.trend)
            return ExponentialSmoothingState(
                level=new_level,
                trend=beta * (new_level - state.level) + (1 - beta) * state.trend,
                time=time
            )

        return jax.lax.cond(state.time == -1.0, initial_update, es_update)

    def sample(
            state: ExponentialSmoothingState,
            key: PRNGKey,
            time: Scalar
    ) -> tfd.Distribution:
        """
        Estimates distance distribution from current exponential smoothing agent state.

        Parameters
        ----------
        state : ExponentialSmoothingState
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

        return DeterministicQ(loc=state.level + (time - state.time) * state.trend)

    return BaseAgent(
        init=jax.jit(init),
        update=jax.jit(update),
        sample=jax.jit(sample)
    )


class ManagersContainer(BaseManagersContainer):
    def __init__(self, seed: int) -> None:
        super().__init__(seed, exponential_smoothing)
