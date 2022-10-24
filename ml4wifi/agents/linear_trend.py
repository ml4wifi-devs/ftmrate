import jax
import tensorflow_probability.substrates.jax as tfp
from chex import dataclass, Scalar, PRNGKey

from ml4wifi.agents import BaseAgent, BaseManagersContainer
from ml4wifi.utils.distributions import DeterministicQ
from ml4wifi.params_fit import LT_ALPHA, LT_BETA

tfd = tfp.distributions


@dataclass
class LinearTrendState:
    level: Scalar
    trend: Scalar
    time: Scalar


def linear_trend(
        alpha: Scalar = LT_ALPHA,
        beta: Scalar = LT_BETA
) -> BaseAgent:

    def init(key: PRNGKey) -> LinearTrendState:
        """
        Returns the Local Linear Trend agent initial state.

        Parameters
        ----------
        key : PRNGKey
            JAX random generator key

        Returns
        -------
        state : LinearTrendState
            Initial Local Linear Trend state
        """

        return LinearTrendState(
            level=0.0,
            trend=0.0,
            time=-1.0
        )

    def update(
            state: LinearTrendState,
            key: PRNGKey,
            distance: Scalar,
            time: Scalar
    ) -> LinearTrendState:
        """
        Performs one step of the Local Linear Trend algorithm, returns the updated state of the agent.

        Parameters
        ----------
        state : LinearTrendState
            Previous agent state
        key : PRNGKey
            JAX random generator key
        distance : float
            Distance measurement
        time : float
            Current time

        Returns
        -------
        state : LinearTrendState
            Updated agent state
        """

        def initial_update() -> LinearTrendState:
            return LinearTrendState(
                level=distance,
                trend=0.0,
                time=time
            )

        def lt_update() -> LinearTrendState:
            new_level = alpha * distance + (1 - alpha) * (state.level + state.trend)
            return LinearTrendState(
                level=new_level,
                trend=beta * (new_level - state.level) + (1 - beta) * state.trend,
                time=time
            )

        return jax.lax.cond(state.time == -1.0, initial_update, lt_update)

    def sample(
            state: LinearTrendState,
            key: PRNGKey,
            time: Scalar
    ) -> tfd.Distribution:
        """
        Estimates distance distribution from current Local Linear Trend state.

        Parameters
        ----------
        state : LinearTrendState
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
        super().__init__(seed, linear_trend)
