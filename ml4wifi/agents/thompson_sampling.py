from typing import Any, Tuple

import jax
import jax.numpy as jnp
from chex import dataclass, Array, Scalar, PRNGKey

from ml4wifi.agents.base_managers_container import BaseAgent, BaseManagersContainer
from ml4wifi.utils.measurement_manager import MeasurementState, MeasurementManager
from ml4wifi.utils.wifi_specs import wifi_modes_rates


@dataclass
class ThompsonSamplingState:
    alpha: Array
    beta: Array
    time: Scalar


def thompson_sampling(decay: Scalar = 1.0) -> BaseAgent:
    def init(key: PRNGKey) -> ThompsonSamplingState:
        """
        Returns the Thompson sampling agent initial state.

        Parameters
        ----------
        key : PRNGKey
            JAX random generator key

        Returns
        -------
        state : ThompsonSamplingState
            Initial Thompson sampling agent state
        """

        return ThompsonSamplingState(
            alpha=jnp.zeros(12),
            beta=jnp.zeros(12),
            time=0.0
        )

    def update(
            state: ThompsonSamplingState,
            action: jnp.int32,
            n_successful: jnp.int32,
            n_failed: jnp.int32,
            time: Scalar
    ) -> ThompsonSamplingState:
        """
        Performs one step of the Thompson sampling, returns the updated state of the agent.
        The agent uses exponential smoothing to update the success and failure rates.

        Parameters
        ----------
        state : ThompsonSamplingState
            Previous agent state
        action : int
            Previously selected MCS
        n_successful : int
            Number of successfully transmitted frames
        n_failed : int
            Number of failed transmitted frames
        time : float
            Current time

        Returns
        -------
        state : ThompsonSamplingState
            Updated Thompson sampling agent state
        """

        smoothing = jnp.exp(-decay * (time - state.time))

        return ThompsonSamplingState(
            alpha=(state.alpha * smoothing).at[action].add(n_successful),
            beta=(state.beta * smoothing).at[action].add(n_failed),
            time=time
        )

    def sample(
            state: ThompsonSamplingState,
            key: PRNGKey,
            context: Array
    ) -> jnp.int32:
        """
        Samples the best MCS based on the Thompson sampling algorithm.

        Parameters
        ----------
        state : ThompsonSamplingState
            Agent state
        key : PRNGKey
            JAX random generator key
        context : Array
            Context of the environment

        Returns
        -------
        mcs : int
            Selected MCS
        """

        success_prob = jax.random.beta(key, state.alpha + 1, state.beta + 1)
        return jnp.argmax(success_prob * context)

    return BaseAgent(
        init=jax.jit(init),
        update=jax.jit(update),
        sample=jax.jit(sample)
    )


def select_ts_mcs(
        key: PRNGKey,
        state: Any,
        m_state: MeasurementState,
        distance: Scalar,
        tx_power: Scalar,
        time: Scalar,
        n_successful: jnp.int32,
        n_failed: jnp.int32,
        mode: jnp.int32,
        agent: BaseAgent,
        measurements_manager: MeasurementManager
) -> Tuple[PRNGKey, Any, MeasurementState, jnp.int32]:
    key, sample_key = jax.random.split(key)
    state = agent.update(state, mode, n_successful, n_failed, time)
    mcs = agent.sample(state, sample_key, wifi_modes_rates)
    return key, state, m_state, mcs


class ManagersContainer(BaseManagersContainer):
    def __init__(self, seed: int) -> None:
        super().__init__(seed, thompson_sampling, select_ts_mcs)
