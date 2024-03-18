from collections import deque
from functools import partial
from typing import Any, Tuple, Callable

import jax
import jax.numpy as jnp
from chex import dataclass, Scalar, Array, PRNGKey

from ml4wifi.agents.base_managers_container import BaseAgent, BaseManagersContainer, select_ftmrate_mcs
from ml4wifi.agents.exponential_smoothing import exponential_smoothing
from ml4wifi.agents.kalman_filter import kalman_filter
from ml4wifi.agents.particle_filter import particle_filter
from ml4wifi.agents.thompson_sampling import thompson_sampling
from ml4wifi.utils.measurement_manager import MeasurementState, MeasurementManager, measurement_manager
from ml4wifi.utils.wifi_specs import wifi_modes_rates

@dataclass
class MABState:
    alpha: Array
    beta: Array
    time: Scalar


def mab(decay: Scalar = 1.0) -> BaseAgent:
    def init(key: PRNGKey) -> MABState:
        """
        Returns the Thompson sampling agent initial state.

        Parameters
        ----------
        key : PRNGKey
            JAX random generator key

        Returns
        -------
        state : MABState
            Initial Thompson sampling agent state
        """

        return MABState(
            alpha=jnp.zeros(2),
            beta=jnp.zeros(2),
            time=0.0
        )

    def update(
            state: MABState,
            action: jnp.int32,
            n_successful: jnp.int32,
            n_failed: jnp.int32,
            time: Scalar
    ) -> MABState:
        """
        Performs one step of the Thompson sampling, returns the updated state of the agent.
        The agent uses exponential smoothing to update the success and failure rates.

        Parameters
        ----------
        state : MABState
            Previous agent state
        action : int
            Previously selected manager
        n_successful : int
            Number of successfully transmitted frames
        n_failed : int
            Number of failed transmitted frames
        time : float
            Current time

        Returns
        -------
        state : MABState
            Updated Thompson sampling agent state
        """

        smoothing = jnp.exp(-decay * (time - state.time))

        return MABState(
            alpha=(state.alpha * smoothing).at[action].add(n_successful),
            beta=(state.beta * smoothing).at[action].add(n_failed),
            time=time
        )

    def sample(
            state: MABState,
            key: PRNGKey,
    ) -> jnp.int32:
        """
        Samples the best MCS based on the Thompson sampling algorithm.

        Parameters
        ----------
        state : MABState
            Agent state
        key : PRNGKey
            JAX random generator key

        Returns
        -------
        manager_id : int
            Selected manager
        """

        success_prob = jax.random.beta(key, state.alpha + 1, state.beta + 1)
        return jnp.argmax(success_prob)

    return BaseAgent(
        init=jax.jit(init),
        update=jax.jit(update),
        sample=jax.jit(sample)
    )


@dataclass
class HybridMABState:
    mab_state: Any
    ftmrate_state: Any
    backup_state: Any
    last_manager: jnp.int8
    ftmrate_rate: jnp.int32


def hybrid_mab(
        mab_agent: BaseAgent,
        main_agent: BaseAgent,
        backup_agent: BaseAgent,
        select_ftmrate_mcs: Callable,
) -> BaseAgent:

    def init(key: PRNGKey) -> HybridMABState:
        """
        Initializes the agent state.

        Parameters
        ----------
        key : PRNGKey
            JAX random generator key

        Returns
        -------
        state : HybridMABState
            Initial agent state
        """

        mab_key, ftmrate_key, backup_key = jax.random.split(key, 3)

        return HybridMABState(
            mab_state=mab_agent.init(mab_key),
            ftmrate_state=main_agent.init(ftmrate_key),
            backup_state=backup_agent.init(backup_key),
            last_manager=0,
            ftmrate_rate=11
        )

    def update(
            state: HybridMABState,
            m_state: MeasurementState,
            key: PRNGKey,
            distance: Scalar,
            tx_power: Scalar,
            time: Scalar,
            n_successful: jnp.int32,
            n_failed: jnp.int32,
            mode: jnp.int32
    ) -> HybridMABState:
        """
        Updates the state of the internal agents, selects the MCS and updates the state of the hybrid agent.
        Main and backup rates are updated based on the success history and the threshold.

        Parameters
        ----------
        state : HybridMABState
            Previous agent state
        m_state : MeasurementState
            Measurement state
        key : PRNGKey
            JAX random generator key
        distance : float
            Distance between the transmitter and the receiver
        tx_power : float
            Transmission power
        time : float
            Current time
        n_successful : int
            Number of successfully transmitted frames
        n_failed : int
            Number of failed transmitted frames
        mode : int
            Previously selected MCS

        Returns
        -------
        state : HybridMABState
            Updated agent state
        """

        mab_key, ftmrate_key, backup_key = jax.random.split(key, 3)

        mab_state = mab_agent.update(state.mab_state, state.last_manager, n_successful, n_failed, time)
        _, ftmrate_state, _, ftmrate_rate = select_ftmrate_mcs(
            ftmrate_key, state.ftmrate_state, m_state, distance, tx_power, time, n_successful, n_failed, mode
        )
        backup_state = backup_agent.update(state.backup_state, mode, n_successful, n_failed, time)

        return HybridMABState(
            mab_state=mab_state,
            ftmrate_state=ftmrate_state,
            backup_state=backup_state,
            last_manager=state.last_manager,
            ftmrate_rate=ftmrate_rate
        )

    def sample(state: HybridMABState, key: PRNGKey) -> Tuple[jnp.int32, HybridMABState]:
        """
        Returns MCS according to the previous transmission plan and the success history.

        Parameters
        ----------
        state : HybridMABState
            Agent state
        key : PRNGKey
            JAX random generator key

        Returns
        -------
        mcs : int
            Selected MCS
        state : HybridMABState
            Updated agent state
        """

        mab_key, backup_key = jax.random.split(key)

        manager_action = mab_agent.sample(state.mab_state, mab_key)
        if manager_action == 0:
            mcs = state.ftmrate_rate
        else:
            mcs = backup_agent.sample(state.backup_state, backup_key, wifi_modes_rates)
        
        return mcs, HybridMABState(
                mab_state=state.mab_state,
                ftmrate_state=state.ftmrate_state,
                backup_state=state.backup_state,
                last_manager=manager_action,
                ftmrate_rate=state.ftmrate_rate
            )

    return BaseAgent(
        init=init,
        update=update,
        sample=sample
    )


def select_hybrid_mcs(
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

    key, update_key, sample_key, noise_key = jax.random.split(key, 4)
    state = agent.update(state, m_state, update_key, distance, tx_power, time, n_successful, n_failed, mode)
    mcs, state = agent.sample(state, sample_key)
    m_state, _ = measurements_manager.update(m_state, distance, time, noise_key)
    return key, state, m_state, mcs


class ManagersContainer(BaseManagersContainer):
    def __init__(
            self,
            seed: int,
            ftmrate_agent: str,
            mab_decay: Scalar
    ) -> None:
        
        mab_agent = mab(mab_decay)

        if ftmrate_agent == 'es':
            ftmrate_agent = exponential_smoothing
        elif ftmrate_agent == 'kf':
            ftmrate_agent = kalman_filter
        elif ftmrate_agent == 'pf':
            ftmrate_agent = particle_filter
        else:
            raise ValueError(f'Unknown FTMRate agent: {ftmrate_agent}')

        self.key = jax.random.PRNGKey(seed)

        self.measurements_manager = measurement_manager()
        self.measurements = {}

        ftmrate_agent = ftmrate_agent()
        backup_agent = thompson_sampling()
        select_ftmrate_mcs_fn = jax.jit(
            partial(select_ftmrate_mcs, agent=ftmrate_agent, measurements_manager=self.measurements_manager)
        )

        self.agent = hybrid_mab(
            mab_agent, ftmrate_agent, backup_agent, select_ftmrate_mcs_fn
        )
        self.states = {}

        self.select_mcs = partial(select_hybrid_mcs, agent=self.agent, measurements_manager=self.measurements_manager)
