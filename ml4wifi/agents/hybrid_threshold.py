from collections import deque
from functools import partial
from typing import Any, Tuple, Callable

import jax
import jax.numpy as jnp
from chex import dataclass, Scalar, PRNGKey

from ml4wifi.agents.base_managers_container import BaseAgent, BaseManagersContainer, select_ftmrate_mcs
from ml4wifi.agents.exponential_smoothing import exponential_smoothing
from ml4wifi.agents.kalman_filter import kalman_filter
from ml4wifi.agents.particle_filter import particle_filter
from ml4wifi.agents.thompson_sampling import thompson_sampling
from ml4wifi.utils.wifi_specs import wifi_modes_rates


@dataclass
class HybridThresholdState:
    ftmrate_state: Any
    backup_state: Any
    main_rate: jnp.int32
    backup_rate: jnp.int32
    success_history: deque


def hybrid_threshold(
        main_agent: BaseAgent,
        backup_agent: BaseAgent,
        select_ftmrate_mcs: Callable,
        history_length: int,
        threshold: float,
        backup_retransmissions: int,
        main_retransmissions: int
) -> BaseAgent:

    def init(key: PRNGKey) -> HybridThresholdState:
        """
        Initializes the agent state.

        Parameters
        ----------
        key : PRNGKey
            JAX random generator key

        Returns
        -------
        state : HybridThresholdState
            Initial agent state
        """

        ftmrate_key, backup_key = jax.random.split(key)

        return HybridThresholdState(
            ftmrate_state=main_agent.init(ftmrate_key),
            backup_state=backup_agent.init(backup_key),
            main_rate=0,
            backup_rate=0,
            success_history=deque(maxlen=history_length)
        )

    def update(
            state: HybridThresholdState,
            key: PRNGKey,
            distance: Scalar,
            measured: bool,
            tx_power: Scalar,
            time: Scalar,
            n_successful: jnp.int32,
            n_failed: jnp.int32,
            mode: jnp.int32
    ) -> HybridThresholdState:
        """
        Updates the state of the internal agents, selects the MCS and updates the state of the hybrid agent.
        Main and backup rates are updated based on the success history and the threshold.

        Parameters
        ----------
        state : HybridThresholdState
            Previous agent state
        key : PRNGKey
            JAX random generator key
        distance : float
            Distance between the transmitter and the receiver
        measured : bool
            Whether the distance is measured or not
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
        state : HybridThresholdState
            Updated agent state
        """

        ftmrate_key, backup_key = jax.random.split(key)

        _, ftmrate_state, ftmrate_rate = select_ftmrate_mcs(
            ftmrate_key, state.ftmrate_state, distance, measured, tx_power, time, n_successful, n_failed, mode
        )
        backup_state = backup_agent.update(state.backup_state, mode, n_successful, n_failed, time)
        backup_rate = backup_agent.sample(backup_state, backup_key, wifi_modes_rates)

        if n_successful > n_failed:
            n_all_successful = sum(res[0] for res in state.success_history)
            n_all_failed = sum(res[1] for res in state.success_history)

            if n_all_successful + n_all_failed > 0:
                tau = n_all_successful / (n_all_successful + n_all_failed)
            else:
                tau = 1.0

            if tau > threshold:
                main_rate, backup_rate = ftmrate_rate, backup_rate
            else:
                main_rate, backup_rate = backup_rate, ftmrate_rate
        else:
            main_rate, backup_rate = state.main_rate, state.backup_rate

        history = state.success_history
        history.append((n_successful, n_failed))

        return HybridThresholdState(
            ftmrate_state=ftmrate_state,
            backup_state=backup_state,
            main_rate=main_rate,
            backup_rate=backup_rate,
            success_history=history
        )

    def sample(state: HybridThresholdState,) -> jnp.int32:
        """
        Returns MCS according to the previous transmission plan and the success history.

        Parameters
        ----------
        state : HybridThresholdState
            Agent state

        Returns
        -------
        mcs : int
            Selected MCS
        """

        idx = 1

        for i in range(idx, min(len(state.success_history), main_retransmissions + idx)):
            if state.success_history[-i][0] > state.success_history[-i][1]:
                return state.main_rate

        idx += main_retransmissions

        for i in range(idx, min(len(state.success_history), backup_retransmissions + idx)):
            if state.success_history[-i][0] > state.success_history[-i][1]:
                return state.backup_rate

        return 0

    return BaseAgent(
        init=init,
        update=update,
        sample=sample
    )


def select_hybrid_mcs(
        key: PRNGKey,
        state: Any,
        distance: Scalar,
        measured: bool,
        tx_power: Scalar,
        time: Scalar,
        n_successful: jnp.int32,
        n_failed: jnp.int32,
        mode: jnp.int32,
        agent: BaseAgent
) -> Tuple[PRNGKey, Any, jnp.int32]:

    key, update_key, noise_key = jax.random.split(key, 3)
    state = agent.update(state, update_key, distance, measured, tx_power, time, n_successful, n_failed, mode)
    mcs = agent.sample(state)
    return key, state, mcs


class ManagersContainer(BaseManagersContainer):
    def __init__(
            self,
            seed: int,
            ftmrate_agent: str,
            history_length: int,
            threshold: float,
            backup_retransmissions: int,
            main_retransmissions: int
    ) -> None:

        if ftmrate_agent == 'es':
            ftmrate_agent = exponential_smoothing
        elif ftmrate_agent == 'kf':
            ftmrate_agent = kalman_filter
        elif ftmrate_agent == 'pf':
            ftmrate_agent = particle_filter
        else:
            raise ValueError(f'Unknown FTMRate agent: {ftmrate_agent}')

        self.key = jax.random.PRNGKey(seed)

        self.measurement_time = {}
        self.requested = {}

        ftmrate_agent = ftmrate_agent()
        backup_agent = thompson_sampling()
        select_ftmrate_mcs_fn = jax.jit(partial(select_ftmrate_mcs, agent=ftmrate_agent))

        self.agent = hybrid_threshold(
            ftmrate_agent, backup_agent, select_ftmrate_mcs_fn, history_length, threshold, backup_retransmissions, main_retransmissions
        )
        self.states = {}

        self.select_mcs = partial(select_hybrid_mcs, agent=self.agent)
