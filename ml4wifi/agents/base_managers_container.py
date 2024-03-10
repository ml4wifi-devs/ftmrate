from functools import partial
from typing import Callable, Any, Tuple

import jax
from chex import dataclass, PRNGKey, Scalar
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ml4wifi.envs.ns3_ai_structures import Env, Act
from ml4wifi.utils.measurement_manager import measurement_manager, MeasurementState, MeasurementManager
from ml4wifi.utils.wifi_specs import expected_rates

tfb = tfp.bijectors


@dataclass
class BaseAgent:
    init: Callable
    update: Callable
    sample: Callable


def select_ftmrate_mcs(
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
        measurements_manager: MeasurementManager,
        n_samples: int = 1000
) -> Tuple[PRNGKey, Any, MeasurementState, jnp.int32]:

    key, update_key, sample_key, noise_key, rate_key = jax.random.split(key, 5)

    m_state, measured = measurements_manager.update(m_state, distance, time, noise_key)
    state = jax.lax.cond(
        measured,
        lambda: agent.update(state, update_key, m_state.distance, time),
        lambda: state
    )

    distance_dist = agent.sample(state, sample_key, time)
    distance_dist = tfb.Softplus()(distance_dist)
    rates_mean = jnp.mean(expected_rates(tx_power)(distance_dist).sample(n_samples, rate_key), axis=0)

    return key, state, m_state, jnp.argmax(rates_mean)


class BaseManagersContainer:
    def __init__(self, seed: int, agent: Callable, select_mcs: Callable = select_ftmrate_mcs) -> None:
        self.key = jax.random.PRNGKey(seed)

        self.agent = agent()
        self.states = {}

        self.measurements_manager = measurement_manager()
        self.measurements = {}

        self.select_mcs = jax.jit(partial(select_mcs, agent=self.agent, measurements_manager=self.measurements_manager))

    def do(self, env: Env, act: Act, ampdu: bool) -> Act:
        if env.type == 0:       # New station created
            act.station_id = sta_id = len(self.states)
            self.key, init_key = jax.random.split(self.key)
            self.states[sta_id] = self.agent.init(init_key)
            self.measurements[sta_id] = self.measurements_manager.init()

        elif env.type == 1:     # Sample new MCS
            sta_id = env.station_id

            # TODO: Verify if the filtration below is needed.
            if (ampdu and env.report_source == 2) or (not ampdu and env.report_source < 2):
                self.key, self.states[sta_id], self.measurements[sta_id], mode = \
                    self.select_mcs(
                        self.key, self.states[sta_id], self.measurements[sta_id],
                        env.distance, env.power, env.time, env.n_successful, env.n_failed, env.mode
                    )
            else:
                mode = env.mode

            act.mode = mode
            act.station_id = sta_id     # Only for check

        return act
