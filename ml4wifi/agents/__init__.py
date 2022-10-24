from functools import partial
from typing import Callable, Any, Tuple

import jax
import jax.numpy as jnp
from chex import dataclass, Scalar, PRNGKey
from tensorflow_probability.substrates import jax as tfp

from ml4wifi.envs.ns3_ai_structures import Env, Act
from ml4wifi.utils.measurement_manager import *
from ml4wifi.utils.wifi_specs import expected_rates

tfb = tfp.bijectors


N_SAMPLES = 1000


@dataclass
class BaseAgent:
    init: Callable
    update: Callable
    sample: Callable


class BaseManagersContainer:
    def __init__(self, seed: int, agent: Callable) -> None:
        self.key = jax.random.PRNGKey(seed)

        self.agent = agent()
        self.states = {}

        self.measurements_manager = measurement_manager()
        self.measurements = {}

        self.select_mcs = jax.jit(partial(
            self.select_mcs,
            agent=self.agent,
            measurements_manager=self.measurements_manager
        ))

    @staticmethod
    def select_mcs(
            key: PRNGKey,
            state: Any,
            m_state: MeasurementState,
            distance: Scalar,
            time: Scalar,
            agent: BaseAgent,
            measurements_manager: MeasurementManager
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
        rates_mean = jnp.mean(expected_rates(distance_dist).sample(N_SAMPLES, rate_key), axis=0)

        return key, state, m_state, jnp.argmax(rates_mean)

    def do(self, env: Env, act: Act) -> Act:
        if env.type == 0:       # New station created
            act.station_id = sta_id = len(self.states)
            self.key, init_key = jax.random.split(self.key)
            self.states[sta_id] = self.agent.init(init_key)
            self.measurements[sta_id] = self.measurements_manager.init()

        elif env.type == 1:     # Sample new MCS
            sta_id = env.station_id
            self.key, self.states[sta_id], self.measurements[sta_id], mode = \
                self.select_mcs(self.key, self.states[sta_id], self.measurements[sta_id], env.distance, env.time)

            act.mode = mode
            act.station_id = sta_id     # Only for check

        return act
