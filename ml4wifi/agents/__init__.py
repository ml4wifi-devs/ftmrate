from functools import partial
from typing import Callable, Any, Tuple

import numpy as np
import tensorflow_probability as tfp

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
        self.agent = agent()
        self.states = {}

        self.measurements_manager = measurement_manager()
        self.measurements = {}

        self.select_mcs = partial(
            self.select_mcs,
            agent=self.agent,
            measurements_manager=self.measurements_manager
        )

    @staticmethod
    def select_mcs(
            key,
            state: Any,
            m_state: MeasurementState,
            distance: float,
            tx_power: float,
            time: float,
            agent: BaseAgent,
            measurements_manager: MeasurementManager
    ) -> Tuple[Any, Any, MeasurementState, np.int32]:

        m_state, measured = measurements_manager.update(m_state, distance, time, None)
        state = agent.update(state, None, m_state.distance, time) if measured else state

        distance_dist = agent.sample(state, None, time)
        distance_dist = tfb.Softplus()(distance_dist)
        rates_mean = np.mean(expected_rates(tx_power)(distance_dist).sample(N_SAMPLES, None), axis=0)

        return key, state, m_state, np.argmax(rates_mean)

    def do(self, env: Env, act: Act) -> Act:
        if env.type == 0:       # New station created
            act.station_id = sta_id = len(self.states)
            self.states[sta_id] = self.agent.init(None)
            self.measurements[sta_id] = self.measurements_manager.init()

        elif env.type == 1:     # Sample new MCS
            sta_id = env.station_id
            self.key, self.states[sta_id], self.measurements[sta_id], mode = \
                self.select_mcs(
                    None,
                    self.states[sta_id],
                    self.measurements[sta_id],
                    env.distance,
                    env.power,
                    env.time
                )

            act.mode = mode
            act.station_id = sta_id     # Only for check

        return act
