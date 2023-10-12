from functools import partial
from typing import Callable, Any, Tuple

import jax
import jax.numpy as jnp
from chex import dataclass, Scalar, PRNGKey
from tensorflow_probability.substrates import jax as tfp

from ml4wifi.envs.ns3_ai_structures import Env, Act
from ml4wifi.utils.wifi_specs import expected_rates

tfb = tfp.bijectors


FTM_INTERVAL = 0.5
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

        self.measurement_time = {}
        self.requested = {}

        self.select_mcs = jax.jit(partial(self.select_mcs, self.agent))

    @staticmethod
    def select_mcs(
            agent: BaseAgent,
            state: Any,
            key: PRNGKey,
            distance: Scalar,
            measured: jnp.bool_,
            tx_power: Scalar,
            time: Scalar
    ) -> Tuple[PRNGKey, Any, jnp.int32]:
        key, update_key, sample_key, rate_key = jax.random.split(key, 4)

        state = jax.lax.cond(measured, lambda: agent.update(state, update_key, distance, time), lambda: state)

        distance_dist = agent.sample(state, sample_key, time)
        distance_dist = tfb.Softplus()(distance_dist)
        rates_mean = jnp.mean(expected_rates(tx_power)(distance_dist).sample(N_SAMPLES, rate_key), axis=0)

        return key, state, jnp.argmax(rates_mean)
    
    def ftm_request_condition(self, env: Env, sta_id: int) -> bool:
        return env.time - self.measurement_time[sta_id] >= FTM_INTERVAL

    def do(self, env: Env, act: Act) -> Act:
        if env.type == 0:       # New station created
            self.key, init_key = jax.random.split(self.key)

            act.station_id = sta_id = len(self.states)
            self.states[sta_id] = self.agent.init(init_key)
            self.measurement_time[sta_id] = -1.
            self.requested[sta_id] = False

        elif env.type == 1:     # Sample new MCS
            sta_id = env.station_id

            self.key, self.states[sta_id], mode = self.select_mcs(
                self.states[sta_id], self.key,
                env.distance, env.ftm_completed, env.power, env.time
            )

            act.mode = mode
            act.station_id = sta_id     # Only for check

            if env.ftm_completed:
                self.measurement_time[sta_id] = env.time
                self.requested[sta_id] = False

            if self.requested[sta_id]:
                act.ftm_request = False
            elif self.ftm_request_condition(env, sta_id):
                self.requested[sta_id] = True
                act.ftm_request = True

        return act
