from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from chex import dataclass, Scalar, PRNGKey

from ml4wifi.utils.wifi_specs import distance_noise


DEFAULT_INTERVAL = 0.5


@dataclass
class MeasurementState:
    distance: Scalar
    time: Scalar


@dataclass
class MeasurementManager:
    init: Callable
    update: Callable


def measurement_manager(interval: Scalar = DEFAULT_INTERVAL) -> MeasurementManager:
    def init() -> MeasurementState:
        return MeasurementState(
            distance=jnp.inf,
            time=-jnp.inf
        )

    def update(state: MeasurementState, distance: Scalar, time: Scalar, key: PRNGKey) -> Tuple[MeasurementState, jnp.bool_]:
        return jax.lax.cond(
            time - state.time >= interval,
            lambda: (MeasurementState(
                distance=jnp.abs(distance + distance_noise.sample(seed=key)),
                time=time
            ), True),
            lambda: (state, False)
        )

    return MeasurementManager(
        init=jax.jit(init),
        update=jax.jit(update)
    )