from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from ml4wifi.utils.wifi_specs import distance_noise


DEFAULT_INTERVAL = 0.5


@dataclass
class MeasurementState:
    distance: float
    time: float


@dataclass
class MeasurementManager:
    init: Callable
    update: Callable


def measurement_manager(interval: float = DEFAULT_INTERVAL) -> MeasurementManager:
    def init() -> MeasurementState:
        return MeasurementState(
            distance=np.inf,
            time=-np.inf
        )

    def update(state: MeasurementState, distance: float, time: float, key) -> Tuple[MeasurementState, bool]:
        if time - state.time >= interval:
            return MeasurementState(
                distance=np.abs(distance + distance_noise.sample(seed=key)),
                time=time
            ), True
        else:
            return state, False

    return MeasurementManager(
        init=init,
        update=update
    )