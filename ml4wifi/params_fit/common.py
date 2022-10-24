import os
from functools import partial
from typing import Tuple

import jax
import pandas as pd
from chex import PRNGKey, Scalar
from jax import numpy as jnp

from ml4wifi.utils.measurement_manager import measurement_manager


@partial(jax.jit, static_argnames=['frames_total'])
def generate_rwpm(
        key: PRNGKey,
        time_total: Scalar,
        measurement_interval: Scalar,
        frames_total: jnp.int32,
        area: Scalar = 40.0,
        speed: Scalar = 1.4,
        pause: Scalar = 20.0
) -> Tuple:
    """
    Generates noisy distance measurements from RWPM movement (square with dimensions `area` x `area`).
    https://en.wikipedia.org/wiki/Random_waypoint_model
    """

    measurements_manager = measurement_manager(measurement_interval)
    m_state = measurements_manager.init()

    distance_true, distance_measurement, distance_measured = jnp.empty((3, frames_total))
    time = jnp.linspace(0.0, time_total, frames_total)

    area = area * jnp.sqrt(2)
    time_pause, time_move, d_v = 0.0, 0.0, 0.0

    d_key, key = jax.random.split(key)
    d = jax.random.uniform(d_key, minval=0.0, maxval=area)

    def fori_fn(i: jnp.int32, carry: Tuple) -> Tuple:
        distance_true, distance_measurement, distance_measured, m_state, time_pause, time_move, d, d_v, key = carry
        speed_key, pause_key, d_key, noise_key, key = jax.random.split(key, 5)

        d = jnp.where(
            (time_pause * frames_total <= time_total * i) & (time_total * i < time_move * frames_total),
            d + d_v * time_total / frames_total,
            d
        )

        def new_point(carry: Tuple) -> Tuple:
            time_pause, time_move, d_v = carry

            time_pause = time_move + jax.random.uniform(pause_key, minval=0.0, maxval=pause)
            d_next = jax.random.uniform(d_key, minval=0.0, maxval=area)
            d_v = jnp.sign(d_next - d) * jax.random.uniform(speed_key, minval=0.0, maxval=speed)
            time_move = time_pause + jnp.abs((d_next - d) / d_v)

            return time_pause, time_move, d_v

        time_pause, time_move, d_v = jax.lax.cond(
            time_total * i >= time_move * frames_total,
            new_point,
            lambda carry: carry,
            (time_pause, time_move, d_v)
        )

        distance_true = distance_true.at[i].set(d)
        m_state, measured = measurements_manager.update(m_state, d, time[i], noise_key)
        distance_measurement = distance_measurement.at[i].set(m_state.distance)
        distance_measured = distance_measured.at[i].set(measured)

        return distance_true, distance_measurement, distance_measured, m_state, time_pause, time_move, d, d_v, key

    init = (distance_true, distance_measurement, distance_measured, m_state, time_pause, time_move, d, d_v, key)
    distance_true, distance_measurement, distance_measured, *_ = jax.lax.fori_loop(0, frames_total, fori_fn, init)

    return distance_true, distance_measurement, distance_measured, time


@partial(jax.jit, static_argnames=['frames_total'])
def generate_data(
        key: PRNGKey,
        initial_pos: Scalar,
        time_total: Scalar,
        velocity: Scalar,
        measurement_interval: Scalar,
        frames_total: jnp.int32
) -> Tuple:
    """
    Generates noisy distance measurements from uniform movement.
    """

    measurements_manager = measurement_manager(measurement_interval)
    m_state = measurements_manager.init()

    distance_true = jnp.linspace(0.0, velocity * time_total, frames_total) + initial_pos
    time = jnp.linspace(0.0, time_total, frames_total)

    distance_measurement = jnp.empty(frames_total)
    distance_measured = jnp.empty(frames_total, dtype=jnp.bool_)

    def fori_fn(i: jnp.int32, carry: Tuple) -> Tuple:
        distance_measurement, distance_measured, m_state, key = carry
        key, noise_key = jax.random.split(key)

        m_state, measured = measurements_manager.update(m_state, distance_true[i], time[i], noise_key)

        distance_measurement = distance_measurement.at[i].set(m_state.distance)
        distance_measured = distance_measured.at[i].set(measured)

        return distance_measurement, distance_measured, m_state, key

    init = (distance_measurement, distance_measured, m_state, key)
    distance_measurement, distance_measured, *_ = jax.lax.fori_loop(0, frames_total, fori_fn, init)

    return distance_true, distance_measurement, distance_measured, time


CSV_FILES_DIR = 'csv_files'
PARAMS_HEADER = 'mcs,loc,scale,skewness,tailweight,lt_alpha,lt_beta,kf_sensor_noise,sigma_x,sigma_v\n'


def load_parameters_file(name: str = 'parameters.csv'):
    path = os.path.join(CSV_FILES_DIR, name)

    if not os.path.isfile(path):
        file = open(path, 'w')
        file.write(PARAMS_HEADER)
        for mcs in range(12):
            file.write(f'{mcs}' + ',' * 9 + '\n')
        file.close()
    
    return pd.read_csv(path)
