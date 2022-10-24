from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from chex import dataclass, Array, Scalar, PRNGKey
from tensorflow_probability.substrates import jax as tfp

from ml4wifi.utils.wifi_specs import *
from ml4wifi.utils.measurement_manager import measurement_manager, MeasurementState

tfd = tfp.distributions
tfb = tfp.bijectors

# mean value based on ns-3 static-stations simulation, 1 STA, const MCS
FRAMES_PER_SECOND = 188

# If first measurement is in time t=0s, Kalman filter refuses to work
FIRST_MEASUREMENT_SHIFT = 0.001

N_SAMPLES = 1000


@dataclass
class FTMEstimates:
    distance_estimated: Scalar
    distance_uncertainty: Scalar
    distance_ci_low: Scalar
    distance_ci_high: Scalar
    snr_estimated: Scalar
    snr_uncertainty: Scalar
    snr_ci_low: Scalar
    snr_ci_high: Scalar
    rate_estimated: Scalar
    rate_uncertainty: Scalar
    mcs_estimated: jnp.int32


@dataclass
class SimulationParameters:
    confidence_level: Scalar
    measurement_interval: Scalar
    seed: jnp.int32
    simulation_time: Scalar
    start_position: Scalar
    velocity: Scalar
    total_frames: jnp.int32


@dataclass
class SimulationResults:
    time: Array
    distance: Dict
    snr: Dict
    mcs: Dict
    rate: Dict


@jax.jit
def ftmrate_log_distance(
        distance_dist: tfd.Distribution,
        confidence_level: jnp.float32,
        key: PRNGKey
) -> FTMEstimates:
    """
    Estimates distance, SNR and MCS from distance samples.

    Parameters
    ----------
    distance_dist : tfd.Distribution
        Distribution of distance estimation
    confidence_level : jnp.float32
        Confidence level of the estimations
    key : PRNGKey
        Seed

    Returns
    -------
    estimates : FTMEstimates
        Dataclass with all the estimations, uncertainties and the selected mcs
    """

    alpha = 1 - confidence_level

    snr_dist = distance_to_snr(tfb.Softplus()(distance_dist))
    rate_dist = expected_rates_log_distance(tfb.Softplus()(distance_dist))

    rate_estimated = rate_dist.quantile(0.5)
    mcs_estimated = jnp.argmax(rate_estimated)

    return FTMEstimates(
        distance_estimated=distance_dist.quantile(0.5),
        distance_uncertainty=0.0,
        distance_ci_low=distance_dist.quantile(alpha / 2),
        distance_ci_high=distance_dist.quantile(1 - alpha / 2),
        snr_estimated=snr_dist.quantile(0.5),
        snr_uncertainty=0.0,
        snr_ci_low=snr_dist.quantile(alpha / 2),
        snr_ci_high=snr_dist.quantile(1 - alpha / 2),
        rate_estimated=rate_estimated,
        rate_uncertainty=0.0,
        mcs_estimated=mcs_estimated,
    )


@jax.jit
def ftmrate_log_distance_monte_carlo(
        distance_dist: tfd.Distribution,
        confidence_level: jnp.float32,
        key: PRNGKey
) -> FTMEstimates:
    """
    Estimates distance, SNR and MCS from distance samples.

    Parameters
    ----------
    distance_dist : tfd.Distribution
        Distribution of distance estimation
    confidence_level : jnp.float32
        Confidence level of the estimations
    key : PRNGKey
        Seed

    Returns
    -------
    estimates : FTMEstimates
        Dataclass with all the estimations, uncertainties and the selected mcs
    """

    alpha = 1 - confidence_level

    distance_samples = distance_dist.sample(N_SAMPLES, key)
    distance_estimated = distance_samples.mean()
    distance_uncertainty = distance_samples.std()

    snr_samples = distance_to_snr(jnp.abs(distance_samples))
    snr_estimated = jnp.mean(snr_samples)
    snr_uncertainty = jnp.std(snr_samples)

    p_s_samples = jax.vmap(success_probability_log_distance)(snr_samples)
    rate_samples = p_s_samples * wifi_modes_rates
    rate_estimated = jnp.mean(rate_samples, axis=0)
    rate_uncertainty = jnp.std(rate_samples, axis=0)

    mcs_estimated = ideal_mcs_log_distance(distance_estimated)

    return FTMEstimates(
        distance_estimated=distance_estimated,
        distance_uncertainty=distance_uncertainty,
        distance_ci_low=jnp.quantile(distance_samples, alpha / 2),
        distance_ci_high=jnp.quantile(distance_samples, 1 - alpha / 2),
        snr_estimated=snr_estimated,
        snr_uncertainty=snr_uncertainty,
        snr_ci_low=jnp.quantile(snr_samples, alpha / 2),
        snr_ci_high=jnp.quantile(snr_samples, 1 - alpha / 2),
        rate_estimated=rate_estimated,
        rate_uncertainty=rate_uncertainty,
        mcs_estimated=mcs_estimated,
    )


@partial(jax.jit, static_argnames=['agent_fn', 'frames_total'])
def run_simulation(
        agent_fn: Callable,
        params: SimulationParameters,
        frames_total: jnp.int32
) -> SimulationResults:
    """
    Run one simple simulation of SNR estimation based on noisy distance measurements. The station moves away from
    the AP at constant velocity from some start position and receives noisy measurements at some time intervals.
    
    Parameters
    ----------
    agent_fn : callable
        Function that initializes the agent.
    params : SimulationParameters
        Parameters of the simulation.
    frames_total : int
        Total number of samples in the simulation.

    Returns
    -------
    results : SimulationResults
        Results of the simulation.
    """

    key = jax.random.PRNGKey(params.seed)
    key, init_key = jax.random.split(key)

    measurements_manager = measurement_manager(params.measurement_interval)
    agent = agent_fn()

    time = jnp.linspace(0.0, params.simulation_time, frames_total) + FIRST_MEASUREMENT_SHIFT
    true_distance = jnp.linspace(0.0, params.velocity * params.simulation_time, frames_total) + params.start_position

    distance = {
        'true': jnp.abs(true_distance),
        'measurement': jnp.empty(frames_total),
        'estimated': jnp.empty(frames_total),
        'uncertainty': jnp.zeros(frames_total),
        'ci_low': jnp.zeros(frames_total),
        'ci_high': jnp.zeros(frames_total),
    }

    snr = {
        'true': distance_to_snr(distance['true']),
        'estimated': jnp.empty(frames_total),
        'uncertainty': jnp.zeros(frames_total),
        'ci_low': jnp.zeros(frames_total),
        'ci_high': jnp.zeros(frames_total),
    }

    mcs = {
        'true': jax.vmap(ideal_mcs_log_distance)(distance['true']),
        'estimated': jnp.empty(frames_total),
    }

    rate = {
        'true': wifi_modes_rates[mcs['true']],
        'estimated': jnp.empty((frames_total, len(wifi_modes_rates))),
        'uncertainty': jnp.zeros((frames_total, len(wifi_modes_rates)))
    }

    def fori_fn(i: jnp.int32, carry: Tuple) -> Tuple:
        results, state, m_state, key = carry
        key, noise_key, update_key, sample_key, results_key = jax.random.split(key, 5)

        m_state, measured = measurements_manager.update(m_state, distance['true'][i], time[i], noise_key)
        state = jax.lax.cond(measured, lambda: agent.update(state, update_key, m_state.distance, time[i]), lambda: state)

        distance_distribution = agent.sample(state, sample_key, time[i])
        ftm_estimates = ftmrate_log_distance(distance_distribution, params.confidence_level, results_key)

        return save_estimates(ftm_estimates, m_state, i, *results), state, m_state, key

    init = ((distance, snr, rate, mcs), agent.init(init_key), measurements_manager.init(), key)
    (distance, snr, rate, mcs), *_ = jax.lax.fori_loop(0, frames_total, fori_fn, init)

    return SimulationResults(
        time=time,
        distance=distance,
        snr=snr,
        mcs=mcs,
        rate=rate
    )


@jax.jit
def save_estimates(
        ftm_estimates: FTMEstimates,
        m_state: MeasurementState,
        i: jnp.int32,
        distance: Array,
        snr: Array,
        rate: Array,
        mcs: Array
) -> Tuple:
    distance['measurement'] = distance['measurement'].at[i].set(m_state.distance)
    distance['estimated'] = distance['estimated'].at[i].set(ftm_estimates.distance_estimated)
    distance['uncertainty'] = distance['uncertainty'].at[i].set(ftm_estimates.distance_uncertainty)
    distance['ci_low'] = distance['ci_low'].at[i].set(ftm_estimates.distance_ci_low)
    distance['ci_high'] = distance['ci_high'].at[i].set(ftm_estimates.distance_ci_high)

    snr['estimated'] = snr['estimated'].at[i].set(ftm_estimates.snr_estimated)
    snr['uncertainty'] = snr['uncertainty'].at[i].set(ftm_estimates.snr_uncertainty)
    snr['ci_low'] = snr['ci_low'].at[i].set(ftm_estimates.snr_ci_low)
    snr['ci_high'] = snr['ci_high'].at[i].set(ftm_estimates.snr_ci_high)

    rate['estimated'] = rate['estimated'].at[i].set(ftm_estimates.rate_estimated)
    rate['uncertainty'] = rate['uncertainty'].at[i].set(ftm_estimates.rate_uncertainty)

    mcs['estimated'] = mcs['estimated'].at[i].set(ftm_estimates.mcs_estimated)

    return distance, snr, rate, mcs
