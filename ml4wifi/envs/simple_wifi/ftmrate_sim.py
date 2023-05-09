from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Tuple

import numpy as np
from tensorflow_probability.substrates import numpy as tfp

from ml4wifi.utils.wifi_specs import *
from ml4wifi.utils.measurement_manager import measurement_manager, MeasurementState

tfd = tfp.distributions
tfb = tfp.bijectors

# mean value based on ns-3 equal distance simulation scenario, 1 STA, const MCS
FRAMES_PER_SECOND = 188

# If first measurement is in time t=0s, Kalman filter refuses to work
FIRST_MEASUREMENT_SHIFT = 0.001

N_SAMPLES = 1000


@dataclass
class FTMEstimates:
    distance_estimated: float
    distance_uncertainty: float
    distance_ci_low: float
    distance_ci_high: float
    snr_estimated: float
    snr_uncertainty: float
    snr_ci_low: float
    snr_ci_high: float
    rate_estimated: float
    rate_uncertainty: float
    mcs_estimated: np.int32


@dataclass
class SimulationParameters:
    confidence_level: float
    measurement_interval: float
    seed: np.int32
    simulation_time: float
    start_position: float
    velocity: float
    total_frames: np.int32


@dataclass
class SimulationResults:
    time: np.ndarray
    distance: Dict
    snr: Dict
    mcs: Dict
    rate: Dict


def ftmrate_log_distance(
        distance_dist: tfd.Distribution,
        confidence_level: np.float32,
        key
) -> FTMEstimates:
    """
    Estimates distance, SNR and MCS from distance samples.

    Parameters
    ----------
    distance_dist : tfd.Distribution
        Distribution of distance estimation
    confidence_level : np.float32
        Confidence level of the estimations
    key 
        Seed

    Returns
    -------
    estimates : FTMEstimates
        Dataclass with all the estimations, uncertainties and the selected mcs
    """

    alpha = 1 - confidence_level

    snr_dist = distance_to_snr(tfb.Softplus()(distance_dist))
    rate_dist = expected_rates_log_distance(DEFAULT_TX_POWER)(tfb.Softplus()(distance_dist))

    rate_estimated = rate_dist.quantile(0.5)
    mcs_estimated = np.argmax(rate_estimated)

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


def run_simulation(
        agent_fn: Callable,
        params: SimulationParameters,
        frames_total: np.int32
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

    measurements_manager = measurement_manager(params.measurement_interval)
    agent = agent_fn()

    time = np.linspace(0.0, params.simulation_time, frames_total) + FIRST_MEASUREMENT_SHIFT
    true_distance = np.linspace(0.0, params.velocity * params.simulation_time, frames_total) + params.start_position

    distance = {
        'true': np.abs(true_distance),
        'measurement': np.empty(frames_total),
        'estimated': np.empty(frames_total),
        'uncertainty': np.zeros(frames_total),
        'ci_low': np.zeros(frames_total),
        'ci_high': np.zeros(frames_total),
    }

    snr = {
        'true': distance_to_snr(distance['true']),
        'estimated': np.empty(frames_total),
        'uncertainty': np.zeros(frames_total),
        'ci_low': np.zeros(frames_total),
        'ci_high': np.zeros(frames_total),
    }

    mcs = {
        'true': np.array([partial(ideal_mcs_log_distance, tx_power=DEFAULT_TX_POWER)(x) for x in distance['true']]),
        'estimated': np.empty(frames_total),
    }

    rate = {
        'true': wifi_modes_rates[mcs['true']],
        'estimated': np.empty((frames_total, len(wifi_modes_rates))),
        'uncertainty': np.zeros((frames_total, len(wifi_modes_rates)))
    }

    state = agent.init(None)
    m_state = measurements_manager.init()

    for i in range(frames_total):
        m_state, measured = measurements_manager.update(m_state, distance['true'][i], time[i], None)
        state = agent.update(state, None, m_state.distance, time[i]) if measured else state

        distance_distribution = agent.sample(state, None, time[i])
        ftm_estimates = ftmrate_log_distance(distance_distribution, params.confidence_level, None)

        save_estimates(ftm_estimates, m_state, i, distance, snr, rate, mcs)

    return SimulationResults(
        time=time,
        distance=distance,
        snr=snr,
        mcs=mcs,
        rate=rate
    )


def save_estimates(
        ftm_estimates: FTMEstimates,
        m_state: MeasurementState,
        i: np.int32,
        distance: np.ndarray,
        snr: np.ndarray,
        rate: np.ndarray,
        mcs: np.ndarray
) -> Tuple:
    distance['measurement'][i] = m_state.distance
    distance['estimated'][i] = ftm_estimates.distance_estimated
    distance['uncertainty'][i] = ftm_estimates.distance_uncertainty
    distance['ci_low'][i] = ftm_estimates.distance_ci_low
    distance['ci_high'][i] = ftm_estimates.distance_ci_high

    snr['estimated'][i] = ftm_estimates.snr_estimated
    snr['uncertainty'][i] = ftm_estimates.snr_uncertainty
    snr['ci_low'][i] = ftm_estimates.snr_ci_low
    snr['ci_high'][i] = ftm_estimates.snr_ci_high

    rate['estimated'][i] = ftm_estimates.rate_estimated
    rate['uncertainty'][i] = ftm_estimates.rate_uncertainty

    mcs['estimated'][i] = ftm_estimates.mcs_estimated

    return distance, snr, rate, mcs
