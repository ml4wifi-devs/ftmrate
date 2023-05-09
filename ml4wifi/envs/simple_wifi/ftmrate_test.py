from argparse import ArgumentParser
from functools import partial
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from ml4wifi.agents.kalman_filter import kalman_filter
from ml4wifi.envs.simple_wifi.ftmrate_sim import *
from ml4wifi.utils.measurement_manager import DEFAULT_INTERVAL
from ml4wifi.utils.wifi_specs import wifi_modes_snrs


def run_test(
        params: SimulationParameters,
        agent: Callable,
        agent_name: str,
        *,
        confidence_level: float,
        warmup_time: float,
        plots_hide: bool,
        plots_ext: str,
        **_
) -> None:

    results = run_simulation(agent, params, params.total_frames)
    validate_results(results, confidence_level, warmup_time, agent_name)
    plot_results(results, agent_name, plots_ext, plots_hide)


def validate_results(
    results: SimulationResults,
    confidence_level: float,
    warmup_time: float,
    name: str
) -> None:

    def calculate_confidence(true, ci_low, ci_high):

        return np.logical_and(true >= ci_low, true <= ci_high).mean()

    print(f'\n{name} (confidence_level={confidence_level})')

    idx_filter = results.time > warmup_time

    distance_confidence = calculate_confidence(
        np.array(results.distance['true'])[idx_filter],
        np.array(results.distance['ci_low'])[idx_filter],
        np.array(results.distance['ci_high'])[idx_filter],
    )
    print(f'  Distance confidence: {distance_confidence}')

    snr_confidence = calculate_confidence(
        np.array(results.snr['true'])[idx_filter],
        np.array(results.snr['ci_low'])[idx_filter],
        np.array(results.snr['ci_high'])[idx_filter],
    )
    print(f'  SNR confidence:      {snr_confidence}')


def inf_to_val(array: np.ndarray, value: float = 2 ** 7) -> np.ndarray:
    return np.where(array == np.inf, value, array)


def plot_results(results: SimulationResults, name: str, extension: str = '.svg', hide: bool = False) -> None:

    # measurements plot
    plt.plot(results.time, results.distance['true'], label='True')
    plt.plot(results.time, results.distance['measurement'], label='Measurements')
    plt.ylabel('Distance [m]')
    plt.xlabel('Time [s]')
    plt.title(f'Input')
    plt.legend()
    plt.savefig(f'Input{extension}', bbox_inches='tight')
    plt.clf() if hide else plt.show()

    # distance estimation plot
    _, ax1 = plt.subplots()
    ax1.plot(results.time, results.distance['true'], label='True')
    ax1.plot(results.time, results.distance['estimated'], label='Estimated')
    ax1.fill_between(
        results.time,
        results.distance['ci_low'],
        results.distance['ci_high'],
        alpha=0.2,
        color='tab:orange'
    )

    if np.any(results.distance['uncertainty']):
        ax2 = ax1.twinx()
        ax2.set_ylabel('Distance uncertainty')
        ax2.set_ylim((0, 6))
        ax2.plot(results.time, results.distance['uncertainty'], label='std', zorder=0, c='g')
        ax2.legend(loc='lower right')

    ax1.set_ylabel('Distance [m]')
    ax1.set_xlabel('Time [s]')
    ax1.set_title(f'{name} distance estimation')
    ax1.legend()
    plt.savefig(f'{name} distance{extension}', bbox_inches='tight')
    plt.clf() if hide else plt.show()

    # SNR estimation plot
    _, ax1 = plt.subplots()
    ax1.plot(results.time, results.snr['true'], label='True', zorder=2)
    ax1.plot(results.time, inf_to_val(results.snr['estimated']), label='Estimated', zorder=1)
    ax1.fill_between(
        results.time,
        inf_to_val(results.snr['ci_low']),
        inf_to_val(results.snr['ci_high']),
        alpha=0.2,
        color='tab:orange'
    )

    if np.any(results.snr['uncertainty']):
        ax2 = ax1.twinx()
        ax2.set_ylabel('SNR uncertainty')
        ax2.set_ylim((0, 15))
        ax2.plot(results.time, inf_to_val(results.snr['uncertainty']), label='std', zorder=0, c='g')
        ax2.legend(loc='lower right')

    ax1.axhline(wifi_modes_snrs[-1], color='r', linestyle='--', label='Min SNR for MCS 11')
    ax1.set_ylabel('SNR [dBm]')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylim((0, 70))
    ax1.set_title(f'{name} SNR estimation')
    ax1.legend()
    plt.savefig(f'{name} snr{extension}', bbox_inches='tight')
    plt.clf() if hide else plt.show()

    # rate estimation plot
    plt.plot(results.time, results.rate['estimated'], label=[f'MCS {i}' for i in range(len(wifi_modes_snrs))])
    plt.ylabel('Data rate [Mb/s]')
    plt.xlabel('Time [s]')
    plt.title(f'{name} data rate estimation')
    plt.legend()
    plt.savefig(f'{name} rate{extension}', bbox_inches='tight')
    plt.clf() if hide else plt.show()

    # MCS selection plot
    plt.plot(results.time, results.mcs['true'], label='Ideal')
    plt.plot(results.time, results.mcs['estimated'], label='Estimated')
    plt.ylabel('MCS')
    plt.xlabel('Time [s]')
    plt.title(f'{name} MCS estimation')
    plt.legend()
    plt.savefig(f'{name} mcs{extension}', bbox_inches='tight')
    plt.clf() if hide else plt.show()


if __name__ == '__main__':
    args = ArgumentParser()

    # simulation parameters
    args.add_argument('--measurement_interval', default=DEFAULT_INTERVAL, type=float)
    args.add_argument('--seed', default=42, type=int)
    args.add_argument('--simulation_time', default=25.0, type=float)
    args.add_argument('--start_position', default=0.0, type=float)
    args.add_argument('--velocity', default=2.0, type=float)
    args.add_argument('--confidence_level', default=0.95, type=float)
    args.add_argument('--warmup_time', default=5.0, type=float)

    # plots parameters
    args.add_argument('--plots_hide', action='store_true', default=False)
    args.add_argument('--plots_ext', default='.svg', type=str)

    args = args.parse_args()

    assert 0 < args.confidence_level < 1, 'Confidence level should be chosen from (0, 1) interval'

    params = SimulationParameters(
        confidence_level=args.confidence_level,
        measurement_interval=args.measurement_interval,
        seed=args.seed,
        simulation_time=args.simulation_time,
        start_position=args.start_position,
        velocity=args.velocity,
        total_frames=int(FRAMES_PER_SECOND * args.simulation_time)
    )

    run_test(params, kalman_filter, 'KFD', **vars(args))
