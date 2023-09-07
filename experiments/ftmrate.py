import os
import re
import signal
import sys
import subprocess
from time import time, sleep

import numpy as np
import tensorflow_probability.substrates.numpy.bijectors as tfb

from kalman_filter import kalman_filter


# Configuration
HOME_DIR = '/home/opus'

# FTMRate constants
FTM_INTERVAL = 0.5
N_SAMPLES = 100
WIFI_MODES_RATES = np.array([6.5, 13., 19.5, 26., 39., 52, 58.5, 65., 13., 26., 39., 52., 78., 104., 117., 130.])

# Values estimated from experiments
KF_SENSOR_NOISE = 2.29184 
SIGMA_X = 0.83820117
SIGMA_V = 0.3324591

# FTM correction (in centimeters)
FTM_BIAS = 510.77923
FTM_COEFF = 152.71236

RSSI_EXPONENT = 1.31177 
RSSI_SHIFT = 67.81936

WIFI_MODES_RSSIS = np.array([
    [-97.58278408366853, 9.999999999999893],
    [-98.54009444684868, 9.99999999999753],
    [-97.16405000702329, 9.99999999999969],
    [-82.76217672125235, 4.328609656855655],
    [-77.14358221792705, 0.9005603407290622],
    [-73.74822068323002, 1.1049948291659586],
    [-72.19748670341089, 0.951968687734157],
    [-61.19181865110476, 9.999999999999893],
    [-99.28086000720667, 9.999999999985102],
    [-83.46493223270045, 4.74762668672445],
    [-76.52596977862262, 2.066187716763535],
    [-73.1854974094934, 2.566245872541818],
    [-57.89706532108574, 9.999999999999998],
    [-59.92445513734765, 9.999999999999915],
    [-58.81721598406903, 9.999999999999952],
    [-56.912850080258735, 9.999999999999526],
])

# Minimum distance that can be measured by FTM
MIN_DISTANCE = 2.0


def expected_rates(delta_tx_power: float = 0.) -> tfb.Bijector:
    """
    Returns a bijector that transforms a distance to a distribution over expected rates.
    Distance is transformed in a following way:
        1. Logarithm                                       \
        2. Scale by -10 * RSSI_EXPONENT / np.log(10.)       | log-distance channel model
        3. Shift by delta_tx_power - RSSI_SHIFT                  /
        4. Shift by -WIFI_MODES_RSSIS[:, 0]                 \
        5. Scale by -WIFI_MODES_RSSIS[:, 1]                  | success probability
        6. Apply CDF of a standard normal distribution     /
        7. Scale by WIFI_MODES_RATES                       |  success probability to expected rate

    Parameters
    ----------
    delta_tx_power : float
        Difference between the default and the current tx power.

    Returns
    -------
    tfb.Bijector
        A distance to expected rates bijector.
    """

    return tfb.Chain([
        tfb.Scale(WIFI_MODES_RATES),
        tfb.NormalCDF(),
        tfb.Scale(WIFI_MODES_RSSIS[:, 1]),
        tfb.Shift(-WIFI_MODES_RSSIS[:, 0]),
        tfb.Shift(delta_tx_power - RSSI_SHIFT),
        tfb.Scale(-10 * RSSI_EXPONENT / np.log(10.)),
        tfb.Log()
    ])


def is_connected() -> bool:
    """
    Returns whether the device is connected to a Wi-Fi network.

    Returns
    -------
    bool
        True if the device is connected to a Wi-Fi network, False otherwise.
    """
    link = ['iw', 'dev', 'wlp1s0', 'link']
    output = subprocess.check_output(link, universal_newlines=True)
    return 'not connected' not in output.lower()


def get_ftm_measurement(min_distance: float) -> float:
    """
    Returns a single FTM measurement. The function calls a shell script that runs the FTM
    measurement and returns the raw distance in centimeters. The measurement is corrected
    by experimentally estimated bias and coefficient. If the measurement is invalid,
    raises a ValueError.

    Parameters
    ----------
    min_distance : float
        The minimum distance to measure.
    Returns
    -------
    float
        The distance in meters.
    """

    output = subprocess.check_output([f'{HOME_DIR}/ftmrate_internal/experiments/measure_distance.sh', HOME_DIR], universal_newlines=True)
    data = output.split("\n")

    status = int(data[1].split(" ")[-1])
    raw_distance = int(data[2].split(" ")[-2])

    if status != 0 or raw_distance < -1000:
        return -np.inf

    distance = (raw_distance - FTM_BIAS) / FTM_COEFF
    return max(distance, min_distance)


def set_mcs(mcs: int) -> None:
    """
    Sets the MCS of the Wi-Fi interface by writing appropriate mask to the `rate_scale_table` file.

    Parameters
    ----------
    mcs : int
        The MCS to set.
    """

    if set_mcs.last_mcs == mcs:
        return

    RATE_MCS_ANT_MSK = 0x0c000
    RATE_MCS_HT_MSK  = 0x00100

    monitor_tx_rate = 0x0
    monitor_tx_rate |= RATE_MCS_HT_MSK
    monitor_tx_rate |= RATE_MCS_ANT_MSK
    monitor_tx_rate |= mcs

    mask = '0x{:05x}'.format(monitor_tx_rate)
    set_mcs.last_mcs = mcs
    print(mcs)
    
    path = '/sys/kernel/debug/ieee80211/'
    path += os.listdir(path)[0] + '/'
    path += [f for f in os.listdir(path) if re.match(r'.*:wl.*', f)][0] + '/stations/'
    path += os.listdir(path)[0] + '/rate_scale_table'

    os.system(f'echo {mask} | tee {path}')


def signal_handler(sig, frame):
    """
    Handles SIGTERM signal by exiting the program.
    """

    set_mcs(0)
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, signal_handler)

    last_time = -np.inf
    set_mcs.last_mcs = -1

    kf = kalman_filter(KF_SENSOR_NOISE, SIGMA_X, SIGMA_V)
    state = kf.init(timestamp=time())

    while True:
        if time() - last_time > FTM_INTERVAL and is_connected():
            while (distance := get_ftm_measurement(MIN_DISTANCE)) == -np.inf:
                pass

            print(f'FTM distance: {distance:.2f} m')
            state = kf.update(state, distance, time())
            last_time = time()

        if is_connected():
            distance_dist = kf.sample(state, time())
            distance_dist = tfb.Softplus()(distance_dist)
            rates_mean = expected_rates()(distance_dist).sample(N_SAMPLES).mean(axis=0)
            best_mcs = np.argmax(rates_mean)

            set_mcs(best_mcs)
            sleep(0.1)
        else:
            print(f'Out of range')
            sleep(0.5)
