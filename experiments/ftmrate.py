import os
import re
import subprocess
from time import time, sleep

import numpy as np
import tensorflow_probability.substrates.numpy.bijectors as tfb

from kalman_filter import kalman_filter


# FTMRate constants
FTM_INTERVAL = 0.5
N_SAMPLES = 100

WIFI_MODES_RATES = np.array([6.5, 13., 19.5, 26., 39., 52, 58.5, 65., 13., 26., 39., 52., 78., 104., 117., 130.])

# Values estimated from experiments
KF_SENSOR_NOISE = 0.7455304265022278
SIGMA_X = 0.83820117
SIGMA_V = 0.3324591

# FTM correction (in centimeters)
FTM_BIAS = 0.
FTM_COEFF = 100.

RSSI_EXPONENT = 2.0
RSSI_SHIFT = 20.

WIFI_MODES_SNRS = np.array([
    [10.613624240405125, 0.3536],
    [10.647249582547907, 0.3536],
    [10.660723984151614, 0.3536],
    [10.682584060100158, 0.3536],
    [11.151267538857537, 0.3536],
    [15.413200906170632, 0.3536],
    [16.735812667249125, 0.3536],
    [18.091175930406580, 0.3536],
    [10.613624240405125, 0.3536],
    [10.647249582547907, 0.3536],
    [10.660723984151614, 0.3536],
    [10.682584060100158, 0.3536],
    [11.151267538857537, 0.3536],
    [15.413200906170632, 0.3536],
    [16.735812667249125, 0.3536],
    [18.091175930406580, 0.3536]
])


def expected_rates(tx_power: float) -> tfb.Bijector:
    """
    Returns a bijector that transforms a distance to a distribution over expected rates.
    Distance is transformed in a following way:
        1. Logarithm                                       \
        2. Scale by -10 * RSSI_EXPONENT / np.log(10.)        | log-distance channel model
        3. Shift by tx_power - RSSI_SHIFT                   /
        4. Shift by -WIFI_MODES_SNRS[:, 0]                 \
        5. Scale by -WIFI_MODES_SNRS[:, 1]                  | success probability
        6. Apply CDF of a standard normal distribution     /
        7. Scale by WIFI_MODES_RATES                       |  success probability to expected rate

    Parameters
    ----------
    tx_power : float
        The transmission power of the sender.

    Returns
    -------
    tfb.Bijector
        A distance to expected rates bijector.
    """

    return tfb.Chain([
        tfb.Scale(WIFI_MODES_RATES),
        tfb.NormalCDF()(tfb.Scale(WIFI_MODES_SNRS[:, 1])(tfb.Shift(-WIFI_MODES_SNRS[:, 0]))),
        tfb.Shift(tx_power - RSSI_SHIFT)(tfb.Scale(-10 * RSSI_EXPONENT / np.log(10.))(tfb.Log()))
    ])


def is_connected() -> bool:
    """
    Returns whether the device is connected to a Wi-Fi network.

    Returns
    -------
    bool
        True if the device is connected to a Wi-Fi network, False otherwise.
    """
    link=['iw', 'dev', 'wlp1s0', 'link']
    output = subprocess.check_output(link, universal_newlines=True)
    return 'not connected' not in output.lower()


def get_ftm_measurement() -> float:
    """
    Returns a single FTM measurement. The function calls a shell script that runs the FTM
    measurement and returns the raw distance in centimeters. The measurement is corrected
    by experimentally estimated bias and coefficient. If the measurement is invalid,
    raises a ValueError.

    Returns
    -------
    float
        The distance in meters.
    """

    output = subprocess.check_output('/home/opus/experiments/measure_distance.sh', universal_newlines=True)
    data = output.split("\n")

    status = int(data[1].split(" ")[-1])
    raw_distance = int(data[2].split(" ")[-2])

    if status != 0 or raw_distance < -1000:
        return -np.inf

    return (raw_distance - FTM_BIAS) / FTM_COEFF


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


if __name__ == '__main__':
    last_time = -np.inf
    set_mcs.last_mcs = -1
    default_tx_power = 20.0

    kf = kalman_filter(KF_SENSOR_NOISE, SIGMA_X, SIGMA_V)
    state = kf.init(timestamp=time())

    while True:
        if time() - last_time > FTM_INTERVAL and is_connected():
            while (distance := get_ftm_measurement()) == -np.inf:
                pass

            print(f'FTM distance: {distance:.2f} m')
            state = kf.update(state, distance, time())
            last_time = time()
        
        if is_connected():
            distance_dist = kf.sample(state, time())
            distance_dist = tfb.Softplus()(distance_dist)
            rates_mean = expected_rates(default_tx_power)(distance_dist).sample(N_SAMPLES).mean(axis=0)
            best_mcs = np.argmax(rates_mean)

    #        print(f'Best MCS: {best_mcs}')
            set_mcs(best_mcs)

            sleep(0.1)

        else:
            print(f'Out of range')
            sleep(0.5)
 
