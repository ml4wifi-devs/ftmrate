import subprocess
from time import time, sleep

import numpy as np
import tensorflow_probability.substrates.numpy.bijectors as tfb

from ml4wifi.agents.kalman_filter import kalman_filter


# FTMRate constants
FTM_INTERVAL = 0.5
N_SAMPLES = 100

# Estimated from experiments
FTM_BIAS = 0.
FTM_COEFF = 1.

SNR_EXPONENT = 2.0
SNR_SHIFT = 20.

WIFI_MODES_SNRS = np.array([
    [10.613624240405125, 0.3536],
    [10.647249582547907, 0.3536],
    [10.660723984151614, 0.3536],
    [10.682584060100158, 0.3536],
    [11.151267538857537, 0.3536],
    [15.413200906170632, 0.3536],
    [16.735812667249125, 0.3536],
    [18.091175930406580, 0.3536]
])

WIFI_MODES_RATES = np.array([6., 9., 12., 18., 24., 36., 48., 54.])


def expected_rates(tx_power: float) -> tfb.Bijector:
    """
    Returns a bijector that transforms a distance to a distribution over expected rates.
    Distance is transformed in a following way:
        1. Logarithm                                       \
        2. Shift by -10 * SNR_EXPONENT / np.log(10.)        | log-distance channel model
        3. Shift by tx_power - SNR_SHIFT                   /
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
    tfp.bijectors.Bijector
        A distance to expected rates bijector.
    """

    return tfb.Chain([
        tfb.Scale(WIFI_MODES_RATES),
        tfb.NormalCDF()(tfb.Scale(WIFI_MODES_SNRS[:, 1])(tfb.Shift(-WIFI_MODES_SNRS[:, 0]))),
        tfb.Shift(tx_power - SNR_SHIFT)(tfb.Scale(-10 * SNR_EXPONENT / np.log(10.))(tfb.Log()))
    ])


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

    output = subprocess.check_output('./measure_distance.sh', universal_newlines=True)
    data = output.split("\n")

    status = int(data[1].split(" ")[-1])
    raw_distance = int(data[2].split(" ")[-2])

    if status != 0 or raw_distance < -1000:
        raise ValueError("FTM status code error")

    return (raw_distance - FTM_BIAS) / FTM_COEFF


if __name__ == '__main__':
    default_tx_power = 20.0

    kf = kalman_filter()
    state = kf.init(None)
    last_time = -np.inf

    while True:
        if time() - last_time > FTM_INTERVAL:
            while (distance := get_ftm_measurement()) == 0:
                pass

            print(distance)
            state = kf.update(state, None, distance, time())
            last_time = time()

        distance_dist = kf.sample(state, None, time())
        distance_dist = tfb.Softplus()(distance_dist)
        rates_mean = np.mean(expected_rates(default_tx_power)(distance_dist).sample(N_SAMPLES, None), axis=0)
        best_mcs = np.argmax(rates_mean)

        print(best_mcs)
        sleep(0.1)
