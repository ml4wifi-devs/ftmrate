import subprocess

import numpy as np
import tensorflow_probability.substrates.numpy as tfp
from time import time

from ml4wifi.agents.kalman_filter import kalman_filter
from ml4wifi.utils.wifi_specs import expected_rates

tfb = tfp.bijectors


FTM_BIAS = 0.
FTM_COEFF = 1.
TX_POWER = 16.0206

N_SAMPLES = 100


def parse_ftm_measurement(data):

    status = int(data[1].split(" ")[-1])
    raw_distance = int(data[2].split(" ")[-2])

    if status != 0 or raw_distance < -1000:
        raise ValueError("FTM status code error")

    return (raw_distance - FTM_BIAS) / FTM_COEFF


def get_ftm_measurement():
    output = subprocess.check_output('measure_distance.sh', shell=True, universal_newlines=True)
    data = output.decode('utf-8').split("\n")
    return parse_ftm_measurement(data)


if __name__ == '__main__':
    kf = kalman_filter()
    state = kf.init()
    current_time = time()

    while True:
        distance = get_ftm_measurement()

        if distance > 0:
            state = kf.update(state, None, distance, time())
            distance_dist = kf.sample(state, None, time())

            distance_dist = tfb.Softplus()(distance_dist)
            rates_mean = np.mean(expected_rates(TX_POWER)(distance_dist).sample(N_SAMPLES, None), axis=0)
            best_mcs = np.argmax(rates_mean)

            current_time = time()

            print(best_mcs)
