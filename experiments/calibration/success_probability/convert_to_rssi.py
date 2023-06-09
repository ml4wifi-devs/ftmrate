from argparse import ArgumentParser

import numpy as np
import pandas as pd


def distance_to_rssi(distance: float, exponent: float, shift: float) -> float:
    return shift - 10 * exponent * np.log10(distance)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--rssi_exponent', type=float, required=True)
    args.add_argument('--rssi_shift', type=float, required=True)
    args = args.parse_args()

    data = pd.read_csv('data.csv')
    data['rssi'] = data['distance'].apply(lambda d: distance_to_rssi(d, args.rssi_exponent, args.rssi_shift))
    data.to_csv('data.csv', index=False)
