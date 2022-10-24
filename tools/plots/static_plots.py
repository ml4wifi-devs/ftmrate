import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from plots.common import all_managers


TOOLS_DIR = os.path.join(os.path.expanduser("~"), 'tools')
DATA_FILE = os.path.join(TOOLS_DIR, 'outputs', 'all_results.csv')


def plot_results(distance):
    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobility == 'Static') & (df.distance == distance)]

    if len(df) == 0:
        return

    for manager in all_managers:
        data = df[df.manager == manager]

        measurements = data.groupby(['nWifiReal'])['throughput'].count()
        mean = data.groupby(['nWifiReal'])['throughput'].mean()
        std = data.groupby(['nWifiReal'])['throughput'].std()

        if measurements.ge(1).all():
            x = measurements.index
            plt.plot(x, mean, marker='o', markersize=4, label=manager.replace('_', ' '))

        if measurements.ge(2).all():
            alpha = 1 - 0.95
            z = t.ppf(1 - alpha / 2, measurements - 1)

            ci_low = mean - z * std / np.sqrt(measurements)
            ci_high = mean + z * std / np.sqrt(measurements)

            plt.fill_between(x, ci_low, ci_high, alpha=0.2)

    plt.ylim(bottom=0)
    plt.xlabel('Number of stations')
    plt.ylabel('Network throughput [Mb/s]')
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'static d{distance} mcs.svg', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    for distance in [0, 20, 40]:
        plot_results(distance)
