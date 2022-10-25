import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from plots.common import all_managers


TOOLS_DIR = os.getenv('TOOLS_DIR', os.path.join(os.path.expanduser("~"), 'ftmrate/tools'))
DATA_FILE = os.path.join(TOOLS_DIR, 'outputs', 'all_results.csv')


def plot_results(velocity):
    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobility == 'Moving') & (df.velocity == velocity)]

    if len(df) == 0:
        return

    for manager in all_managers:
        data = df[df.manager == manager]

        measurements = data.groupby(['distance'])['throughput'].count()
        mean = data.groupby(['distance'])['throughput'].mean()
        std = data.groupby(['distance'])['throughput'].std()

        if measurements.ge(1).all():
            x = measurements.index
            plt.plot(x, mean, marker='o', markersize=3, label=manager.replace('_', ' '))

        if measurements.ge(2).all():
            alpha = 1 - 0.95
            z = t.ppf(1 - alpha / 2, measurements - 1)

            ci_low = mean - z * std / np.sqrt(measurements)
            ci_high = mean + z * std / np.sqrt(measurements)

            plt.fill_between(x, ci_low, ci_high, alpha=0.2)

    plt.ylim(bottom=0)
    plt.xlabel('Distance from AP [m]')
    plt.ylabel('Station throughput [Mb/s]')
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'moving v{velocity} thr.svg', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    for velocity in [1, 2]:
        plot_results(velocity)
