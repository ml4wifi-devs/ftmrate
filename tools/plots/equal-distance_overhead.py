import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import pandas as pd

from tools.plots.common import *


MAX_N_WIFI = 16


def plot_results(distance: float) -> None:
    colors = pl.cm.viridis(np.linspace(0., 1., len(ALL_MANAGERS) - 1))

    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobility == 'Distance') & (df.distance == distance)]
    df = df[df.nWifi == df.nWifiReal]

    oracle_mean, _, _ = get_thr_ci(df[df.manager == 'Oracle'], 'nWifiReal')

    for i, (manager, c) in enumerate(zip(['Minstrel', 'TS', 'OracleFTM'], colors[:2].tolist() + ['gray'])):
        mean, _, _ = get_thr_ci(df[df.manager == manager], 'nWifiReal')
        difference = (oracle_mean - mean) / oracle_mean * 100

        manager_name = ALL_MANAGERS.get(manager, 'FTM overhead')
        linestype = '--' if manager == 'OracleFTM' else '-'

        plt.plot(mean.index, difference, marker='o', markersize=2, label=manager_name, c=c, linestyle=linestype)

    plt.xlim((0, MAX_N_WIFI))
    plt.xticks(range(0, MAX_N_WIFI + 1, 2))
    plt.xlabel('Number of stations')

    plt.ylim((0, 100))
    plt.ylabel('Decrease in throughput [\%]')

    plt.legend()
    plt.grid()

    plt.savefig(f'equal-distance-overhead.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    plt.rcParams['figure.figsize'] = (COLUMN_WIDTH, 2 * COLUMN_WIDTH / (1 + np.sqrt(5)))

    plot_results(distance=1)
