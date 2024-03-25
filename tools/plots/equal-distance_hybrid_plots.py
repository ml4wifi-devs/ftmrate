import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import pandas as pd

from tools.plots.common import *


MAX_N_WIFI = 16
MANAGERS = {
    'TS': 'Thompson sampling',
    'MAB_KF': 'Hybrid w/ KF',
    'KF': 'FTMRate w/ KF',
}


def plot_results() -> None:
    colors = pl.cm.viridis(np.linspace(0., 1., 5))
    colors_map = {
        'MAB_KF': "tab:red",
        'TS': colors[1],
        'KF': colors[3],
    }

    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobility == 'Distance') & (df.distance == 1.)]
    df = df[df.nWifi == df.nWifiReal]

    for i, (manager, manager_name) in enumerate(MANAGERS.items()):
        mean, ci_low, ci_high = get_thr_ci(df[df.manager == manager], 'nWifiReal')

        if manager == 'Oracle':
            plt.plot(mean.index, mean, linestyle='--', c='gray', label=manager_name)
        else:
            plt.plot(mean.index, mean, marker=MARKERS[manager], markersize=2, label=manager_name, c=colors_map[manager])
            plt.fill_between(mean.index, ci_low, ci_high, alpha=0.3, color=colors_map[manager], linewidth=0.0)

    plt.xlim((0, MAX_N_WIFI))
    plt.xticks(range(0, MAX_N_WIFI + 1, 2))
    plt.xlabel('Number of stations')

    plt.ylim((0, 125))
    plt.ylabel('Aggregate throughput [Mb/s]')

    plt.grid()
    plt.legend(loc='lower left')


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    plt.rcParams['figure.figsize'] = (COLUMN_WIDTH, COLUMN_HIGHT / 2)

    plot_results()

    plt.savefig(f'equal-distance-hybrid-thr.pdf', bbox_inches='tight')
    plt.clf()
