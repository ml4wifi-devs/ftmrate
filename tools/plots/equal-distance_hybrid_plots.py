import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import pandas as pd

from tools.plots.common import *


MAX_N_WIFI = 16

MANAGERS = {
    'THR_KF_06': r'Hybrid w/ KF, $\tau=0.6$',
    'THR_KF_07': r'Hybrid w/ KF, $\tau=0.7$',
    'Oracle': 'Oracle',
    'THR_KF_08': r'Hybrid w/ KF, $\tau=0.8$'
}


def plot_results() -> None:
    colors = pl.cm.viridis(np.linspace(0., 0.75, len(MANAGERS)))

    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobility == 'Distance') & (df.distance == 1.)]
    df = df[df.nWifi == df.nWifiReal]

    for i, (manager, manager_name) in enumerate(MANAGERS.items()):
        mean, ci_low, ci_high = get_thr_ci(df[df.manager == manager], 'nWifiReal')

        if manager == 'Oracle':
            plt.plot(mean.index, mean, linestyle='--', c='gray', label=manager_name)
        else:
            plt.plot(mean.index, mean, marker='o', markersize=2, label=manager_name, c=colors[i])
            plt.fill_between(mean.index, ci_low, ci_high, alpha=0.3, color=colors[i], linewidth=0.0)

    plt.xlim((0, MAX_N_WIFI))
    plt.xticks(range(0, MAX_N_WIFI + 1, 2))
    plt.xlabel('Number of stations')

    plt.ylim((0, 125))
    plt.ylabel('Aggregate throughput [Mb/s]')

    plt.grid()
    plt.legend()


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    plt.rcParams['figure.figsize'] = (COLUMN_WIDTH, 2 * COLUMN_WIDTH / (1 + np.sqrt(5)))

    plot_results()

    plt.savefig(f'equal-distance-hybrid-thr.pdf', bbox_inches='tight')
    plt.clf()
