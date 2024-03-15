import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import pandas as pd

from tools.plots.common import *


MIN_DISTANCE = 25
MAX_DISTANCE = 60


def plot_results(ax: plt.Axes, n_wifi: int) -> None:
    colors = pl.cm.viridis(np.linspace(0., 1., len(ALL_MANAGERS) - 1))

    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobility == 'Hidden') & (df.nWifiReal == n_wifi)]

    for i, (manager, manager_name) in enumerate(ALL_MANAGERS.items()):
        mean, ci_low, ci_high = get_thr_ci(df[df.manager == manager], 'distance')

        if manager == 'Oracle':
            ax.plot(mean.index, mean, linestyle='--', c='gray', label=manager_name)
        else:
            ax.plot(mean.index, mean, marker=MARKERS[manager], markersize=2, label=manager_name, c=colors[i])
            ax.fill_between(mean.index, ci_low, ci_high, alpha=0.3, color=colors[i], linewidth=0.0)

    ax.set_xlim((MIN_DISTANCE, MAX_DISTANCE))
    ax.set_ylim((0, 50))

    ax.set_ylabel('Aggregate throughput [Mb/s]')

    ax.grid()


if __name__ == '__main__':

    PLOT_PARAMS["figure.figsize"] = (COLUMN_WIDTH, COLUMN_HIGHT / 2)
    plt.rcParams.update(PLOT_PARAMS)
    fig, ax = plt.subplots()

    plot_results(ax, n_wifi=2)

    ax.set_xlabel(fr'Distance $\rho$ [m]')
    ax.legend()

    plt.savefig(f'hn-bar_thr_distance.pdf', bbox_inches='tight')
    plt.clf()
