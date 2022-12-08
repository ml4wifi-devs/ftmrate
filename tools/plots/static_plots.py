import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import pandas as pd

from tools.plots.common import *


MAX_N_WIFI = 30


def plot_results(ax: plt.Axes, distance: float) -> None:
    colors = pl.cm.viridis(np.linspace(0., 1., len(ALL_MANAGERS) - 1))

    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobility == 'Static') & (df.distance == distance) & (df.nWifiReal <= MAX_N_WIFI)]

    for i, (manager, manager_name) in enumerate(ALL_MANAGERS.items()):
        mean, ci_low, ci_high = get_thr_ci(df[df.manager == manager], 'nWifiReal')

        if manager == 'Oracle':
            ax.plot(mean.index, mean, linestyle='--', c='gray', label=manager_name)
        else:
            ax.plot(mean.index, mean, marker='o', markersize=2, label=manager_name, c=colors[i])
            ax.fill_between(mean.index, ci_low, ci_high, alpha=0.3, color=colors[i], linewidth=0.0)

    ax.set_xlim((0, MAX_N_WIFI))
    ax.set_ylim(bottom=0)

    ax.set_ylabel('Aggregate throughput [Mb/s]')
    ax.set_title(fr'$\rho$ = {distance} m')

    ax.grid()


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    fig, axes = plt.subplots(2, 1, sharex='col')

    for distance, ax in zip([0, 20], axes):
        plot_results(ax, distance)

    axes[0].legend()
    axes[0].tick_params('x', labelbottom=False, bottom=False)
    axes[1].set_xlabel('Number of stations')

    plt.savefig(f'static-thr.pdf', bbox_inches='tight')
    plt.clf()
