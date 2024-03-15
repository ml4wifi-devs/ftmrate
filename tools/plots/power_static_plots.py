import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import pandas as pd

from tools.plots.common import *


def plot_results(ax: plt.Axes, delta: float, n_wifi: float, distance: float) -> None:
    colors = pl.cm.viridis(np.linspace(0., 1., len(ALL_MANAGERS) - 1))

    df = pd.read_csv(DATA_FILE)
    df = df[df.nWifi == n_wifi]

    for i, (manager, manager_name) in enumerate(ALL_MANAGERS.items()):
        mean, ci_low, ci_high = get_thr_ci(df[df.manager == manager], 'interval')

        if manager == 'Oracle':
            ax.plot(mean.index, mean, linestyle='--', c='gray', label=manager_name)
        else:
            ax.plot(mean.index, mean, marker=MARKERS[manager], markersize=1, label=manager_name, c=colors[i])
            ax.fill_between(mean.index, ci_low, ci_high, alpha=0.3, color=colors[i], linewidth=0.0)

    ax.set_xscale('log')
    
    ax.set_ylim((0, 125))

    ax.set_ylabel('Throughput [Mb/s]')
    ax.set_title(fr'{n_wifi} STAs, $\Delta$ = {delta} dB, $\rho$ = {distance}')

    ax.grid()


if __name__ == "__main__":
    plt.rcParams.update(PLOT_PARAMS)
    plt.rcParams['figure.figsize'] = (COLUMN_WIDTH, COLUMN_HIGHT)

    fig, axes = plt.subplots(2, 1)

    for i, (delta, n_wifi, distance, ax) in enumerate(zip([15, 15], [1, 10], [7, 7], axes.flatten())):
        plot_results(ax, delta, n_wifi, distance)

    axes[0].legend(loc='lower right')
    axes[0].tick_params('x', labelbottom=False, bottom=False)
    axes[1].set_xlabel('Mean interval $\\frac{1}{\lambda}$ [s]')

    plt.savefig(f'power-static-thr.pdf', bbox_inches='tight')
    plt.clf()
