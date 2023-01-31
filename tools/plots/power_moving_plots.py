import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import pandas as pd

from tools.plots.common import *


MAX_DISTANCE = 55

POWER_CHANGE = {
    4: [1.41183, 2.8495, 4.75461, 5.19437, 7.88647, 15.4031, 23.3933,
        29.6835, 37.3989, 45.5318, 50.1398, 52.251, 52.8932, 58.84],
    10: [3.52958, 7.12375, 11.8865, 12.9859, 19.7162, 38.5077, 58.4833]
}


def plot_results(ax: plt.Axes, delta: float, interval: float) -> None:
    colors = pl.cm.viridis(np.linspace(0., 1., len(ALL_MANAGERS) - 1))

    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobility == 'Moving') & (df.delta == delta) & (df.interval == interval)]

    for i, (manager, manager_name) in enumerate(ALL_MANAGERS.items()):
        mean, ci_low, ci_high = get_thr_ci(df[df.manager == manager], 'distance')

        if manager == 'Oracle':
            ax.plot(mean.index, mean, linestyle='--', c='gray', label=manager_name)
        else:
            ax.plot(mean.index, mean, marker='o', markersize=1, label=manager_name, c=colors[i])
            ax.fill_between(mean.index, ci_low, ci_high, alpha=0.3, color=colors[i], linewidth=0.0)

    for i, x in enumerate(POWER_CHANGE[interval]):
        ax.axvline(x, linestyle='--', c='r', alpha=0.4, label='Power change' if i == 0 else None)

    ax.set_xlim((0, MAX_DISTANCE))
    ax.set_ylim((0, 125))

    ax.set_ylabel('Station throughput [Mb/s]')
    ax.set_title(fr'$\Delta$ = {delta} dB, 1/$\lambda$ = {interval} s')

    ax.grid()


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    plt.rcParams['figure.figsize'] = (2 * COLUMN_WIDTH + 2, COLUMN_HIGHT)

    fig, axes = plt.subplots(2, 2, sharex='col')

    for delta, interval, ax in zip([5, 5, 15, 15], [4, 10, 4, 10], axes.flatten()):
        plot_results(ax, delta, interval)

    axes[0, 0].tick_params('x', labelbottom=False, bottom=False)
    axes[0, 1].tick_params('x', labelbottom=False, bottom=False)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].legend()

    plt.savefig(f'power-moving-thr.pdf', bbox_inches='tight')
    plt.clf()
