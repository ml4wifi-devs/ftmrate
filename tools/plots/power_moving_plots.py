import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import pandas as pd

from tools.plots.common import *


MAX_DISTANCE = 55

POWER_CHANGE = {
    4: [1.41183, 2.8495, 4.75461, 5.19437, 7.88647, 15.4031, 23.3933,
        29.6835, 37.3989, 45.5318, 50.1398, 52.251, 52.8932],
    8: [2.82367, 5.699, 9.50922, 10.3887, 15.7729, 30.8061, 46.7867]
}


def plot_results(ax: plt.Axes, ax_id: int, delta: float, interval: float, velocity: float) -> None:
    colors = pl.cm.viridis(np.linspace(0., 1., len(ALL_MANAGERS) - 1))

    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobility == 'Moving') & (df.delta == delta) & (df.interval == interval) & (df.velocity == velocity)]

    for i, (manager, manager_name) in enumerate(ALL_MANAGERS.items()):
        mean, ci_low, ci_high = get_thr_ci(df[df.manager == manager], 'time')

        if manager == 'Oracle':
            ax.plot(mean.index, mean, linestyle='--', c='gray', label=manager_name)
        else:
            ax.plot(mean.index, mean, marker='o', markersize=1, label=manager_name, c=colors[i])
            ax.fill_between(mean.index, ci_low, ci_high, alpha=0.3, color=colors[i], linewidth=0.0)

    for i, x in enumerate(POWER_CHANGE[interval]):
        ax.axvline(x, linestyle='--', c='r', alpha=0.4, label='Power change' if i == 0 else None)

    ax.set_xlim((0, MAX_DISTANCE) if ax_id != 2 else (14, 25))
    ax.set_ylim((0, 125))

    ax.set_ylabel('Station throughput [Mb/s]')
    ax.set_title(fr'$\Delta$ = {delta} dB, 1/$\lambda$ = {interval} s, $\nu$ = {velocity} m/s')

    ax.grid()


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    plt.rcParams['figure.figsize'] = (2 * COLUMN_WIDTH + 2, COLUMN_HIGHT + 2)

    fig, axes = plt.subplots(2, 2)

    for i, (delta, interval, v, ax) in enumerate(zip([5, 5, 15, 15], [4, 8, 4, 8], [1, 0, 1, 0], axes.flatten())):
        plot_results(ax, i, delta, interval, v)

    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].legend()

    plt.savefig(f'power-moving-thr.pdf', bbox_inches='tight')
    plt.clf()
