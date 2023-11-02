import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import pandas as pd

from tools.plots.common import *


ALL_MANAGERS = {
    'ts_false': 'without RTS/CTS',
    'ts_true': 'with RTS/CTS',
    'kf_false': 'without RTS/CTS',
    'kf_true': 'with RTS/CTS'
}


def plot_results(ax: plt.Axes, distance: int) -> None:
    colors = pl.cm.viridis(np.linspace(0., 1., 5))
    colors_map = {
        'minstrel_false': colors[0],
        'kf_false': colors[3],
        'ts_false': colors[1],
        'minstrel_true': colors[0],
        'kf_true': colors[3],
        'ts_true': colors[1],
    }

    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobility == 'Hidden') & (df.distance == distance) & (df.nWifi == df.nWifiReal) & (df.nWifi <= 16)]

    for i, (manager, manager_name) in enumerate(ALL_MANAGERS.items()):
        mean, ci_low, ci_high = get_thr_ci(df[df.manager.str.lower() == manager], 'nWifi')

        if manager.startswith('oracle'):
            linestyle = '-.' if 'true' in manager else '--'
            ax.plot(mean.index, mean, linestyle=linestyle, c='gray', label=manager_name, linewidth=2)
        else:
            marker = 'd' if 'true' in manager else 'o'
            ax.plot(mean.index, mean, marker=marker, markersize=2, label=manager_name, c=colors_map[manager])
            ax.fill_between(mean.index, ci_low, ci_high, alpha=0.3, color=colors_map[manager], linewidth=0.0)

    ax.set_xlim((0, 16))
    ax.set_ylim((0, 25))
    ax.set_ylabel('Aggregate throughput [Mb/s]')
    ax.grid()


if __name__ == '__main__':

    PLOT_PARAMS["figure.figsize"] = (COLUMN_WIDTH, COLUMN_HIGHT / 2)
    plt.rcParams.update(PLOT_PARAMS)
    fig, ax = plt.subplots()

    plot_results(ax, distance=40)

    ax.set_xlabel(fr'Total number of stations')
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))

    lines = plt.gca().get_lines()
    legend1 = plt.legend([lines[i] for i in [1, 0]], [lines[i].get_label() for i in [1, 0]], loc="lower left", fontsize=6, title='Thompson sampling')
    legend2 = plt.legend([lines[i] for i in [3, 2]], [lines[i].get_label() for i in [3, 2]], loc="upper right", fontsize=6, title='FTMRate w/ KF')
    plt.gca().add_artist(legend1)

    plt.savefig(f'hn-bar_thr_nwifi.pdf', bbox_inches='tight')
    plt.clf()
