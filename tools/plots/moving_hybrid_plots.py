import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import pandas as pd

from tools.plots.common import *


MAX_DISTANCE = 25
MANAGERS = {
    'Oracle': 'Oracle',
    'TS': 'Thompson sampling',
    'MAB_KF': 'MAB w/ KF',
    'KF': 'FTMRate w/ KF',
}
WALLS = [5, 10, 15, 20]


def plot_results() -> None:
    colors = pl.cm.viridis(np.linspace(0., 1., 5))
    colors_map = {
        'MAB_KF': "tab:red",
        'TS': colors[1],
        'KF': colors[3],
    }

    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobility == 'Moving') & (df.velocity == 0.5)]

    for i, (manager, manager_name) in enumerate(MANAGERS.items()):
        mean, ci_low, ci_high = get_thr_ci(df[df.manager == manager], 'distance')

        if manager == 'Oracle':
            plt.plot(mean.index, mean, linestyle='--', c='gray', label=manager_name)
        else:
            plt.plot(mean.index, mean, marker=MARKERS[manager], markersize=0.5, label=manager_name, c=colors_map[manager])
            plt.fill_between(mean.index, ci_low, ci_high, alpha=0.3, color=colors_map[manager], linewidth=0.0)

    for i, x in enumerate(WALLS):
        plt.axvline(x, linestyle='--', c='r', alpha=0.7, label='Wall' if i == 0 else None)

    plt.xlim((0, MAX_DISTANCE))
    plt.ylim((0, 125))

    plt.ylabel('Station throughput [Mb/s]')
    plt.xlabel('Distance from AP [m]')

    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    plt.rcParams['figure.figsize'] = (COLUMN_WIDTH, 2 * COLUMN_WIDTH / (1 + np.sqrt(5)))

    plot_results()

    plt.savefig(f'moving-hybrid-thr.pdf', bbox_inches='tight')
    plt.clf()
