from argparse import ArgumentParser
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability.substrates.numpy.bijectors as tfb

from ftmrate import expected_rates, WIFI_MODES_RATES


COLUMN_WIDTH = 3.5
COLUMN_HIGHT = 2 * COLUMN_WIDTH / (1 + np.sqrt(5))
PLOT_PARAMS = {
    'figure.figsize': (COLUMN_WIDTH, COLUMN_HIGHT),
    'figure.dpi': 72,
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': 'cm',
    'axes.titlesize': 9,
    'axes.linewidth': 0.5,
    'grid.alpha': 0.42,
    'grid.linewidth': 0.5,
    'legend.title_fontsize': 7,
    'legend.fontsize': 6,
    'lines.linewidth': 0.5,
    'text.usetex': True,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5
}

N_POINTS = 300


def plot_curves(
        curve_fn: Callable,
        ylabel: str,
        ylim: tuple,
        distance_range: tuple,
        mcs_list: list,
        path: str = None
) -> None:
    """
    Plots the expected rates for given MCS values and a distance range.

    Parameters
    ----------
    curve_fn : Callable
        The function to plot.
    ylabel : str
        The label of the y-axis.
    ylim : tuple
        The range of the y-axis.
    distance_range : tuple
        The range of distances to plot.
    mcs_list : list
        The list of MCS values to plot.
    path : str, optional
        The path to save the plot to.
    """

    distances = np.linspace(*distance_range, N_POINTS)
    rates = np.empty((N_POINTS, 16))

    n_ss = sum(mcs <= 7 for mcs in mcs_list)
    colors = plt.cm.viridis(np.concatenate([np.linspace(0, 1, n_ss)] * 2))

    for i, distance in enumerate(distances):
        rates[i] = curve_fn(distance)

    for c, mcs in zip(colors, mcs_list):
        plt.plot(distances, rates[:, mcs], label=f'MCS {mcs}', c=c, linestyle='solid' if mcs <= 7 else 'dashdot')

    plt.xlabel('Distance [m]')
    plt.xlim(distance_range)
    plt.ylabel(ylabel)
    plt.ylim(ylim)

    plt.grid()
    plt.legend(ncol=2, loc='upper right')
    plt.tight_layout()

    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--distance_min', type=float, default=2.)
    args.add_argument('--distance_max', type=float, default=12.)
    args.add_argument('--mcs', type=int, action='append', required=True)
    args.add_argument('--path', type=str, default=None)
    args.add_argument('--type', default='rate', choices=['rate', 'ps'])
    args = args.parse_args()

    plt.rcParams.update(PLOT_PARAMS)
    plot_args = ((args.distance_min, args.distance_max), args.mcs, args.path)

    if args.type == 'rate':
        plot_curves(expected_rates(), 'Expected rate [Mb/s]', (0, 60), *plot_args)
    else:
        plot_curves(tfb.Scale(1 / WIFI_MODES_RATES)(expected_rates()), 'Success probability', (0, 1), *plot_args)
