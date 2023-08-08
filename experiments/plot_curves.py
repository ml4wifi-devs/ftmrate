from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from ftmrate import expected_rates


def plot_curves(distance_range: tuple, points: int = 200, path: str = None) -> None:
    """
    Plots the expected rates for each MCS given a distance range.

    Parameters
    ----------
    distance_range : tuple
        The range of distances to plot.
    points : int, optional
        The number of points to plot for each distance.
    path : str, optional
        The path to save the plot to.
    """

    distances = np.linspace(*distance_range, points)
    rates = np.empty((points, 16))

    for i, distance in enumerate(distances):
        rates[i] = expected_rates()(distance)

    for mcs in range(16):
        plt.plot(distances, rates[:, mcs], label=f'MCS {mcs}')

    plt.xlabel('Distance [m]')
    plt.ylabel('Expected rate [Mb/s]')
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
    args.add_argument('--points', type=int, default=200)
    args.add_argument('--path', type=str, default=None)
    args = args.parse_args()

    plt.rcParams['figure.figsize'] = (6, 4)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['legend.fontsize'] = 9

    plot_curves((args.distance_min, args.distance_max), args.points, args.path)
