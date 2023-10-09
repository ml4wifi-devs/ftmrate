from argparse import ArgumentParser
from typing import Callable, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


PLOT_PARAMS = {
    'figure.figsize': (3.5, 2.45),
    'figure.dpi': 72,
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': 'cm',
    'axes.titlesize': 9,
    'axes.linewidth': 0.5,
    'grid.alpha': 0.42,
    'grid.linewidth': 0.5,
    'legend.title_fontsize': 7,
    'legend.fontsize': 7,
    'lines.linewidth': 0.5,
    'text.usetex': True,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
}

CONFIDENCE_INTERVAL = 0.99


def cdf(xs: np.ndarray, data: pd.DataFrame) -> np.ndarray:
    x = data['time'].values
    y = np.cumsum(data['rate'].values)
    y = y / y[-1]

    return np.interp(xs, x, y)


def get_ci(data: List, ci_interval: float = CONFIDENCE_INTERVAL) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    measurements = len(data)
    mean, std = np.mean(data, axis=0), np.std(data, axis=0)

    alpha = 1 - ci_interval
    z = t.ppf(1 - alpha / 2, measurements - 1)

    ci_low = mean - z * std / np.sqrt(measurements)
    ci_high = mean + z * std / np.sqrt(measurements)

    return mean, ci_low, ci_high


def get_cdf(xs: np.ndarray, df: pd.DataFrame) -> Tuple:
    runs = [df[(df['run'] == run) & (df['device'] == 'ap')] for run in df['run'].unique()]
    cdfs = [cdf(xs, run) for run in runs]
    return get_ci(cdfs)


def get_throughput(xs: np.ndarray, df: pd.DataFrame) -> Tuple:
    runs = [df[(df['run'] == run) & (df['device'] == 'ap')] for run in df['run'].unique()]
    cdfs = [cdf(xs, run) for run in runs]
    grads = [np.gradient(cdf, xs) for cdf in cdfs]
    grads = [grad / grad.mean() * run['rate'].mean() for grad, run in zip(grads, runs)]
    return get_ci(grads)


def get_data_rate(xs: np.ndarray, df: pd.DataFrame) -> Tuple:
    runs = [df[(df['run'] == run) & (df['device'] == 'sta')] for run in df['run'].unique()]
    rates = [np.interp(xs, run['time'].values, run['rate'].values) for run in runs]
    return get_ci(rates)


def single_plot(
        ftmrate_df: pd.DataFrame,
        iwlwifi_df: pd.DataFrame,
        data_fn: Callable,
        label: str,
        ylim: Tuple,
        n_points: int,
        experiment_time: int,
        filename: str
) -> None:
    xs = np.linspace(0, experiment_time, n_points)
    ftmrate_mean, ftmrate_low, ftmrate_high = data_fn(xs, ftmrate_df)
    iwlwifi_mean, iwlwifi_low, iwlwifi_high = data_fn(xs, iwlwifi_df)

    colors = plt.cm.viridis(np.linspace(0., 1., 5))

    plt.plot(xs, ftmrate_mean, label='FTMRate w/ KF', color=colors[3])
    plt.fill_between(xs, ftmrate_low, ftmrate_high, alpha=0.3, color=colors[3], linewidth=0)

    plt.plot(xs, iwlwifi_mean, label='iwlwifi', color=colors[0])
    plt.fill_between(xs, iwlwifi_low, iwlwifi_high, alpha=0.3, color=colors[0], linewidth=0)

    plt.xlim(0, experiment_time)
    plt.ylim(ylim)
    plt.xlabel('Time [s]')
    plt.ylabel(label)

    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--filename', type=str, required=True)
    args.add_argument('--n_points', type=int, default=100)
    args.add_argument('--experiment_time', type=int, required=True)
    args = args.parse_args()

    plt.rcParams.update(PLOT_PARAMS)

    df = pd.read_csv(args.filename)
    df = df[df['time'] < args.experiment_time]

    ftmrate_df = df[df['manager'] == 'ftmrate']
    iwlwifi_df = df[df['manager'] == 'iwlwifi']

    single_plot(ftmrate_df, iwlwifi_df, get_cdf, 'CDF', (0, 1), args.n_points, args.experiment_time, 'cdf.pdf')
    single_plot(ftmrate_df, iwlwifi_df, get_throughput, 'Approximated throughput [Mb/s]', (0, 80), args.n_points, args.experiment_time, 'throughput.pdf')
    single_plot(ftmrate_df, iwlwifi_df, get_data_rate, 'Per-frame data rate [Mb/s]', (0, 80), args.n_points, args.experiment_time, 'data_rate.pdf')
