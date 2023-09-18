from argparse import ArgumentParser
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLUMN_WIDTH = 3.5
COLUMN_HIGHT = 1.4 * COLUMN_WIDTH
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
    'legend.fontsize': 7,
    'lines.linewidth': 0.5,
    'text.usetex': True,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
}

MARKERS = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H']
LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

EXPERIMENT_TIME = 25
CDF_POINTS = 100


def plot_data_rate(df: pd.DataFrame, axes: plt.Axes) -> None:
    runs = df['run'].unique()

    for i, run in enumerate(runs):
        run_df = df[(df['run'] == run) & (df['device'] == 'ap')]
        axes.plot(run_df['time'], run_df['rate'], linestyle=LINESTYLES[i % len(LINESTYLES)], marker=MARKERS[i % len(MARKERS)], label=f'Run {run}', markersize=1.5)

    axes.set_xlim(0, EXPERIMENT_TIME)
    axes.set_ylim(0, 80)
    axes.set_ylabel('Data rate [Mb/s]')

    axes.grid()


def twin_plot(ftmrate_df: pd.DataFrame, iwlwifi_df: pd.DataFrame, plot_fn: Callable, filename: str) -> None:
    _, axes = plt.subplots(2, 1, sharex=True)

    plot_fn(ftmrate_df, axes[0])
    plot_fn(iwlwifi_df, axes[1])

    axes[0].set_title('FTMRate w/ KF')
    axes[1].set_title('iwlwifi')

    axes[0].tick_params('x', labelbottom=False, bottom=False)
    axes[1].set_xlabel('Time [s]')
    axes[0].legend(ncol=3)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')


def cdf(xs: np.ndarray, data: pd.DataFrame) -> np.ndarray:
    x = data['time'].values
    y = np.cumsum(data['rate'].values)
    y = y / y[-1]

    return np.interp(xs, x, y)


def get_cdf(ftmrate_df: pd.DataFrame, iwlwifi_df: pd.DataFrame) -> Tuple:

    def calculate(xs: np.ndarray, data: pd.DataFrame) -> np.ndarray:
        runs = [data[(data['run'] == run) & (data['device'] == 'ap')] for run in data['run'].unique()]
        cdfs = [cdf(xs, run) for run in runs]
        mean = np.mean(cdfs, axis=0)
        std = np.std(cdfs, axis=0)

        return mean, std

    xs = np.linspace(0., EXPERIMENT_TIME, CDF_POINTS)
    return xs, calculate(xs, ftmrate_df), calculate(xs, iwlwifi_df)


def get_throughput(ftmrate_df: pd.DataFrame, iwlwifi_df: pd.DataFrame) -> Tuple:

    def approx_throughput(xs: np.ndarray, data: pd.DataFrame) -> np.ndarray:
        runs = [data[(data['run'] == run) & (data['device'] == 'ap')] for run in data['run'].unique()]
        cdfs = [cdf(xs, run) for run in runs]
        grads = [np.gradient(cdf, xs) for cdf in cdfs]
        grads = [grad / grad.mean() * run['rate'].mean() for grad, run in zip(grads, runs)]
        mean = np.mean(grads, axis=0)
        std = np.std(grads, axis=0)

        return mean, std

    xs = np.linspace(0., EXPERIMENT_TIME, CDF_POINTS)
    return xs, approx_throughput(xs, ftmrate_df), approx_throughput(xs, iwlwifi_df)


def single_plot(ftmrate_df: pd.DataFrame, iwlwifi_df: pd.DataFrame, data_fn: Callable, label: str, ylim: Tuple, filename: str) -> None:
    plt.figure(figsize=(COLUMN_WIDTH, COLUMN_HIGHT / 2))
    
    xs, (ftmrate_mean, ftmrate_std), (iwlwifi_mean, iwlwifi_std) = data_fn(ftmrate_df, iwlwifi_df)
    colors = plt.cm.viridis(np.linspace(0., 1., 5))

    plt.plot(xs, ftmrate_mean, label='FTMRate w/ KF', color=colors[3])
    plt.fill_between(xs, ftmrate_mean - ftmrate_std, ftmrate_mean + ftmrate_std, alpha=0.3, color=colors[3], linewidth=0)

    plt.plot(xs, iwlwifi_mean, label='iwlwifi', color=colors[0])
    plt.fill_between(xs, iwlwifi_mean - iwlwifi_std, iwlwifi_mean + iwlwifi_std, alpha=0.3, color=colors[0], linewidth=0)

    plt.xlim(0, EXPERIMENT_TIME)
    plt.ylim(ylim)
    plt.xlabel('Time [s]')
    plt.ylabel(label)

    plt.grid()
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--filename', type=str, required=True)
    args = args.parse_args()

    plt.rcParams.update(PLOT_PARAMS)

    df = pd.read_csv(args.filename)
    df = df[df['time'] < EXPERIMENT_TIME]

    ftmrate_df = df[df['manager'] == 'ftmrate']
    iwlwifi_df = df[df['manager'] == 'iwlwifi']

    twin_plot(ftmrate_df, iwlwifi_df, plot_data_rate, f'data_rate.pdf')
    single_plot(ftmrate_df, iwlwifi_df, get_cdf, 'CDF', (0, 1), f'cdf.pdf')
    single_plot(ftmrate_df, iwlwifi_df, get_throughput, 'Approximated throughput [Mb/s]', (0, 80), f'throughput.pdf')
