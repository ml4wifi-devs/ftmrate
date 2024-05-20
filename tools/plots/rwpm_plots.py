import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import pandas as pd
import seaborn as sns

from tools.plots.common import *


N_WIFI = 10


def plot_results(ax: plt.Axes, velocity: float) -> None:
    colors = pl.cm.viridis(np.linspace(0., 1., len(ALL_MANAGERS) - 1)).tolist() + ['gray']

    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobility == 'RWPM') & (df.nWifiReal == N_WIFI) & (df.velocity == velocity)]

    sns.violinplot(ax=ax, data=df, x='manager', y='throughput', order=ALL_MANAGERS.keys(), palette=colors)
    oracle = df[df.manager == 'Oracle']['throughput'].mean()
    ax.axhline(oracle, linestyle='--', c='gray', label=f'{ALL_MANAGERS["Oracle"]} mean')

    ax.set_ylim((0, 60))
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])

    ax.set_ylabel('Aggregate throughput [Mb/s]')
    ax.set_xlabel('')

    ax.set_axisbelow(True)
    ax.grid(axis='y')


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    plt.rcParams['figure.figsize'] = (COLUMN_WIDTH, COLUMN_HIGHT / 2)
    fig, axes = plt.subplots(1, 1)

    for velocity, ax in zip([1.4], [axes]):
        plot_results(ax, velocity)

    axes.set_xticklabels(ALL_MANAGERS.values())
    plt.setp(axes.xaxis.get_majorticklabels(), rotation=30, ha="right")
    axes.legend()

    plt.savefig(f'rwpm-thr.pdf', bbox_inches='tight')
    plt.clf()
