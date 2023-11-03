import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import pandas as pd
import seaborn as sns

from tools.plots.common import *


N_WIFI = 2
DISTANCE = 40


def plot_results(ax: plt.Axes) -> None:
    colors = pl.cm.viridis(np.linspace(0., 1., len(ALL_MANAGERS) - 1)).tolist() + ['gray']

    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobility == 'Hidden') & (df.nWifiReal == N_WIFI) & (df.distance == DISTANCE)]

    sns.violinplot(ax=ax, data=df, x='manager', y='throughput', order=ALL_MANAGERS.keys(), palette=colors)
    oracle = df[df.manager == 'Oracle']['throughput'].mean()
    ax.axhline(oracle, linestyle='--', c='gray', label=f'{ALL_MANAGERS["Oracle"]} mean')

    ax.set_ylim((0, 30))
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30])

    ax.set_ylabel('Aggregate throughput [Mb/s]')
    ax.set_xlabel('')

    ax.set_axisbelow(True)
    ax.grid(axis='y')


if __name__ == '__main__':

    PLOT_PARAMS["figure.figsize"] = (COLUMN_WIDTH, COLUMN_HIGHT / 2)
    plt.rcParams.update(PLOT_PARAMS)
    fig, ax = plt.subplots()

    plot_results(ax)

    ax.set_xticklabels(ALL_MANAGERS.values())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.legend()

    plt.savefig(f'hn-bar_thr_violin.pdf', bbox_inches='tight')
    plt.clf()
