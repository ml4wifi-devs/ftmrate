import matplotlib.pyplot as plt
import pandas as pd

from tools.plots.common import *


MAX_N_WIFI = 30


def plot_results(distance: float) -> None:
    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobility == 'Static') & (df.distance == distance) & (df.nWifiReal <= MAX_N_WIFI)]

    for manager in ALL_MANAGERS:
        mean, ci_low, ci_high = get_thr_ci(df[df.manager == manager], 'nWifiReal')
        plt.plot(mean.index, mean, marker='o', markersize=3, label=manager)
        plt.fill_between(mean.index, ci_low, ci_high, alpha=0.2)

    plt.ylim(bottom=0)
    plt.xlabel('Number of stations')
    plt.ylabel('Network throughput [Mb/s]')
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'static d{distance} thr.svg', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    for distance in [0, 20]:
        plot_results(distance)
