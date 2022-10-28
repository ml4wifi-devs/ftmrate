import matplotlib.pyplot as plt
import pandas as pd

from tools.plots.common import *


def plot_results(velocity: float) -> None:
    df = pd.read_csv(DATA_FILE)
    df = df[(df.mobility == 'Moving') & (df.velocity == velocity)]

    for manager in ALL_MANAGERS:
        mean, ci_low, ci_high = get_thr_ci(df[df.manager == manager], 'distance')
        plt.plot(mean.index, mean, marker='o', markersize=3, label=manager)
        plt.fill_between(mean.index, ci_low, ci_high, alpha=0.2)

    plt.ylim(bottom=0)
    plt.xlabel('Distance from AP [m]')
    plt.ylabel('Station throughput [Mb/s]')
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'moving v{velocity} thr.svg', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    for velocity in [1, 2]:
        plot_results(velocity)
