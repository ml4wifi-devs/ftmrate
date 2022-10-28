import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tools.plots.common import *


def plot_results(distance: float, n_wifi: int, output_file: str) -> None:
    df = pd.read_csv(DATA_FILE)
    df = df.loc[(df.mobility == 'Static') & (df.nWifiReal == n_wifi) & (df.distance == distance)]

    results = get_thr_ttest(df)
    mask = np.tril(np.ones_like(results))

    ax = sns.heatmap(results, xticklabels=ALL_MANAGERS, yticklabels=ALL_MANAGERS, annot=True, mask=mask, cmap='flare')

    ax.figure.subplots_adjust(left=0.3)
    ax.figure.subplots_adjust(bottom=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
    plt.tight_layout()

    plt.savefig(f'{output_file}.svg', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    n_wifi_to_compare = 16

    for distance in [0, 20, 40]:
        plot_results(distance, n_wifi_to_compare, f'static d{distance} n{n_wifi_to_compare} t-test')
