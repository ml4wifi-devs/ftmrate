import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tools.plots.common import *


def plot_results(velocity: float, distance: float) -> None:
    df = pd.read_csv(DATA_FILE)
    df = df.loc[(df.mobility == 'Moving') & (df.velocity == velocity) & (df.distance == distance)]

    results = get_thr_ttest(df)
    mask = np.tril(np.ones_like(results))

    ax = sns.heatmap(results, xticklabels=ALL_MANAGERS, yticklabels=ALL_MANAGERS, annot=True, mask=mask, cmap='flare')

    ax.figure.subplots_adjust(left=0.3)
    ax.figure.subplots_adjust(bottom=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
    plt.tight_layout()

    plt.savefig(f'moving v{velocity} d{distance} t-test.svg', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    distance_to_compare = 10

    for velocity in [1, 2]:
        plot_results(velocity, distance_to_compare)
