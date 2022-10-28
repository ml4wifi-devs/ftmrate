import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tools.plots.common import *


def plot_results(output_file: str) -> None:
    df = pd.read_csv(DATA_FILE)
    df = df[df.mobility == 'RWPM']

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
    plot_results(f'rwpm t-test')
