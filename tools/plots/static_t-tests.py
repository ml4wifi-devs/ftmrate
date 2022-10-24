import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from scipy.stats import ttest_ind, f_oneway

from plots.common import all_managers


TOOLS_DIR = os.path.join(os.path.expanduser("~"), 'tools')
DATA_FILE = os.path.join(TOOLS_DIR, 'outputs', 'all_results.csv')


def plot_results(distance, n_wifi, output_file):
    df = pd.read_csv(DATA_FILE)
    df = df.loc[(df.mobility == 'Static') & (df.nWifi == n_wifi) & (df.distance == distance)]

    throughputs = []
    managers = []

    for manager in all_managers:
        if len(data := df[df.manager == manager].throughput) == 0:
            continue

        throughputs.append(data)
        managers.append(manager.replace('_', ' '))

    results = np.zeros((len(managers), len(managers)))

    for i in range(len(managers)):
        for j in range(i, len(managers)):
            stats, pval = ttest_ind(throughputs[i], throughputs[j], equal_var=False)
            results[i, j] = pval

    stats, pval = f_oneway(*throughputs)

    mask = np.tril(np.ones_like(results))
    ax = sn.heatmap(results, xticklabels=managers, yticklabels=managers, annot=True, mask=mask, cmap='flare')

    ax.figure.subplots_adjust(left=0.3)
    ax.figure.subplots_adjust(bottom=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
    plt.title(f'ANOVA p-value: {pval:.3f}')
    plt.tight_layout()

    plt.savefig(f'{output_file}.svg', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    n_wifi_to_compare = 30

    for distance in [0, 20, 40]:
        plot_results(distance, n_wifi_to_compare, f'static d{distance} n{n_wifi_to_compare} t-test')
