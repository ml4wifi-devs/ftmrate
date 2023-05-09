import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tools.plots.common import *


if __name__ == '__main__':
    df = pd.read_csv('tools/outputs/low-rate_results.csv')
    df = df[df.manager == 'Minstrel']
    df = df[df.mobility == 'RWPM']

    plt.rcParams.update(PLOT_PARAMS)
    fig, axs = plt.subplots(2, 1)

    for v, ax in zip([0., 1.4], axs):
        data = df[df.velocity == v]
        ax.hist(data.nWifiReal, label='Minstrel', bins=np.arange(0, 11) + 0.5, rwidth=0.5)
        ax.set_title(f'RWPM [v = {v} m/s]')
        ax.set_xlim(-0.5, 10.5)
        ax.set_xticks(range(0, 11))
        ax.set_yticks(range(0, 21, 4))
        ax.set_xlabel('Number of stations')
        ax.set_ylabel('Count')
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('wifi-ftm-ns3.35-low-rate-warmup.pdf', bbox_inches='tight')
    plt.show()
