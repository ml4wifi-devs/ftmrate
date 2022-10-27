import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tools.plots.common import all_managers


TOOLS_DIR = os.getenv('TOOLS_DIR', os.path.join(os.path.expanduser("~"), 'ftmrate/tools'))
DATA_FILE = os.path.join(TOOLS_DIR, 'outputs', 'all_results.csv')


def plot_results():
    df = pd.read_csv(DATA_FILE)
    df = df[df.mobility == 'RWPM']

    if len(df) == 0:
        return

    df = df[df.manager.isin(all_managers)].sort_values('manager')
    sns.boxplot(df, x='manager', y='throughput')
        
    plt.ylim(bottom=0)
    plt.xlabel('')
    plt.ylabel('Network throughput [Mb/s]')
    plt.tight_layout()

    plt.savefig('rwpm thr.svg', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    plot_results()
