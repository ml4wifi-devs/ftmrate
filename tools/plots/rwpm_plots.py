import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tools.plots.common import DATA_FILE, ALL_MANAGERS


def plot_results() -> None:
    df = pd.read_csv(DATA_FILE)
    df = df[df.mobility == 'RWPM']

    df = df[df.manager.isin(ALL_MANAGERS)].sort_values('manager')
    sns.boxplot(df, x='manager', y='throughput')
        
    plt.ylim(bottom=0)
    plt.xlabel('')
    plt.ylabel('Network throughput [Mb/s]')
    plt.tight_layout()

    plt.savefig('rwpm thr.svg', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    plot_results()
