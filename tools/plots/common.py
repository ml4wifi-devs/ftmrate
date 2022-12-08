import os
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_ind


TOOLS_DIR = os.getenv('TOOLS_DIR', os.path.join(os.path.expanduser("~"), 'ftmrate/tools'))
DATA_FILE = os.path.join(TOOLS_DIR, 'outputs', 'all_results.csv')

ALL_MANAGERS = {
    'Minstrel': 'Minstrel',
    'TS': 'Thompson Sampling',
    'ES': 'FTMRate w/ ES',
    'KF': 'FTMRate w/ KF',
    'PF': 'FTMRate w/ PF',
    'Oracle': 'Oracle'
}
MIN_REPS = 5
CONFIDENCE_INTERVAL = 0.99

COLUMN_WIDTH = 3.5
COLUMN_HIGHT = 2 * COLUMN_WIDTH / (1 + np.sqrt(5))
PLOT_PARAMS = {
    'figure.figsize': (COLUMN_WIDTH, COLUMN_HIGHT),
    'figure.dpi': 72,
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': 'cm',
    'axes.titlesize': 9,
    'text.usetex': True,
    'lines.linewidth': 0.5,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'legend.title_fontsize': 5,
    'legend.fontsize': 5,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.35,
}


def get_thr_ci(
        data: pd.DataFrame,
        column: str,
        ci_interval: float = CONFIDENCE_INTERVAL
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    count = data.groupby([column])['throughput'].count()
    mask = count[count >= MIN_REPS].index.tolist()

    data = data[data[column].isin(mask)]
    data = data.groupby([column])['throughput']

    measurements = data.count()
    mean = data.mean()
    std = data.std()

    alpha = 1 - ci_interval
    z = t.ppf(1 - alpha / 2, measurements - 1)

    ci_low = mean - z * std / np.sqrt(measurements)
    ci_high = mean + z * std / np.sqrt(measurements)

    return mean, ci_low, ci_high


def get_thr_ttest(data: pd.DataFrame) -> np.ndarray:
    throughputs = [data[data.manager == manager]['throughput'] for manager in ALL_MANAGERS]

    n = len(throughputs)
    results = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            stats, pval = ttest_ind(throughputs[i], throughputs[j], equal_var=False)
            results[i, j] = pval

    return results
