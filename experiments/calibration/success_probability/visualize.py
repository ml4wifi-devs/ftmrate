import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv('data_augmented.csv')

    for mode in range(16):
        df = data[(data['mode'] == mode) & (data['n'] > 0)].sort_values('distance')

        d = df['distance'].to_numpy(dtype=np.float64)
        p_s = (df['k'] / df['n']).to_numpy(dtype=np.float64)
        p_s = np.clip(p_s, 0, 1)

        plt.scatter(d, p_s, marker='o', s=40, alpha=0.3)

        plt.ylim((-0.015, np.max(p_s) + 0.015 if p_s.shape[0] else 1.015))
        plt.xlim((0, 20))
        plt.xticks(list(range(0, 22, 2)))

        plt.title(f'MCS {mode}')
        plt.ylabel('Success probability')
        plt.xlabel('Distance [m]')

        plt.grid()
        plt.show()
