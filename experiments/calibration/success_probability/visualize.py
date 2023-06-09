import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv('data_raw.csv')

    for mode in range(16):
        df = data[(data['mode'] == mode) & (data['n'] > 0)].sort_values('distance')

        d = df['distance'].to_numpy(dtype=np.float64)
        p_s = (df['k'] / df['n']).to_numpy(dtype=np.float64)
        plt.scatter(d, p_s, marker='D', s=15)

        plt.ylim((0, 1))
        plt.xlim((0, 75))

        plt.title(f'MCS {mode}')
        plt.ylabel('Success probability')
        plt.xlabel('Distance [m]')

        plt.grid()
        plt.show()
