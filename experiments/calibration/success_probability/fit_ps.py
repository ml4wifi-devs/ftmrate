import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tensorflow_probability.substrates import numpy as tfp

tfd = tfp.distributions


def plot_estimated(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray, title: str, legend_title: str = None) -> None:
    plt.scatter(x, y, label='Measurements', marker='D', s=15)
    plt.plot(x, y_hat, label='Estimated')

    plt.xlim((-80, -50))
    plt.ylim((0, 1))
    plt.ylabel('Success probability')
    plt.xlabel('RSSI [dBm]')
    plt.title(title)

    plt.legend(title=legend_title)
    plt.grid()
    plt.show()


def cdf_fn(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    return tfd.Normal(loc, scale).cdf(x)


if __name__ == '__main__':
    data = pd.read_csv('data.csv')

    for mode in range(16):
        df = data[(data['mode'] == mode) & (data['n'] > 0)].sort_values('rssi')
        rssi = df['rssi'].to_numpy(dtype=np.float64)
        p_s = (df['k'] / df['n']).to_numpy(dtype=np.float64)

        (loc, scale), cov = curve_fit(f=cdf_fn, xdata=rssi, ydata=p_s)
        loc_std, scale_std = np.sqrt(np.diag(cov))

        params_str = f'loc={loc:.3f} +/- {loc_std:.3f}\nscale={scale:.3f} +/- {scale_std:.3f}\n'
        print(f'[{loc}, {scale}],')

        plot_estimated(rssi, p_s, cdf_fn(rssi, loc, scale), params_str, f'MCS {mode}')
