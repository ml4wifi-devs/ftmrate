import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tensorflow_probability.substrates import numpy as tfp

tfd = tfp.distributions


def plot_estimated(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray, title: str, legend_title: str = None) -> None:
    plt.scatter(x, y, label='Measurements', marker='o', s=40, alpha=0.3)
    plt.plot(x, y_hat, label='Estimated')

    plt.ylim((-0.015, 1.015))
    plt.ylabel('Success probability')
    plt.xlabel('RSSI [dBm]')
    plt.title(title)

    plt.legend(title=legend_title)
    plt.grid()
    plt.savefig(f"ps_mcs-{legend_title[4:]}.png", bbox_inches="tight")
    plt.show()


def cdf_fn(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    return tfd.Normal(loc, scale).cdf(x)


if __name__ == '__main__':
    data = pd.read_csv('data.csv')

    for mode in range(16):
        df = data[(data['mode'] == mode) & (data['n'] > 0)].sort_values('rssi')
        rssi = df['rssi'].to_numpy(dtype=np.float64)
        p_s = (df['k'] / df['n']).to_numpy(dtype=np.float64)
        p_s = np.clip(p_s, 0, 1)

        (loc, scale), cov = curve_fit(f=cdf_fn, xdata=rssi, ydata=p_s, bounds=[
            [-150, 0], [-50, 10]
        ])
        loc_std, scale_std = np.sqrt(np.diag(cov))

        params_str = f'loc={loc:.3f} +/- {loc_std:.3f}\nscale={scale:.3f} +/- {scale_std:.3f}\n'
        print(f'[{loc}, {scale}],')

        plot_estimated(rssi, p_s, cdf_fn(rssi, loc, scale), params_str, f'MCS {mode}')
