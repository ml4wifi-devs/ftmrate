import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit


def channel_model(x, bias, exponent):
    return bias - 10 * exponent * np.log10(x)


if __name__ == '__main__':
    rssi_df = pd.read_csv('rssi.csv')

    (bias, exponent), rssi_cov = curve_fit(
        f=channel_model,
        xdata=np.array(rssi_df.distance),
        ydata=np.array(rssi_df.rssi),
        bounds=([-np.inf, -1.0], [np.inf, 7.0])
    )
    bias_std, exponent_std = np.sqrt(np.diag(rssi_cov))

    print(f"bias = {bias}+-{bias_std}")
    print(f"exponent = {exponent}+-{exponent_std}")

    xs = np.linspace(0, 50, 501)

    sns.boxplot(data=rssi_df, x="distance", y="rssi", order=np.arange(51))
    plt.plot(xs, channel_model(xs, bias, exponent), linestyle="--", color="red")
    plt.ylabel("RSSI [dBm]")
    plt.xlabel("Distance [m]")
    plt.title(f"Exponent = {round(exponent, 4)} +/- {round(exponent_std, 4)}   \n    Bias = {round(bias, 4)} +/- {round(bias_std, 4)} [dB]")
    plt.savefig("rssi.pdf", bbox_inches="tight")
    plt.show()
