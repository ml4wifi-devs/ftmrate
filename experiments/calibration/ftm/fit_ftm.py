import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit


def linear_model(x, a_val, b_val):
    return a_val * x + b_val


if __name__ == '__main__':
    ftm_df = pd.read_csv('ftm.csv')

    (a_val, b_val), ftm_cov = curve_fit(
        f=linear_model,
        xdata=np.array(ftm_df.distance),
        ydata=np.array(ftm_df.ftmdist)
    )
    a_val_std, b_val_std = np.sqrt(np.diag(ftm_cov))

    print(f"a = {a_val}+-{a_val_std}")
    print(f"b = {b_val}+-{b_val_std}")

    ftm_df_corrected = ftm_df.copy()
    ftm_df_corrected.loc[:, 'ftmdist'] = (ftm_df_corrected.loc[:, 'ftmdist'] - b_val) / (a_val)

    xs = np.linspace(0, 50, 501)
    plt.plot(xs, xs, linestyle="--", color="#000000", label="True Value")

    sns.boxplot(data=ftm_df_corrected, x="distance", y="ftmdist", order=np.arange(51))
    plt.ylabel("FTM [m]")
    plt.xlabel("Distance [m]")
    plt.title(f"a = {round(a_val, 4)} +/- {round(a_val_std, 4)}   \n    b = {round(b_val, 4)} +/- {round(b_val_std, 4)} [m]")
    plt.legend()
    plt.savefig("ftm.pdf", bbox_inches="tight")
    plt.show()
