from argparse import ArgumentParser

import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tensorflow_probability.substrates import numpy as tfp

from ml4wifi.params_fit import load_parameters_file, CSV_FILES_DIR

tfd = tfp.distributions


def plot_estimated(x, y, y_hat, title):
    plt.plot(x, y, label='True')
    plt.plot(x, y_hat, label='Estimated')

    plt.xlim((6, 55))
    plt.ylim((0, 1))
    plt.ylabel('Success probability')
    plt.xlabel('SNR [dBm]')
    plt.title(title)

    plt.legend()
    plt.savefig(f'{title}.svg', bbox_inches='tight')
    plt.clf()


def cdf_fn(x, loc, scale, skewness, tailweight):
    return tfd.SinhArcsinh(loc, scale, skewness, tailweight).cdf(x)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--data_path', default='csv_files/success_probability.csv', type=str)
    args.add_argument('--output_file', default='parameters.csv', type=str)
    args.add_argument('--plot', action='store_false', default=True)
    args = args.parse_args()

    data = pd.read_csv(args.data_path)
    params_df = load_parameters_file(name=args.output_file)

    for mode in range(12):
        df = data[(data['mode'] == mode) & (data['n'] > 0)].sort_values('snr')
        snr = df['snr'].to_numpy(dtype=np.float64)
        p_s = (df['k'] / df['n']).to_numpy(dtype=np.float64)

        (loc, scale, skewness, tailweight), cov = curve_fit(
            f=cdf_fn,
            xdata=snr,
            ydata=p_s,
            bounds=([5.0, 0.0, 0.0, 0.0], [50.0, np.inf, np.inf, np.inf])
        )
        loc_std, scale_std, skewness_std, tailweight_std = np.sqrt(np.diag(cov))

        print(f'Best fitted SinhArcsinh distribution [MCS = {mode}]:')
        print(f'loc={loc} +/- {loc_std}')
        print(f'scale={scale} +/- {scale_std}')
        print(f'skewness={skewness} +/- {skewness_std}')
        print(f'tailweight={tailweight} +/- {tailweight_std}\n')

        if args.plot:
            plot_estimated(snr, p_s, cdf_fn(snr, loc, scale, skewness, tailweight), f'MCS {mode}')

        row = params_df.loc[mode]
        params_df.iloc[mode, :5] = mode, loc, scale, skewness, tailweight

    params_df.to_csv(os.path.join(CSV_FILES_DIR, args.output_file), index=False)
