import jax
import jax.numpy as jnp
import matplotlib.pylab as pl
import matplotlib.pyplot as plt

from common import PLOT_PARAMS
from ml4wifi.utils.wifi_specs import distance_to_snr, expected_rates, wifi_modes_rates


def plot_data_rates() -> None:
    plt.rcParams.update(PLOT_PARAMS)

    colors = pl.cm.jet(jnp.linspace(0., 1., 12))
    n_points = 200

    snr = jnp.linspace(5., 50., n_points)
    distance = distance_to_snr.inverse(snr)
    exp_rates = jax.vmap(expected_rates)(distance)

    for mode, (exp_rate, data_rate, c) in enumerate(zip(exp_rates.T, wifi_modes_rates, colors)):
        plt.plot(snr, exp_rate, c=c, label=mode)
        plt.axhline(data_rate, alpha=0.3, c=c, linestyle='--')

    plt.ylabel(r'$\lambda$ [Mb/s]')
    plt.xlabel(r'$\gamma$ [dBm]')
    plt.xlim((5., 50.))
    plt.ylim((0., 130.))
    plt.legend(title='MCS')

    plt.savefig(f'data_rates.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot_data_rates()
