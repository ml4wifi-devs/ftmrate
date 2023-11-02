import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from common import PLOT_PARAMS, COLUMN_WIDTH
from ml4wifi.utils.wifi_specs import expected_rates, wifi_modes_rates, DEFAULT_TX_POWER


def plot_data_rates() -> None:
    n_points = 200
    distance = jnp.linspace(0., 60., n_points)
    exp_rates = jax.vmap(expected_rates(DEFAULT_TX_POWER))(distance)

    for mode, (exp_rate, data_rate) in enumerate(zip(exp_rates.T, wifi_modes_rates)):
        plt.plot(distance, exp_rate, c='C0', linestyle='solid' if mode % 2 == 1 else 'dashdot')
        plt.axhline(data_rate, alpha=0.4, c='C0', linestyle='--')
        plt.text(0.5, data_rate + 1, f'MCS {mode}', fontsize=6)

    plt.ylabel(r'Expected data rate $\lambda$ [Mb/s]')
    plt.xlabel(r'Distance $\rho$ [m]')
    plt.xlim((distance.min(), distance.max()))
    plt.ylim((0., 130.))


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    plt.figure(figsize=(COLUMN_WIDTH, 2 * COLUMN_WIDTH / (1 + jnp.sqrt(5))))

    plot_data_rates()

    plt.savefig(f'data-rates.pdf', bbox_inches='tight')
    plt.clf()
