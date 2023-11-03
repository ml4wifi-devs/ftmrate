import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import seaborn as sns
import matplotlib.pyplot as plt

from ml4wifi.utils.wifi_specs import *

tfb = tfp.bijectors


if __name__ == '__main__':
    n_samples = 10e4
    key = jax.random.PRNGKey(42)

    true_distance = 20
    true_snr = REFERENCE_SNR - (REFERENCE_LOSS + 10 * EXPONENT * jnp.log10(true_distance))

    noisy_distance = tfb.Shift(true_distance)(distance_noise)
    noisy_snr = distance_to_snr(tfb.Softplus()(noisy_distance))
    noisy_rate = expected_rates_log_distance(DEFAULT_TX_POWER)(tfb.Softplus()(noisy_distance))

    sns.histplot(data=noisy_distance.sample(n_samples, key), kde=True, bins=50, stat='density')
    plt.xlabel("Distance uncertainty [m]")
    plt.axvline(true_distance, color='r', linestyle='--', label='True distance')
    plt.legend()
    plt.tight_layout()
    plt.savefig("distance.pdf", bbox_inches='tight')
    plt.clf()

    sns.histplot(data=noisy_snr.sample(n_samples, key), kde=True, bins=50, stat='density')
    plt.xlabel("SNR uncertainty [dB]")
    plt.axvline(true_snr, color='r', linestyle='--', label='True SNR')
    plt.legend()
    plt.tight_layout()
    plt.savefig("snr.pdf", bbox_inches='tight')
    plt.clf()

    sns.histplot(data=noisy_rate.sample(n_samples, key)[:, 6:10], bins=100, legend=False, stat='density')
    plt.xlabel("Data rate uncertainty [Mb/s]")
    plt.legend(title='MCS', loc='upper left', labels=range(10, 4, -1))
    plt.tight_layout()
    plt.savefig("rates.pdf", bbox_inches='tight')
    plt.clf()
