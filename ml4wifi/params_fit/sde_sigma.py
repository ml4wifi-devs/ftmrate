from argparse import ArgumentParser
from functools import partial
from typing import Tuple

import os.path
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import seaborn as sns
import tensorflow_probability.substrates.numpy as tfp
from chex import Array, Scalar, PRNGKey

from ml4wifi.params_fit import load_parameters_file, CSV_FILES_DIR
from ml4wifi.utils.measurement_manager import DEFAULT_INTERVAL


@partial(jax.jit, static_argnames=['n_steps', 'n_samples'])
def vi(
        lr: Scalar,
        decay: Scalar,
        n_samples: jnp.int32,
        n_steps: jnp.int32,
        data: Array,
        key: PRNGKey
) -> Tuple:
    """
    Fits `level_scale` and `slope_scale` parameters of the LocalLinearTrendStateSpaceModel using Variational Inference.
    Mentioned parameters correspond to the `sigma_x` and `sigma_v` values in SDE model.
    """

    init_key, vi_key, sample_key = jax.random.split(key, 3)

    model = tfp.sts.LocalLinearTrend(observed_time_series=data)
    init_fn, build_surrogate_fn = tfp.sts.build_factored_surrogate_posterior_stateless(model)

    optimized_parameters, losses = tfp.vi.fit_surrogate_posterior_stateless(
        target_log_prob_fn=model.joint_distribution(data).log_prob,
        build_surrogate_posterior_fn=build_surrogate_fn,
        initial_parameters=init_fn(init_key),
        optimizer=optax.rmsprop(lr, decay),
        num_steps=n_steps,
        seed=vi_key
    )

    variational_posteriors = build_surrogate_fn(*optimized_parameters)
    return variational_posteriors.sample((n_samples,), sample_key), losses


@partial(jax.jit, static_argnames=['n_frames'])
def generate_run_fn(data: Array, key: PRNGKey, i: jnp.int32, n_frames: jnp.int32) -> Tuple:
    """
    Generates one RWPM run (only steps with measurement).
    """

    data_key, key = jax.random.split(key)
    _, distance_measurement, *_ = generate_rwpm(
        key=data_key,
        time_total=int((n_frames - 1) * DEFAULT_INTERVAL),
        measurement_interval=DEFAULT_INTERVAL,
        frames_total=n_frames
    )
    data = data.at[i, :, 0].set(distance_measurement)

    return data, key


def params_fit(
        lr: Scalar,
        decay: Scalar,
        n_datasets: int,
        n_frames: int,
        n_samples: int,
        n_steps: int,
        key: PRNGKey,
        plot: bool = False,
        **_
) -> Tuple:
    """
    Fits sigma values for SDE model on `n_datasets` datasets.
    """

    data = jnp.empty((n_datasets, n_frames, 1))

    for i in range(n_datasets):
        data, key = generate_run_fn(data, key, i, n_frames)

    samples, loss = vi(lr, decay, n_samples, n_steps, data, key)
    sigma_x = samples['level_scale'].flatten()
    sigma_v = samples['slope_scale'].flatten()

    if plot:
        plt.plot(loss.mean(axis=1))
        plt.title('loss')
        plt.show()

        sns.histplot(sigma_x, kde=True, stat='probability')
        plt.title('sigma_x')
        plt.show()

        sns.histplot(sigma_v, kde=True, stat='probability')
        plt.title('sigma_v')
        plt.show()

    return sigma_x.mean(), sigma_v.mean()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--lr', default=0.01, type=float)
    args.add_argument('--decay', default=0.95, type=float)
    args.add_argument('--n_datasets', default=2000, type=int)
    args.add_argument('--n_frames', default=2001, type=int)
    args.add_argument('--n_samples', default=2000, type=int)
    args.add_argument('--n_steps', default=800, type=int)
    args.add_argument('--output_name', default='parameters.csv', type=str)
    args.add_argument('--plot', action='store_true', default=False)
    args.add_argument('--seed', default=42, type=int)
    args = args.parse_args()

    params_df = load_parameters_file(name=args.output_name)

    sigma_x, sigma_v = params_fit(**vars(args), key=jax.random.PRNGKey(args.seed))
    print(f'\nSigma_x: {sigma_x}\nSigma_v: {sigma_v}')

    params_df.loc[:, 'sigma_x'] = sigma_x
    params_df.loc[:, 'sigma_v'] = sigma_v
    params_df.to_csv(os.path.join(CSV_FILES_DIR, args.output_name), index=False)
