from argparse import ArgumentParser
from functools import partial
from typing import Tuple, Mapping

import os.path
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from chex import Array, Scalar, PRNGKey
from tqdm import trange

from ml4wifi.agents.exponential_smoothing import exponential_smoothing, ExponentialSmoothingState
from ml4wifi.params_fit import generate_rwpm, load_parameters_file, CSV_FILES_DIR
from ml4wifi.utils.measurement_manager import DEFAULT_INTERVAL

optimizer: optax.GradientTransformation = None

DUMMY_KEY = jax.random.PRNGKey(42)


@jax.jit
def loss(
        params: Tuple,
        data: Array
) -> Scalar:
    """
    Returns MSE loss between true distance and distance estimated by ExponentialSmoothing agent given noisy measurements.
    """

    agent = exponential_smoothing(*jax.tree_map(jax.nn.sigmoid, params))

    def one_run_fn(_, run: Array):

        def step_fn(state: ExponentialSmoothingState, step: Array) -> Tuple:
            measurement, time = step
            state = agent.update(state, DUMMY_KEY, measurement, time)
            return state, agent.sample(state, DUMMY_KEY, time).loc

        distance_true, distance_measurement, time = run
        _, distance_estimated = jax.lax.scan(step_fn, agent.init(DUMMY_KEY), jnp.stack([distance_measurement, time], axis=1))
        distance_true, distance_estimated = jax.tree_map(jnp.abs, (distance_true, distance_estimated))

        return None, optax.l2_loss(distance_true, distance_estimated).mean()

    _, losses = jax.lax.scan(one_run_fn, None, data)
    return losses.mean()


@jax.jit
def step_fn(
        opt_state: optax.OptState,
        params: Mapping,
        data: Tuple
) -> Tuple:
    """
    Performs one step of a gradient optimization.
    """

    loss_val, grad = jax.value_and_grad(loss)(params, data)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val


@partial(jax.jit, static_argnames=['n_frames'])
def generate_run_fn(data: Array, key: PRNGKey, i: jnp.int32, n_frames: jnp.int32) -> Tuple:
    """
    Generates one RWPM run (only steps with measurement).
    """

    data_key, key = jax.random.split(key)
    distance_true, distance_measurement, _, time = generate_rwpm(
        key=data_key,
        time_total=int((n_frames - 1) * DEFAULT_INTERVAL),
        measurement_interval=DEFAULT_INTERVAL,
        frames_total=n_frames
    )
    data = data.at[i, 0, :].set(distance_true)
    data = data.at[i, 1, :].set(distance_measurement)
    data = data.at[i, 2, :].set(time)

    return data, key


def params_fit(
        lr: Scalar,
        n_datasets: int,
        n_frames: int,
        n_steps: int,
        key: PRNGKey,
        plot: bool = False,
        **_
) -> Tuple:
    """
    Fits the best parameters for ExponentialSmoothing agent on `n_datasets` datasets.
    """

    global optimizer

    params = (-1.0, -0.5)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    data = jnp.empty((n_datasets, 3, n_frames))

    for i in range(n_datasets):
        data, key = generate_run_fn(data, key, i, n_frames)

    alphas, betas, losses = [], [], []

    for _ in trange(n_steps):
        params, opt_state, loss_val = step_fn(opt_state, params, data)

        alpha, beta = jax.tree_map(jax.nn.sigmoid, params)
        alphas.append(alpha)
        betas.append(beta)
        losses.append(loss_val)

    if plot:
        plt.plot(alphas)
        plt.title('alpha')
        plt.show()

        plt.plot(betas)
        plt.title('beta')
        plt.show()

        plt.plot(losses)
        plt.title('loss')
        plt.show()

    return jax.tree_map(jax.nn.sigmoid, params)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--lr', default=0.01, type=float)
    args.add_argument('--n_datasets', default=2000, type=int)
    args.add_argument('--n_frames', default=2001, type=int)
    args.add_argument('--n_steps', default=400, type=int)
    args.add_argument('--output_name', default="parameters.csv", type=str)
    args.add_argument('--plot', action="store_true", default=False)
    args.add_argument('--seed', default=42, type=int)
    args = args.parse_args()

    params_df = load_parameters_file(name=args.output_name)

    alpha, beta = params_fit(**vars(args), key=jax.random.PRNGKey(args.seed))
    print(f'\nAlpha: {alpha}\nBeta:  {beta}')

    params_df.loc[:, "es_alpha"] = alpha
    params_df.loc[:, "es_beta"] = beta
    params_df.to_csv(os.path.join(CSV_FILES_DIR, args.output_name), index=False)
