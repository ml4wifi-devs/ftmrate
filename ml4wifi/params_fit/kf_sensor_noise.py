from argparse import ArgumentParser
from typing import Tuple

import os.path
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
from chex import Array, Scalar
from tensorflow_probability.substrates import jax as tfp
from tqdm import trange

from ml4wifi.utils.wifi_specs import distance_noise
from ml4wifi.params_fit import load_parameters_file, CSV_FILES_DIR

tfd = tfp.distributions


optimizer: optax.GradientTransformation = None

DOMAIN = jnp.linspace(-3., 5., 501)
distance_noise_prob = distance_noise.prob(DOMAIN)


@jax.jit
def kl_div(x: Array, y: Array) -> Scalar:
    """
    Calculates KL divergence between two sets of samples
    """

    return jnp.sum(x * jnp.log(x / y))


def loss(scale: Scalar) -> Scalar:
    """
    Loss function defined as KL divergence between `distance_noise` distribution and modeled normal distribution
    """

    return kl_div(tfd.Normal(0., jax.nn.softplus(scale)).prob(DOMAIN), distance_noise_prob)


def step_fn(opt_state: optax.OptState, scale: Scalar) -> Tuple:
    """
    Performs one step of a gradient optimization.
    """

    loss_val, grad = jax.value_and_grad(loss)(scale)
    updates, opt_state = optimizer.update(grad, opt_state, scale)
    scale = optax.apply_updates(scale, updates)

    return scale, opt_state, loss_val


def params_fit(init_scale: Scalar, lr: Scalar, n_steps: Scalar) -> Tuple:
    """
    Fits the most similar normal distribution to `distance_noise` in terms of KL divergence.
    """

    global optimizer

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(init_scale)

    scale = init_scale
    losses = []

    for _ in trange(n_steps):
        scale, opt_state, loss_val = step_fn(opt_state, scale)
        losses.append(loss_val)
    
    return jax.nn.softplus(scale), losses


def plot_distributions(true_dist: tfd.Distribution, model_dist: tfd.Distribution) -> None:
    plt.plot(DOMAIN, true_dist.prob(DOMAIN), label='True Distribution')
    plt.plot(DOMAIN, model_dist.prob(DOMAIN), label='Model Distribution')
    plt.ylabel('pdf')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--init_scale', default=1.0, type=float)
    args.add_argument('--lr', default=0.01, type=float)
    args.add_argument('--n_steps', default=250, type=int)
    args.add_argument('--output_name', default='parameters.csv', type=str)
    args.add_argument('--plot', action='store_true', default=False)
    args = args.parse_args()

    plot_distributions(distance_noise, tfd.Normal(0.0, args.init_scale)) if args.plot else None

    params_df = load_parameters_file(name=args.output_name)
    scale, losses = params_fit(args.init_scale, args.lr, args.n_steps)

    print(f'Scale: {scale}')
    plt.plot(losses)
    plt.title('loss')
    plt.show()

    plot_distributions(distance_noise, tfd.Normal(0.0, scale)) if args.plot else None

    params_df.loc[:, 'kf_sensor_noise'] = scale
    params_df.to_csv(os.path.join(CSV_FILES_DIR, args.output_name), index=False)

