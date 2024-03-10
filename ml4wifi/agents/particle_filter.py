from functools import partial
from typing import Tuple, Dict

import jax
import jax.numpy as jnp
from chex import Scalar, PRNGKey, dataclass, Array
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax.experimental.mcmc import SequentialMonteCarlo, SequentialMonteCarloResults, \
    WeightedParticles

from ml4wifi.agents.base_managers_container import BaseAgent, BaseManagersContainer
from ml4wifi.envs import ContinuousLocalLinearTrend
from ml4wifi.utils.distributions import FiniteDiscreteQ
from ml4wifi.utils.wifi_specs import distance_noise

tfb = tfp.bijectors
tfd = tfp.distributions


@dataclass
class ParticleFilterState:
    particles: WeightedParticles
    results: SequentialMonteCarloResults
    time: Scalar


F_fn, Q_fn = ContinuousLocalLinearTrend().jaxify(cholesky=True)


def initial_state_prior_fn(low: Array, high: Array) -> tfd.Distribution:
    return tfd.JointDistributionNamed(dict(x=tfd.Independent(tfd.Uniform(low, high), reinterpreted_batch_ndims=1),
                                           y=tfd.Independent(tfd.Uniform(low, high), reinterpreted_batch_ndims=1)))


def transition_fn(particles: Dict[str, Array], t_delta: Scalar) -> tfd.Distribution:
    # Variance is a random walk; is the same in components and independent axes.
    F = F_fn(t_delta)
    Q = Q_fn(t_delta)

    d = dict(
        x=tfd.MultivariateNormalTriL(particles['x'] @ F.T, Q),
        y=tfd.MultivariateNormalTriL(particles['y'] @ F.T, Q)
    )
    return tfd.JointDistributionNamed(d)


def _dict_2r(particles: Dict[str, Array]) -> Array:
    """
    Computes distance from X and Y coordinates.

    Parameters
    ----------
    particles : dict
        Dictionary of particles with keys 'x' and 'y'
        
    Returns
    -------
    distance : Array
        Distance to the point (0,0)
    """

    r = jnp.sqrt(jnp.square(particles['x'][..., 0]) + jnp.square(particles['y'][..., 0]))
    return r


def observation_fn(particles: Dict[str, Array]) -> tfd.Distribution:
    r = _dict_2r(particles)
    return tfb.Shift(r)(distance_noise)


def propose_and_update_weights(
        step: jnp.int32,
        particles: WeightedParticles,
        seed: PRNGKey,
        distance: Scalar,
        measured: jnp.bool_,
        t_delta: Scalar
) -> WeightedParticles:
    proposed_particles = transition_fn(particles.particles, t_delta).sample(seed=seed)

    return jax.lax.cond(
        measured,
        lambda: WeightedParticles(proposed_particles,
                                  particles.log_weights + observation_fn(proposed_particles).log_prob(distance)),
        lambda: WeightedParticles(proposed_particles, particles.log_weights)
    )


def particle_filter(
        min_xy: Scalar = -50.0,
        max_xy: Scalar = 50.0,
        min_v: Scalar = -4.0,
        max_v: Scalar = 4.0,
        particles_num: jnp.int32 = 1024
) -> BaseAgent:
    """
    Parameters
    ----------
    min_xy : float, default=0.0
        Minimal particle position corresponding to X and Y coordinates
    max_xy : float, default=50.0
        Maximal particle position corresponding to X and Y coordinates
    min_v : float, default=-4.0
        Minimal particle position corresponding to velocity
    max_v : float, default=4.0
        Maximal particle position corresponding to velocity
    particles_num : int, default=1000
        Number of particles in filter

    Returns
    -------
    agent : BaseAgent
        Container for functions of the PF
    """

    def init(key: PRNGKey) -> ParticleFilterState:
        """
        Returns the Particle Filter agent initial state.

        Parameters
        ----------
        key : PRNGKey
            JAX random generator key

        Returns
        -------
        state : ParticleFilterState
            Initial Particle Filter state
        """

        kernel = SequentialMonteCarlo(propose_and_update_weights)
        initial_state = WeightedParticles(
            particles=initial_state_prior_fn([min_xy, min_v], [max_xy, max_v]).sample(particles_num, seed=key),
            log_weights=jnp.zeros(particles_num)
        )

        return ParticleFilterState(
            particles=initial_state,
            results=kernel.bootstrap_results(initial_state),
            time=0.0
        )

    def _one_step(
            state: ParticleFilterState,
            key: PRNGKey,
            distance: Scalar,
            measured: jnp.bool_,
            time: Scalar
    ) -> Tuple[WeightedParticles, SequentialMonteCarloResults]:
        kernel = SequentialMonteCarlo(partial(
            propose_and_update_weights,
            distance=distance,
            measured=measured,
            t_delta=time - state.time
        ))
        return kernel.one_step(state.particles, state.results, key)

    def update(
            state: ParticleFilterState,
            key: PRNGKey,
            distance: Scalar,
            time: Scalar
    ) -> ParticleFilterState:
        """
        Performs one step of the Particle Filter algorithm, returns the updated state of the agent.

        Parameters
        ----------
        state : ParticleFilterState
            Previous agent state
        key : PRNGKey
            JAX random generator key
        distance : float
            Distance measurement
        time : float
            Current time

        Returns
        -------
        state : ParticleFilterState
            Updated agent state
        """

        particles, results = _one_step(state, key, distance, True, time)
        return ParticleFilterState(
            particles=particles,
            results=results,
            time=time
        )

    def sample(
            state: ParticleFilterState,
            key: PRNGKey,
            time: Scalar
    ) -> tfd.Distribution:
        """
        Estimates distance distribution from current Particle Filter state.

        Parameters
        ----------
        state : ParticleFilterState
            Current agent state
        key : PRNGKey
            JAX random generator key
        time : float
            Current time

        Returns
        -------
        dist : tfd.Distribution
            Predicted distance distribution at t=time
        """

        particles, _ = _one_step(state, key, jnp.nan, False, time)
        r = _dict_2r(particles.particles)

        sorted_idxs = jnp.argsort(r)
        outcomes, logits = r[sorted_idxs], particles.log_weights[sorted_idxs]

        return FiniteDiscreteQ(outcomes, logits)

    return BaseAgent(
        init=jax.jit(init),
        update=jax.jit(update),
        sample=jax.jit(sample)
    )


class ManagersContainer(BaseManagersContainer):
    def __init__(self, seed: int) -> None:
        super().__init__(seed, particle_filter)
