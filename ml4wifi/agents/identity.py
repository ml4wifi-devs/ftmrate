import jax
import tensorflow_probability.substrates.jax as tfp
from chex import dataclass, Scalar, PRNGKey

from ml4wifi.agents.base_managers_container import BaseAgent, BaseManagersContainer
from ml4wifi.utils.distributions import DeterministicQ

tfd = tfp.distributions


@dataclass
class IdentityState:
    distance: Scalar


def identity() -> BaseAgent:

    def init(key: PRNGKey) -> IdentityState:
        return IdentityState(distance=0.)

    def update(state: IdentityState, key: PRNGKey, distance: Scalar, time: Scalar) -> IdentityState:
        return IdentityState(distance=distance)

    def sample(state: IdentityState, key: PRNGKey, time: Scalar) -> tfd.Distribution:
        return DeterministicQ(loc=state.distance)

    return BaseAgent(
        init=jax.jit(init),
        update=jax.jit(update),
        sample=jax.jit(sample)
    )


class ManagersContainer(BaseManagersContainer):
    def __init__(self, seed: int) -> None:
        super().__init__(seed, identity)
