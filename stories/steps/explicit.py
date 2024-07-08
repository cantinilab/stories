from typing import Callable

import flax.linen as nn
from flax.core.scope import VariableDict
from jax import grad, vmap, Array
from .proximal_step import ProximalStep


class ExplicitStep(ProximalStep):
    """This class implements the explicit proximal step associated with the Wasserstein
    distance, i.e. :math:`v = -\nabla J(x)`, where :math:`J` is a potential."""

    def inference_step(
        self,
        x: Array,
        a: Array,
        potential_fun: Callable,
        tau: float,
    ) -> Array:
        """Performs an explicit step on the input distribution and returns the
        updated distribution, given a potential function.

        Args:
            x (Array): The input distribution of size (batch_size, n_dims)
            a (Array): The input histogram (batch_size,)
            potential_fun (Callable): A potential function.
            tau (float): The time step, which should be greater than 0.

        Returns:
            Array: The updated distribution of size (batch_size, n_dims).
        """

        # The explicit step is a step of gradient descent.
        return x - tau * vmap(grad(potential_fun))(x)

    def training_step(
        self,
        x: Array,
        a: Array,
        potential_network: nn.Module,
        potential_params: VariableDict,
        tau: float,
    ) -> Array:
        """Performs an explicit step on the input distribution and returns the
        updated distribution. This function differs from the inference step in that it
        takes a potential network as input and returns the updated distribution.

        Args:
            x (Array): The input distribution of size (batch_size, n_dims)
            a (Array): The input histogram (batch_size,)
            potential_network (nn.Module): A potential function parameterized by a
            neural network.
            potential_params (optax.Params): The parameters of the potential network.
            tau (float): The time step, which should be greater than 0.

        Returns:
            Array: The updated distribution of size (batch_size, n_dims).
        """

        # Turn the potential network into a function.
        potential_fun = lambda u: potential_network.apply(potential_params, u)

        # Then simply apply the inference step since it's differentiable.
        return self.inference_step(x, a, potential_fun, tau)
