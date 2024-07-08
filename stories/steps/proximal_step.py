from abc import ABC, abstractmethod
from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
import optax
import jax


class ProximalStep(ABC):
    """This abstract class defines the interface for proximal steps. Given a potential
    function, the proximal step updates the input distribution :math:`\\mu_t`.

    A proximal step should implement both an inference step and a training step. The
    inference step is used to generate samples from the model, while the training step
    should eb differentiable and is used to train the model parameters."""

    @abstractmethod
    def inference_step(
        self,
        x: jnp.ndarray,
        a: jnp.ndarray,
        potential_fun: Callable,
        tau: float,
    ) -> jnp.ndarray:
        """Given a distribution of cells :math:`\\mu_t` and a potential function, this
        function returns :math:`\\mu_{t+\\tau}`.

        Args:
            x (jnp.ndarray): The input distribution of size (batch_size, n_dims)
            a (jnp.ndarray): The input histogram (batch_size,)
            potential_fun (Callable): A potential function.
            tau (float): The time step, which should be greater than 0.

        Returns:
            jnp.ndarray: The updated distribution of size (batch_size, n_dims).
        """
        pass

    @abstractmethod
    def training_step(
        self,
        x: jnp.ndarray,
        a: jnp.ndarray,
        potential_network: nn.Module,
        potential_params: optax.Params,
        tau: float,
    ) -> jnp.ndarray:
        """Given a distribution of cells :math:`\\mu_t` and a potential function
        parameterized by a neural network, this function returns :math:`\\mu_{t+\\tau}`.

        Args:
            x (jnp.ndarray): The input distribution of size (batch_size, n_dims)
            a (jnp.ndarray): The input histogram (batch_size,)
            potential_network (nn.Module): A potential function parameterized by a
            neural network.
            potential_params (optax.Params): The parameters of the potential network.
            tau (float): The time step, which should be greater than 0.

        Returns:
            jnp.ndarray: The updated distribution of size (batch_size, n_dims)
        """
        pass

    def chained_inference_steps(
        self,
        x: jnp.ndarray,
        a: jnp.ndarray,
        potential_fun: Callable,
        tau: float,
        n_steps: int = 1,
    ) -> jnp.ndarray:
        """Given a distribution of cells :math:`\\mu_t` and a potential function, this
        function returns :math:`\\mu_{t+\\tau}`, using `n_steps` steps of size
        `tau/n_steps`.

        Args:
            x (jnp.ndarray): The input distribution of size (batch_size, n_dims)
            a (jnp.ndarray): The input histogram (batch_size,)
            potential_fun (Callable): A potential function.
            tau (float): The time step, which should be greater than 0.
            n_steps (int): The number of steps to take.

        Returns:
            jnp.ndarray: The updated distribution of size (batch_size, n_dims).
        """

        y = x.copy()
        for _ in range(n_steps):
            y = self.inference_step(y, a, potential_fun, tau / n_steps)
        return y

    def chained_training_steps(
        self,
        x: jnp.ndarray,
        a: jnp.ndarray,
        potential_network: nn.Module,
        potential_params: optax.Params,
        tau: float,
        n_steps: int = 1,
    ) -> jnp.ndarray:
        """Given a distribution of cells :math:`\\mu_t` and a potential function
        parameterized by a neural network, this function returns :math:`\\mu_{t+\\tau}`,
        using `n_steps` steps of size `tau/n_steps`.

        Args:
            x (jnp.ndarray): The input distribution of size (batch_size, n_dims)
            a (jnp.ndarray): The input histogram (batch_size,)
            potential_network (nn.Module): A potential function parameterized by a
            neural network.
            potential_params (optax.Params): The parameters of the potential network.
            tau (float): The time step, which should be greater than 0.
            n_steps (int): The number of steps to take.

        Returns:
            jnp.ndarray: The updated distribution of size (batch_size, n_dims)
        """

        return jax.lax.fori_loop(
            0,
            n_steps,
            lambda i, x: self.training_step(
                x, a, potential_network, potential_params, tau / n_steps
            ),
            x,
        )
