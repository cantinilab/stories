from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxopt
import optax
from .proximal_step import ProximalStep
from ott.neural.models import ICNN


class ICNNImplicitStep(ProximalStep):
    """This class defines an implicit proximal step corresponding to the squared
    Wasserstein distance, assuming the transportation plan is the identity (each cell
    mapped to itself). This step is "implicit" in the sense that instead of computing a
    velocity field it predicts the next timepoint as an argmin and thus requires solving
    an optimization problem.

    Args:
        maxiter (int, optional): The maximum number of iterations for the optimization
            loop. Defaults to 100.
        implicit_diff (bool, optional): Whether to differentiate implicitly through the
            optimization loop. Defaults to True.
        log_callback (Callable, optional): A callback used to log the proximal loss.
            Defaults to None.
        tol (float, optional): The tolerance for the optimization loop. Defaults to 1e-8.
    """

    def __init__(
        self,
        maxiter: int = 100,
        implicit_diff: bool = True,
        log_callback: Callable | None = None,
        tol: float = 1e-8,
    ):
        self.log_callback = log_callback
        self.maxiter = maxiter
        self.implicit_diff = implicit_diff
        self.tol = tol

        self.opt_hyperparams = {
            "maxiter": maxiter,
            "implicit_diff": implicit_diff,
            "tol": tol,
        }

    def inference_step(
        self,
        x: jax.Array,
        a: jax.Array,
        potential_fun: Callable,
        tau: float,
    ) -> jax.Array:
        """Performs an implicit step on the input distribution and returns the
        updated distribution, given a potential function. If logging is available,
        logs the proximal cost.

        Args:
            x (jax.Array): The input distribution of size (batch_size, n_dims)
            a (jax.Array): The input histogram (batch_size,)
            potential_fun (Callable): A potential function.
            tau (float): The time step, which should be greater than 0.

        Returns:
            jax.Array: The updated distribution of size (batch_size, n_dims).
        """

        icnn = ICNN(dim_data=x.shape[1], dim_hidden=[32, 32, 32])
        params_icnn = icnn.init(jax.random.PRNGKey(0), x)["params"]

        # Define a helper function to compute the proximal cost.
        def proximal_cost(params_icnn, inner_x, inner_a, inner_tau):
            params_dict = {"params": params_icnn}
            icnn_fun = lambda u: jax.grad(icnn.apply, argnums=1)(params_dict, u)
            y = jax.vmap(icnn_fun)(inner_x)

            potential_term = jnp.sum(inner_a * potential_fun(y))
            prox_term = jnp.sum(inner_a.reshape(-1, 1) * (inner_x - y) ** 2)
            return potential_term + inner_tau * 0.5 * prox_term

        # Define the optimizer.
        opt = jaxopt.OptaxSolver(
            fun=proximal_cost, opt=optax.adam(1e-2), **self.opt_hyperparams
        )

        @jax.jit
        def jitted_update(y, state):
            return opt.update(y, state, inner_x=x, inner_a=a, inner_tau=tau)

        # Run the gradient descent, and log the proximal cost.
        state = opt.init_state(params_icnn, inner_x=x, inner_a=a, inner_tau=tau)
        for _ in range(self.maxiter):
            params_icnn, state = jitted_update(params_icnn, state)
            if self.log_callback:
                self.log_callback({"proximal_cost": state.error})
            if state.error < self.tol:
                break

        # Return the new omics coordinates.
        y = jax.vmap(
            lambda u: jax.grad(icnn.apply, argnums=1)({"params": params_icnn}, u)
        )(x)
        return y

    def training_step(
        self,
        x: jax.Array,
        a: jax.Array,
        potential_network: nn.Module,
        potential_params: optax.Params,
        tau: float,
    ) -> jax.Array:
        """Performs an implicit step on the input distribution and returns the
        updated distribution. This function differs from the inference step in that it
        takes a potential network as input and returns the updated distribution. Logging
        is not available in this function because it prevents implicit differentiation.

        Args:
            x (jax.Array): The input distribution of size (batch_size, n_dims)
            a (jax.Array): The input histogram (batch_size,)
            potential_network (nn.Module): A potential function parameterized by a
            neural network.
            potential_params (optax.Params): The parameters of the potential network.
            tau (float): The time step, which should be greater than 0.

        Returns:
            jax.Array: The updated distribution of size (batch_size, n_dims).
        """

        icnn = ICNN(dim_data=x.shape[1], dim_hidden=[32, 32, 32])
        params_icnn = icnn.init(jax.random.PRNGKey(0), x)["params"]

        # Define a helper function to compute the proximal cost.
        def proximal_cost(params_icnn, inner_x, inner_pot_params, inner_tau, inner_a):
            params_dict = {"params": params_icnn}
            icnn_fun = lambda u: jax.grad(icnn.apply, argnums=1)(params_dict, u)
            y = jax.vmap(icnn_fun)(inner_x)
            pot_fun = lambda u: potential_network.apply(inner_pot_params, u)
            potential_term = jnp.sum(inner_a * pot_fun(y))
            prox_term = jnp.sum(inner_a.reshape(-1, 1) * (inner_x - y) ** 2)
            return potential_term + inner_tau * 0.5 * prox_term

        # Define the optimizer.
        opt = jaxopt.OptaxSolver(
            fun=proximal_cost, opt=optax.adam(1e-2), **self.opt_hyperparams
        )

        # Run the optimization loop.
        params_icnn, _ = opt.run(
            params_icnn,
            inner_x=x,
            inner_pot_params=potential_params,
            inner_tau=tau,
            inner_a=a,
        )

        # Return the new omics coordinates.
        icnn_fun = lambda u: jax.grad(icnn.apply, argnums=1)({"params": params_icnn}, u)
        return jax.vmap(icnn_fun)(x)
