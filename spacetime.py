from typing import Callable

from dataclasses import dataclass
from anndata import AnnData
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import orbax_utils
from flax.training.early_stopping import EarlyStopping
from jax.random import PRNGKey
from jax import Array
import uuid
from orbax.checkpoint.args import StandardRestore
from optax import GradientTransformation
from orbax.checkpoint import CheckpointManager
from orbax.checkpoint.args import StandardSave
from tqdm import tqdm

from .steps.proximal_step import ProximalStep
from .steps.explicit import ExplicitStep
from .potentials import MLPPotential
from .tools import DataLoader, default_checkpoint_manager
from .loss import loss_fn


@dataclass
class SpaceTime:
    """Wasserstein gradient flow model for spatio-temporal omics data.

    Args:
        potential (nn.Module, optional): The potential function. Defaults to a MLP.
        proximal_step (ProximalStep, optional): The proximal step. Defaults forward Euler
        n_steps (int, optional): The number of steps. Defaults to 1.
        teacher_forcing (bool, optional): Use teacher forcing. Defaults to True.
        quadratic (bool, optional): Use a Fused GW loss. Defaults to True.
        debias (bool, optional): Whether to debias the loss. Defaults to True.
        epsilon (float, optional): The (relative) entropic reg. Defaults to 0.01.
        log_callback (Callable, optional): The callback for logging. Defaults to None.
        quadratic_weight (float, optional): Weight of the quadratic term, in [0, 1].
    """

    potential: nn.Module = MLPPotential()
    proximal_step: ProximalStep = ExplicitStep()
    n_steps: int = 1
    teacher_forcing: bool = True
    quadratic: bool = True
    debias: bool = True
    epsilon: float = 0.01
    log_callback: Callable | None = None
    quadratic_weight: float = 5e-3

    def fit(
        self,
        adata: AnnData,
        time_key: str,
        omics_key: str,
        space_key: str,
        weight_key: str | None = None,
        optimizer: GradientTransformation = optax.adamw(1e-2),
        max_iter: int = 10_000,
        batch_size: int = 1_000,
        train_val_split: float = 0.75,
        min_delta: float = 0.0,
        patience: int = 150,
        checkpoint_manager: CheckpointManager | str | None = None,
        key: Array = PRNGKey(0),
        restore: bool = True,
    ) -> None:
        """Fit the model.

        Args:
            adata (AnnData): The AnnData object.
            time_key (str): The name of the time observation.
            omics_key (str): The name of the obsm field containing cell coordinates.
            space_key (str): The name of the obsm field containing space coordinates.
            weight_key (str): The name of the obs field containing weights.
            optimizer (GradientTransformation, optional): The optimizer.
            max_iter (int, optional): The max number of iterations. Defaults to 10_000.
            batch_size (int, optional): The batch size. Defaults to 1_000.
            train_val_split (float, optional): The proportion of train in the split.
            min_delta (float, optional): The minimum delta for early stopping.
            patience (int, optional): The patience for early stopping.
            checkpoint_manager (CheckpointManager, optional): Checkpoint manager or path.
            key (KeyArray, optional): The random key. Defaults to PRNGKey(0).
            restore (bool, optional): By default, load the checkpointed params.
        """

        # Assert the quadratic weight is strictly between 0 and 1 if self.quadratic.
        if self.quadratic:
            assert (
                0 < self.quadratic_weight < 1
            ), "Relative quadratic weight must be strictly between 0 and 1."

        # If the checkpoint_manager is a string, make the default one.
        if not checkpoint_manager:
            checkpoint_manager = default_checkpoint_manager(f"/tmp/{uuid.uuid4()}")
        elif checkpoint_manager and isinstance(checkpoint_manager, str):
            checkpoint_manager = default_checkpoint_manager(checkpoint_manager)

        # Initialize the statistics for logging.
        self.train_it, self.train_losses = [], []
        self.val_it, self.val_losses = [], []

        # Create a data loader for the AnnData object.
        dataloader = DataLoader(
            adata,
            time_key=time_key,
            omics_key=omics_key,
            space_key=space_key,
            weight_key=weight_key,
            batch_size=batch_size,
            train_val_split=train_val_split,
        )

        # Split the cells ids into train and validation.
        split_key, key = jax.random.split(key)
        dataloader.make_train_val_split(split_key)

        # Compute tau from the timepoints.
        tau_diff = jnp.array(np.diff(dataloader.timepoints).astype(float))

        # Define some arguments shared by the train and validation loss.
        lkwargs = {"teacher_forcing": self.teacher_forcing, "potential": self.potential}
        lkwargs = {**lkwargs, "quadratic": self.quadratic, "debias": self.debias}
        lkwargs = {**lkwargs, "n_steps": self.n_steps, "epsilon": self.epsilon}
        lkwargs = {**lkwargs, "proximal_step": self.proximal_step, "tau_diff": tau_diff}
        lkwargs = {**lkwargs, "quadratic_weight": self.quadratic_weight}

        @jax.jit
        def jitted_update(params, opt_state, batch):
            """A jitted update function."""
            v, g = jax.value_and_grad(loss_fn)(params, batch, **lkwargs)
            updates, opt_state = optimizer.update(g, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, v

        @jax.jit
        def jitted_validation(params, batch):
            """A jitted validation function."""
            return loss_fn(params, batch, **lkwargs)

        # Initialize the parameters of the potential function and of the optimizer.
        init_key, key = jax.random.split(key)
        dummy_x = jnp.ones((batch_size, dataloader.n_features))  # Used to infer sizes.
        self.params = self.potential.init(init_key, dummy_x)
        opt_state = optimizer.init(self.params)

        # Define the early stopping criterion and checkpointing parameters.
        early_stop = EarlyStopping(min_delta=min_delta, patience=patience)

        # The training loop.
        pbar = tqdm(range(1, max_iter + 1))
        for it in pbar:
            # Choose whether to train or validate, weighted by train-val split.
            batch_key, key = jax.random.split(key, num=2)
            is_train = dataloader.train_or_val(it)

            # If train, update the parameters. If val, just compute the loss.
            if is_train:
                self.params, opt_state, train_loss = jitted_update(
                    self.params, opt_state, dataloader.next(batch_key, "train")
                )
                self.train_losses.append(train_loss)
                self.train_it.append(it)
            else:
                next_batch = dataloader.next(batch_key, "val")
                val_loss = jitted_validation(self.params, next_batch)
                self.val_losses.append(val_loss)
                self.val_it.append(it)

            # Recap the statistics for the current iteration.
            len_train, len_val = len(self.train_losses), len(self.val_losses)
            iteration_stats = {
                "iteration": it,
                "train_loss": np.inf if len_train == 0 else train_loss,
                "val_loss": np.inf if len_val == 0 else val_loss,
            }

            # Update the progress bar and log.
            pbar.set_postfix(iteration_stats)
            if self.log_callback:
                self.log_callback(iteration_stats)

            # Should we try early stopping and try to save the parameters?
            # If we have validation set, this is done every validation batch.
            # If we don't have validation set, this is done every train batch.
            update_train = train_val_split == 1.0 and is_train
            update_val = train_val_split < 1.0 and not is_train
            if update_train or update_val:
                # Check early stopping.
                last_l = self.train_losses[-1] if update_train else self.val_losses[-1]
                early_stop = early_stop.update(last_l)
                if early_stop.should_stop:
                    print("Met early stopping criteria, breaking...")
                    break

                # Try to save the parameters.
                metrics = {"loss": np.float64(last_l)}
                checkpoint_manager.save(
                    it, args=StandardSave(self.params), metrics=metrics
                )
                checkpoint_manager.wait_until_finished()

        if restore:
            self.best_step = checkpoint_manager.best_step()
            self.params = checkpoint_manager.restore(
                self.best_step, args=StandardRestore(self.params)
            )

    def transform(
        self,
        adata: AnnData,
        omics_key: str,
        tau: float,
        batch_size: int = 1_000,
        key: Array = PRNGKey(0),
    ) -> np.ndarray:
        """Transform an AnnData object.

        Args:
            adata (AnnData): The AnnData object to transform.
            omics_key (str): The obsm field containing the data to transform.
            tau (float, optional): The time step.
            batch_size (int, optional): The batch size. Defaults to 250.
            key (jax.Array, optional): The random key. Defaults to PRNGKey(0).

        Returns:
            np.ndarray: The predictions.
        """

        # Define the potential function.
        potential_fun = lambda u: self.potential.apply(self.params, u)

        # Helper function to transform a batch.
        def _transform_batch(idx_batch):
            x = jnp.array(adata.obsm[omics_key][idx_batch])
            a = jnp.ones(len(idx_batch)) / len(idx_batch)
            return self.proximal_step.chained_inference_steps(
                x, a, potential_fun, tau, self.n_steps
            )

        # Initizalize the prediction.
        x_pred = np.zeros(adata.obsm[omics_key].shape)

        # Iterate over batches and store the predictions.
        idx = np.array(jax.random.permutation(key, jnp.arange(x_pred.shape[0])))
        for idx_batch in np.array_split(idx, x_pred.shape[0] // batch_size):
            x_pred[idx_batch] = np.array(_transform_batch(idx_batch))

        # Return the predictions.
        return x_pred
