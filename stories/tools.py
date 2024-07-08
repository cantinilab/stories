from typing import Dict

from anndata import AnnData
import jax
from jax._src.random import KeyArray
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
import logging
from .spacetime import SpaceTime


@dataclass
class DataLoader:
    """DataLoader feeds data from an AnnData object to the model as JAX arrays. It
    samples without replacement for a given batch size.

    Args:
        adata (AnnData): The input AnnData object.
        time_key (str): The obs field with float time observations
        omics_key (str): The obsm field with the omics coordinates.
        space_key (str): The obsm field with the spatial coordinates.
        batch_size (int): The batch size.
        train_val_split (float, optional): The proportion of train in the split.
        weight_key (str, optional): The obs field with the marginal weights.
    """

    adata: AnnData
    time_key: str
    omics_key: str
    space_key: str
    batch_size: int
    train_val_split: float
    weight_key: str | None = None

    def __post_init__(self) -> None:
        """Initialize the DataLoader."""

        # Check that we have a valid time observation.
        assert_msg = "Time observations must be numeric."
        assert self.adata.obs[self.time_key].dtype.kind in "biuf", assert_msg

        # If time is valid, then we can get hold of the timepoints and their indices.
        self.timepoints = np.sort(np.unique(self.adata.obs[self.time_key]))
        get_idx = lambda t: np.where(self.adata.obs[self.time_key] == t)[0]
        self.idx = [get_idx(t) for t in self.timepoints]

        # Get the number of features, spatial dimensions, and timepoints.
        self.n_features = self.adata.obsm[self.omics_key].shape[1]
        self.n_space = self.adata.obsm[self.space_key].shape[1]
        self.n_timepoints = len(self.timepoints)

    def make_train_val_split(self, key: KeyArray) -> None:
        """Make a train/validation split. Must be called before training.

        Args:
            key (PRNGKey): The random number generator key for permutations.
        """

        # Initialize the train and validation indices, from which we will sample batches.
        self.idx_train, self.idx_val = [], []

        # Iterate over timepoints.
        for idx_t in self.idx:
            # Permute the indices in order to make the split random.
            key, key_permutation = jax.random.split(key)
            permuted_idx = jax.random.permutation(key_permutation, idx_t)

            # Split the indices between train and validation.
            split = int(self.train_val_split * len(idx_t))
            self.idx_train.append(permuted_idx[:split])
            self.idx_val.append(permuted_idx[split:])

        # Log some stats about the split.
        logging.info(f"Train (# cells): {[len(idx) for idx in self.idx_train]}")
        logging.info(f"Val (# cells): {[len(idx) for idx in self.idx_val]}")

    def next(self, key: KeyArray, train_or_val: str) -> Dict[str, jax.Array]:
        """Get the next batch from either train or val indices.

        Args:
            key (KeyArray): The random number generator key for sampling.
            train_or_val (str): Either "train" or "val".

        Returns:
            Dict[str, jax.Array]: A dictionary of JAX arrays.
        """

        # Check that we have a valid train or val argument.
        assert train_or_val in ["train", "val"], "Select either 'train' or 'val'."
        idx = self.idx_train if train_or_val == "train" else self.idx_val

        # Initialize the lists of omics and spatial coordinates over timepoints.
        x, space, a = [], [], []

        # Iterate over timepoints.
        for idx_t in idx:
            key, key_choice = jax.random.split(key)
            len_t = len(idx_t)

            # if the batch size is smaller or equal to the number of cells n, then we
            # want to sample a minibatch without replacement.
            if self.batch_size <= len_t:
                shape = (self.batch_size,)
                batch_idx = jax.random.choice(key_choice, idx_t, shape, replace=False)

                if self.weight_key:
                    batch_a = self.adata.obs[self.weight_key].iloc[batch_idx].values
                    batch_a /= batch_a.sum()
                else:
                    batch_a = np.ones(shape[0])
                    batch_a /= batch_a.sum()  # Weights are uniform.

            # if the batch size is greater than the number of cells n, then we want
            # to pad the cells with random cells and pad a with zeroes.
            else:
                shape = (self.batch_size - len_t,)
                batch_idx = jax.random.choice(key_choice, idx_t, shape, replace=True)
                batch_idx = np.concatenate((idx_t, batch_idx))

                if self.weight_key:
                    batch_a = self.adata.obs[self.weight_key].iloc[idx_t].values
                    batch_a = np.concatenate((batch_a, np.zeros(shape[0])))
                    batch_a /= batch_a.sum()
                else:
                    batch_a = np.concatenate((np.ones(len_t), np.zeros(shape[0])))
                    batch_a /= batch_a.sum()  # Weights are uniform.

            # Get the omics and spatial coordinates for the batch.
            x.append(self.adata.obsm[self.omics_key][batch_idx])
            space.append(self.adata.obsm[self.space_key][batch_idx])
            a.append(batch_a)

        # Return a dictionary of JAX arrays, the first axis being time.
        jnp_stack = lambda x: jnp.array(np.stack(x))
        return {"x": jnp_stack(x), "space": jnp_stack(space), "a": jnp_stack(a)}

    def train_or_val(self, iteration: int) -> bool:
        """Sample whether to train or validate.

        Args:
            iteration (int): The current iteration.

        Returns:
            bool: True for train, False for val.
        """
        freq_val = 1 - self.train_val_split
        return iteration % int(1 / freq_val) != 0


def compute_potential(
    adata: AnnData,
    model: SpaceTime,
    omics_key: str,
    key_added: str = "potential",
) -> None:
    """Compute the potential for all cells in an AnnData object.

    Args:
        adata (AnnData): Input data
        model (SpaceTime): Trained model
        omics_key (str): The omics key
        key_added (str): The obs key to store the potential. Defaults to "potential"

    """
    potential_fn = lambda x: model.potential.apply(model.params, x)
    adata.obs[key_added] = np.array(potential_fn(adata.obsm[omics_key]))


def compute_velocity(
    adata: AnnData,
    model: SpaceTime,
    omics_key: str,
    key_added: str = "X_velo",
) -> None:
    """Compute -grad J for all cells in an AnnData object, where J is the potential.

    Args:
        adata (AnnData): Input data
        model (SpaceTime): Trained model
        omics_key (str): The omics key
        key_added (str): The obsm key to store the potential. Defaults to "X_velo"

    """
    potential_fn = lambda x: model.potential.apply(model.params, x)
    velo_fn = lambda x: -jax.vmap(jax.grad(potential_fn))(x)
    adata.obsm[key_added] = np.array(velo_fn(adata.obsm[omics_key]))


def plot_velocity(
    adata: AnnData,
    omics_key: str,
    velocity_key: str,
    basis: str,
    **kwargs,
) -> None:
    """Plot velocity, as computed by `compute_velocity`

    Args:
        adata (AnnData): Input data
        omics_key (str): The obsm key for omics
        velocity_key (str): The obsm key for the velocity
    """
    import cellrank as cr

    vk = cr.kernels.VelocityKernel(
        adata, attr="obsm", xkey=omics_key, vkey=velocity_key
    ).compute_transition_matrix()
    vk.plot_projection(basis=basis, recompute=True, **kwargs)
