import jax.numpy as jnp
import jax


@jax.jit
def chamfer_distance(
    x_real: jax.Array,
    x_pred: jax.Array,
) -> jax.Array:
    """Compute the Chamfer distance between two point clouds.

    Args:
        x_real (jax.Array): The real distribution.
        x_pred (jax.Array): The predicted distribution.

    Returns:
        jax.Array: The Chamfer distance.
    """

    # Compute the distance matrix.
    dist_matrix = jnp.sum((x_real[:, None, :] - x_pred[None, :, :]) ** 2, axis=-1)

    # Compute the Chamfer distance.
    left_mean = jnp.mean(jnp.min(dist_matrix, axis=0))
    right_mean = jnp.mean(jnp.min(dist_matrix, axis=1))
    return jnp.array(left_mean + right_mean, float)


@jax.jit
def hausdorff_distance(
    x_real: jax.Array,
    x_pred: jax.Array,
) -> jax.Array:
    """Compute the Hausdorff distance between two point clouds.

    Args:
        x_real (jax.Array): The real distribution.
        x_pred (jax.Array): The predicted distribution.

    Returns:
        jax.Array: The Hausdorff distance.
    """

    # Compute the distance matrix.
    dist_matrix = jnp.sum((x_real[:, None, :] - x_pred[None, :, :]) ** 2, axis=-1)

    # Compute the Hausdorff distance.
    left_max = jnp.max(jnp.min(dist_matrix, axis=0))
    right_max = jnp.max(jnp.min(dist_matrix, axis=1))
    return jnp.array(left_max + right_max, float)
