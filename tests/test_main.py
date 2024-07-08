import stories
import stories.steps
import anndata as ad
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
key_0, key_1, key_2 = jax.random.split(key, 3)

x_0 = jax.random.normal(key_0, shape=(100, 20))
x_1 = jax.random.normal(key_1, shape=(100, 20))
x_2 = jax.random.normal(key_2, shape=(100, 20))

adata_0 = ad.AnnData(x_0)
adata_0.obs["time"] = 0
adata_1 = ad.AnnData(x_1)
adata_1.obs["time"] = 1
adata_2 = ad.AnnData(x_2)
adata_2.obs["time"] = 2

adata = ad.concat((adata_0, adata_1, adata_2))
adata.obsm["X_pca"] = adata.X.copy()
adata.obsm["spatial"] = adata.X[:, :2].copy()


def test_model_explicit():
    step = stories.steps.ExplicitStep()
    model = stories.SpaceTime()
    model.fit(
        adata,
        time_key="time",
        omics_key="X_pca",
        space_key="spatial",
        batch_size=50,
        max_iter=5,
    )


def test_model_linear():
    step = stories.steps.ExplicitStep()
    model = stories.SpaceTime(quadratic=False)
    model.fit(
        adata,
        time_key="time",
        omics_key="X_pca",
        space_key="spatial",
        batch_size=50,
        max_iter=5,
    )


def test_model_ICNN_implicit():
    step = stories.steps.ICNNImplicitStep()
    model = stories.SpaceTime()
    model.fit(
        adata,
        time_key="time",
        omics_key="X_pca",
        space_key="spatial",
        batch_size=50,
        max_iter=5,
    )


def test_model_monge_implicit():
    step = stories.steps.MongeImplicitStep()
    model = stories.SpaceTime()
    model.fit(
        adata,
        time_key="time",
        omics_key="X_pca",
        space_key="spatial",
        batch_size=50,
        max_iter=5,
    )