.. stories-jax documentation master file, created by
   sphinx-quickstart on Mon Jul  8 16:49:34 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

STORIES documentation
=====================


.. rst-class:: lead

   Learning cell fate landscapes from spatial transcriptomics using Fused Gromov-Wasserstein

----

.. toctree::
   :hidden:
   :maxdepth: 1
   :glob:
   :caption: Getting started

   vignettes/*

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: API

   model
   loss
   potentials
   steps
   tools

STORIES is a novel trajectory inference method for spatial transcriptomics data profiled at several time points, relying on Wasserstein gradient flow learning and Fused Gromov-Wasserstein. `Read the preprint here <https://www.biorxiv.org/content/xxxxx>`_ and `fork the code here <https://github.com/cantinilab/stories>`_!

.. image:: _static/fig1.png
   :alt: Explanatory figure

Install the package
-------------------

STORIES is implemented as a Python package seamlessly integrated within the scverse ecosystem. It relies on JAX for fast GPU computations and JIT compilation, and OTT for Optimal Transport computations.

via PyPI (recommended)
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install stories-jax

via GitHub (development version)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone git@github.com:cantinilab/stories.git
   pip install ./stories/

Getting started
---------------

STORIES takes as an input an AnnData object, where omics information and spatial coordinates are stored in `obsm`, and `obs` contains time information, and optionally a proliferation weight. Visit the **Getting started** and **API** sections for more documentation and tutorials.

You may download a preprocessed Stereo-seq demo dataset at https://figshare.com/s/xxxxxx.

.. code-block:: python

   import anndata as ad
   from stories import SpaceTime

   # Load data into a AnnData object.
   adata = ad.load_h5ad("my_data.h5ad")

   # Initialize and train the model.
   model = SpaceTime()
   model.fit()

   cccccc

Citation
--------

.. code-block:: bibtex

  xxxxx
