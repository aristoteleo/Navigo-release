# Tutorials

The easiest way to get familiar with Navigo is to work through the tutorials.
Tutorial notebooks live under `docs/tutorials/notebooks/`, while shared inputs are provided from `data/` and `checkpoints/`.

## Download Tutorial Assets

To run these notebooks, download the tutorial data and checkpoint bundles:

- Tutorial data bundle: [data.tar.gz](https://drive.google.com/file/d/1WelqIkm2y2TrQRMxYKprVp5evbwRk0_S/view?usp=sharing)
- Tutorial checkpoint bundle: [checkpoints.tar.gz](https://drive.google.com/file/d/1lPOn01Z87zZ9q9vCfszEOagoOGdiyUiV/view?usp=sharing)

From the repository root, you can download and extract them with:

```bash
curl -L 'https://drive.usercontent.google.com/download?id=1WelqIkm2y2TrQRMxYKprVp5evbwRk0_S&export=download&confirm=t' -o data.tar.gz
curl -L 'https://drive.usercontent.google.com/download?id=1lPOn01Z87zZ9q9vCfszEOagoOGdiyUiV&export=download&confirm=t' -o checkpoints.tar.gz
tar -xzf data.tar.gz
tar -xzf checkpoints.tar.gz
```

The tutorials expect top-level `data/` and `checkpoints/` directories in the same repository as `docs/` and `navigo/`.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Training Demo
:link: index_training_demo
:link-type: doc

A compact training example.
:::

:::{grid-item-card} Interpolation
:link: index_interpolation
:link-type: doc

Interpolation and denoising examples.
:::

:::{grid-item-card} GRN
:link: index_grn
:link-type: doc

GRN analysis examples.
:::

:::{grid-item-card} Knockout
:link: index_knockout
:link-type: doc

Knockout analysis examples.
:::

:::{grid-item-card} Reprogramming
:link: index_reprogramming
:link-type: doc

Reprogramming examples.
:::
::::

```{toctree}
:maxdepth: 2

index_training_demo
index_interpolation
index_grn
index_knockout
index_reprogramming

```
