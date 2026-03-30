# Tutorials

This directory contains the Navigo tutorial pages and notebooks.

## Structure

- `notebooks/`: section notebooks rendered by the documentation site.
- `outputs/`: generated outputs from notebook workflows.

## Root-Level Inputs

- `data/`: centralized datasets, CSV/JSON inputs, and reference tables used by the tutorials.
- `checkpoints/`: centralized `.pth` model checkpoints used by the tutorials.
- `navigo/`: reusable tutorial scripts that are kept directly in the main package when notebook-local code would be too repetitive.

Tutorial notebooks use repository assets from `data/` and `checkpoints/`. Reusable helper code is kept in top-level modules under `navigo/`.

## Download Tutorial Assets

To run these notebooks, download the two archive files below:

- Tutorial data bundle: [data.tar.gz](https://drive.google.com/file/d/1WelqIkm2y2TrQRMxYKprVp5evbwRk0_S/view?usp=sharing)
- Tutorial checkpoint bundle: [checkpoints.tar.gz](https://drive.google.com/file/d/1lPOn01Z87zZ9q9vCfszEOagoOGdiyUiV/view?usp=sharing)

From the repository root:

```bash
curl -L 'https://drive.usercontent.google.com/download?id=1WelqIkm2y2TrQRMxYKprVp5evbwRk0_S&export=download&confirm=t' -o data.tar.gz
curl -L 'https://drive.usercontent.google.com/download?id=1lPOn01Z87zZ9q9vCfszEOagoOGdiyUiV&export=download&confirm=t' -o checkpoints.tar.gz
tar -xzf data.tar.gz
tar -xzf checkpoints.tar.gz
```

After extraction, the repository should contain:

- `data/`
- `checkpoints/`

The notebooks assume those directories exist at the repository root.
