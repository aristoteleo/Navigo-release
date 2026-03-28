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
