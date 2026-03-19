# Repository Layout

The Navigo repository is organized so the package, tutorial workspace, and website source stay together.

## Top-Level Structure

- `navigo/`: installable Python package code.
- `docs/`: Sphinx/MyST source for the documentation website.
- `docs/tutorials/`: unified tutorial notebooks, helper resources, and outputs.
- `data/`: centralized tutorial datasets, CSV/JSON inputs, and reference tables.
- `checkpoints/`: centralized tutorial model checkpoints.
- `submission/`: submission-oriented entrypoints and helper scripts.

## Tutorial Source Of Truth

The repository now keeps a single in-repo tutorial tree under `docs/tutorials/`.

- `docs/tutorials/notebooks/` contains the notebooks rendered on the documentation website.
- `data/` and `checkpoints/` centralize the inputs used across tutorial sections.
- `docs/tutorials/resources/` stores helper scripts and legacy support files still needed by some notebooks.
- `docs/tutorials/outputs/` stores generated tutorial outputs.

## What Was Removed From The Old Split Layout

- The old embedded `dynamo/` source tree from the previous docs repository.
- Local documentation build output in `docs/_build/`.
- Local virtual-environment files and Git metadata from the old docs folder.
- Cache directories such as `__pycache__/`.
- The duplicate root-level `tutorials/` tree was moved out of the repo to `../tutorial_backup_navigo`.
