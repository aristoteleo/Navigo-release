# Installation

## Quick install

Navigo can be installed directly from this unified repository. We recommend using a virtual environment.


```bash
git clone <your-github-url> Navigo
cd Navigo
pip install -r requirements.txt
pip install -e .
```

## Build the documentation locally

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

## Repository notes

- Package source lives in `navigo/`.
- Tutorial notebooks live in `docs/tutorials/`.
- Shared tutorial data and checkpoints live in `data/` and `checkpoints/`.
- Website source lives in `docs/`.
