# Introduction

**Navigo is a generative model for developmental single-cell transcriptomics** that learns a **continuous developmental vector field** from temporal scRNA-seq atlases by combining **population-level flow matching** with **molecular-level RNA kinetics**.

This learned field can then be queried for **trajectory mapping**, **temporal interpolation**, **denoising**, **perturbation simulation**, and **regulatory analysis**. In practice, Navigo supports applications ranging from cardiac GRN interpretation and zero-shot knockout analysis to fibroblast reprogramming.

The method is built around two core ideas:

1. Development should be modeled at the **molecular level** using RNA kinetics, so the learned velocity field is tied to transcription, splicing, and degradation rather than an arbitrary black-box transition function.
2. Development should also be modeled at the **population level**, where the system must learn coherent transitions between time points even though destructive single-cell measurements do not provide true cell-to-cell correspondences across time.

This section introduces the technical principles behind that design, with emphasis on **four key parts**:

- **Model principles**: how Navigo represents cell state and defines the developmental vector field through RNA kinetics.
- **Training**: how flow matching, reflow, and metacell aggregation make the model trainable on atlas-scale data.
- **Inference**: how the learned vector field is used for forward simulation, aligned interpolation, and denoising.
- **Perturbation and GRN Analysis**: how the same model supports in silico perturbation and Jacobian-based regulatory interpretation.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Model Principles
:link: model
:link-type: doc

How Navigo represents cell state, parameterizes the developmental vector field, and connects RNA kinetics to gene regulation.
:::

:::{grid-item-card} Training
:link: training
:link-type: doc

How flow matching and metacell aggregation make training feasible on atlas-scale data.
:::

:::{grid-item-card} Inference
:link: inference
:link-type: doc

How the learned field is used for forward simulation, aligned interpolation, and denoising at specific developmental times.
:::

:::{grid-item-card} Perturbation and GRN Analysis
:link: perturbation
:link-type: doc

How Navigo performs in silico perturbation, zero-shot knockout prediction, and Jacobian-based regulatory analysis.
:::
::::

```{toctree}
:maxdepth: 2

model
training
inference
perturbation
```
